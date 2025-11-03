"""
Async Redis storage backend for Hishel HTTP caching.

Implements AsyncBaseStorage interface for Redis-backed async HTTP response caching
with support for multi-agent concurrent access, service-specific TTLs, and
streaming responses.

This is the async parallel to SyncRedisStorage, using redis.asyncio for
aiohttp-based API helpers (AniList, Kitsu, AniDB).
"""

from __future__ import annotations

import time
import uuid
from typing import AsyncIterator, Callable, List, Optional, Union

# Import Hishel core types
from hishel._core._storages._async_base import AsyncBaseStorage
from hishel._core._storages._packing import pack, unpack
from hishel._core.models import Entry, EntryMeta, Request, Response
from redis.asyncio import Redis


class AsyncRedisStorage(AsyncBaseStorage):
    """
    Async Redis-backed storage for Hishel HTTP caching.

    Supports:
    - Multi-agent concurrent access
    - TTL-based expiration per service
    - Streaming response storage
    - Soft deletion with cleanup

    Redis Key Structure:
        cache:entry:{uuid}          → Hash with serialized Entry metadata
        cache:stream:{uuid}         → List of response stream chunks
        cache:key_index:{cache_key} → Set of entry UUIDs
    """

    _COMPLETE_CHUNK_MARKER = b"__STREAM_COMPLETE__"

    def __init__(
        self,
        *,
        client: Optional[Redis[bytes]] = None,
        redis_url: str = "redis://localhost:6379/0",
        default_ttl: Optional[float] = None,
        refresh_ttl_on_access: bool = True,
        key_prefix: str = "hishel_cache",
    ) -> None:
        """
        Configure AsyncRedisStorage and initialize or accept a Redis client, storing TTL and key-prefix settings.
        
        Parameters:
        	client (Optional[Redis[bytes]]): Existing async Redis client to use; if omitted, an internal client is created from `redis_url`.
        	redis_url (str): Connection URL used to create a client when `client` is not provided.
        	default_ttl (Optional[float]): Default time-to-live in seconds applied to entries when no per-request TTL is present.
        	refresh_ttl_on_access (bool): If True, refresh entry TTLs on access.
        	key_prefix (str): Prefix applied to all Redis keys for namespacing.
        """
        self._owns_client = client is None
        self.client = client or Redis.from_url(redis_url, decode_responses=False)
        self.default_ttl = default_ttl
        self.refresh_ttl_on_access = refresh_ttl_on_access
        self.key_prefix = key_prefix

        # Note: Connection test done lazily on first use

    def _make_key(self, key_type: str, identifier: str) -> str:
        """
        Builds a Redis key by joining the configured prefix, a key type, and an identifier.
        
        Parameters:
            key_type (str): Logical type component of the key (e.g., "entry", "stream", "key_index").
            identifier (str): Unique identifier component for the key (e.g., UUID or encoded cache key).
        
        Returns:
            str: Combined Redis key in the form "{prefix}:{key_type}:{identifier}".
        """
        return f"{self.key_prefix}:{key_type}:{identifier}"

    def _entry_key(self, entry_id: uuid.UUID) -> str:
        """
        Builds the Redis key used to store an entry's metadata and data.
        
        Returns:
            str: Redis key for the given entry ID.
        """
        return self._make_key("entry", str(entry_id))

    def _stream_key(self, entry_id: uuid.UUID) -> str:
        """
        Return the Redis key used to store response stream chunks for the given cache entry.
        
        Returns:
            stream_key (str): Redis key for the entry's stream list.
        """
        return self._make_key("stream", str(entry_id))

    def _index_key(self, cache_key: str) -> str:
        """
        Builds the Redis key for the set that tracks entry IDs associated with a cache key.
        
        Parameters:
            cache_key (str): Cache key to index; encoded as UTF-8 and hex-encoded before composing the Redis key.
        
        Returns:
            str: Redis key for the cache-key index set.
        """
        # Use hex encoding for cache key (bytes)
        cache_key_hex = (
            cache_key.encode("utf-8").hex()
            if isinstance(cache_key, str)
            else cache_key.hex()
        )
        return self._make_key("key_index", cache_key_hex)

    async def create_entry(
        self,
        request: Request,
        response: Response,
        key: str,
        id_: uuid.UUID | None = None,
    ) -> Entry:
        """
        Create and store a new cache entry for the given request/response and cache key.
        
        The response stream is wrapped so response chunks are persisted to Redis while being yielded. The entry metadata and serialized entry are stored in Redis, the entry ID is added to the per-key index, and configured TTLs (per-request or default) are applied to the entry, its stream list, and the index.
        
        Parameters:
            request: The original HTTP request associated with the entry.
            response: The HTTP response whose stream will be persisted and returned as part of the stored entry.
            key: Cache key used to index this entry.
            id_: Optional UUID to use for the entry; a new UUID is generated if not provided.
        
        Returns:
            The created Entry object with its response stream wrapped to persist chunks to Redis.
        """
        entry_id = id_ if id_ is not None else uuid.uuid4()
        entry_meta = EntryMeta(created_at=time.time())

        # Replace response stream with saving wrapper
        assert isinstance(response.stream, AsyncIterator)
        response_with_stream = Response(
            status_code=response.status_code,
            headers=response.headers,
            stream=self._save_stream(response.stream, entry_id),
            metadata=response.metadata,
        )

        # Create complete entry
        complete_entry = Entry(
            id=entry_id,
            request=request,
            response=response_with_stream,
            meta=entry_meta,
            cache_key=key.encode("utf-8"),
        )

        # Serialize entry
        entry_data = pack(complete_entry, kind="pair")

        # Store in Redis
        entry_key = self._entry_key(entry_id)
        index_key = self._index_key(key)

        # Use pipeline for atomic operations
        pipe = self.client.pipeline()

        # Store entry data
        pipe.hset(
            entry_key,
            mapping={
                b"data": entry_data,
                b"created_at": str(entry_meta.created_at).encode("utf-8"),
                b"cache_key": key.encode("utf-8"),
            },
        )

        # Add to cache key index
        pipe.sadd(index_key, str(entry_id).encode("utf-8"))

        # Set TTL if configured
        ttl = self._get_entry_ttl(request)
        if ttl is not None:
            pipe.expire(entry_key, int(ttl))
            pipe.expire(self._stream_key(entry_id), int(ttl))
            pipe.expire(index_key, int(ttl))

        await pipe.execute()

        return complete_entry

    async def get_entries(self, key: str) -> List[Entry]:
        """
        Return reconstructed cache entries associated with the given cache key.
        
        Retrieves entry IDs from the key's index, loads and deserializes each stored entry, skips missing/invalid/soft-deleted entries, and reattaches a Redis-backed async stream iterator to each entry's response so the response body can be streamed from Redis. If configured, refreshes the entry and stream TTL on access.
        
        Parameters:
            key (str): Cache key whose associated entries to retrieve.
        
        Returns:
            List[Entry]: Entries matching the cache key with their response.stream restored; entries that are missing, invalid, or soft-deleted are omitted.
        """
        index_key = self._index_key(key)
        entry_ids_bytes = await self.client.smembers(index_key)

        if not entry_ids_bytes:
            return []

        final_entries: List[Entry] = []

        for entry_id_bytes in entry_ids_bytes:
            entry_id = uuid.UUID(entry_id_bytes.decode("utf-8"))
            entry_key = self._entry_key(entry_id)

            # Get entry data
            entry_hash = await self.client.hgetall(entry_key)
            if not entry_hash or b"data" not in entry_hash:
                continue

            # Deserialize entry
            entry = unpack(entry_hash[b"data"], kind="pair")
            if not isinstance(entry, Entry) or entry.response is None:
                continue

            # Check if soft deleted
            if self.is_soft_deleted(entry):
                continue

            # Restore stream from Redis
            entry_with_stream = Entry(
                id=entry.id,
                request=entry.request,
                response=Response(
                    status_code=entry.response.status_code,
                    headers=entry.response.headers,
                    stream=self._stream_data_from_cache(entry_id),
                    metadata=entry.response.metadata,
                ),
                meta=entry.meta,
                cache_key=entry.cache_key,
            )

            # Refresh TTL on access if configured
            if self.refresh_ttl_on_access:
                ttl = self._get_entry_ttl(entry.request)
                if ttl is not None:
                    await self.client.expire(entry_key, int(ttl))
                    await self.client.expire(self._stream_key(entry_id), int(ttl))

            final_entries.append(entry_with_stream)

        return final_entries

    async def update_entry(
        self,
        id: uuid.UUID,
        new_entry: Union[Entry, Callable[[Entry], Entry]],
    ) -> Optional[Entry]:
        """
        Update an existing cache entry.
        
        Parameters:
        	id (uuid.UUID): UUID of the entry to update.
        	new_entry (Union[Entry, Callable[[Entry], Entry]]): Either a replacement Entry or a callable that takes the current Entry and returns an updated Entry.
        
        Returns:
        	Optional[Entry]: The updated Entry, or `None` if the entry was not found or invalid.
        
        Raises:
        	ValueError: If the updated Entry's id does not match the existing entry's id.
        """
        entry_key = self._entry_key(id)
        entry_hash = await self.client.hgetall(entry_key)

        if not entry_hash or b"data" not in entry_hash:
            return None

        # Deserialize current entry
        current_entry = unpack(entry_hash[b"data"], kind="pair")
        if not isinstance(current_entry, Entry) or current_entry.response is None:
            return None

        # Apply update
        if isinstance(new_entry, Entry):
            complete_entry = new_entry
        else:
            complete_entry = new_entry(current_entry)

        if current_entry.id != complete_entry.id:
            raise ValueError("Entry ID mismatch")

        # Serialize and store
        entry_data = pack(complete_entry, kind="pair")

        pipe = self.client.pipeline()
        pipe.hset(entry_key, b"data", entry_data)

        # Update cache key index if changed
        if current_entry.cache_key != complete_entry.cache_key:
            old_index_key = self._index_key(current_entry.cache_key.decode("utf-8"))
            new_index_key = self._index_key(complete_entry.cache_key.decode("utf-8"))

            pipe.srem(old_index_key, str(id).encode("utf-8"))
            pipe.sadd(new_index_key, str(id).encode("utf-8"))
            pipe.hset(entry_key, b"cache_key", complete_entry.cache_key)

        await pipe.execute()

        return complete_entry

    async def remove_entry(self, id: uuid.UUID) -> None:
        """
        Soft-delete the stored cache entry by setting its deleted timestamp in Redis.
        
        If the entry is missing or its stored data is not a valid Entry, the function does nothing.
        Parameters:
            id (uuid.UUID): UUID of the entry to mark as deleted.
        """
        entry_key = self._entry_key(id)
        entry_hash = await self.client.hgetall(entry_key)

        if not entry_hash or b"data" not in entry_hash:
            return

        # Deserialize entry
        entry = unpack(entry_hash[b"data"], kind="pair")
        if not isinstance(entry, Entry):
            return

        # Mark as deleted
        deleted_entry = self.mark_pair_as_deleted(entry)
        entry_data = pack(deleted_entry, kind="pair")

        # Update entry with deleted_at timestamp
        await self.client.hset(
            entry_key,
            mapping={
                b"data": entry_data,
                b"deleted_at": str(deleted_entry.meta.deleted_at).encode("utf-8"),
            },
        )

    async def close(self) -> None:
        """
        Close the owned Redis client connection.
        
        If this storage instance does not own the Redis client, no action is taken. Any exceptions raised while closing are suppressed.
        """
        if not self._owns_client:
            return
        try:
            await self.client.close()
        except Exception:
            pass

    async def _save_stream(
        self, stream: AsyncIterator[bytes], entry_id: uuid.UUID
    ) -> AsyncIterator[bytes]:
        """
        Persist a response byte stream to Redis by appending each chunk and yield the chunks unchanged.
        
        Parameters:
            stream (AsyncIterator[bytes]): Source response byte chunks.
            entry_id (uuid.UUID): Identifier used to derive the Redis stream key.
        
        Returns:
            AsyncIterator[bytes]: Yields the same byte chunks from the source as they are stored in Redis. A completion marker is appended to the Redis list when the source stream ends.
        """
        stream_key = self._stream_key(entry_id)

        async for chunk in stream:
            # Save chunk to Redis list
            await self.client.rpush(stream_key, chunk)
            yield chunk

        # Mark stream as complete
        await self.client.rpush(stream_key, self._COMPLETE_CHUNK_MARKER)

    async def _stream_data_from_cache(
        self, entry_id: uuid.UUID
    ) -> AsyncIterator[bytes]:
        """
        Yield stored response stream chunks for the given entry from Redis.
        
        Parameters:
            entry_id (uuid.UUID): Entry UUID used to locate the stream chunks in Redis.
        
        Returns:
            AsyncIterator[bytes]: Yields `bytes` chunks from the stored response stream until a completion marker is encountered.
        """
        stream_key = self._stream_key(entry_id)

        # Get all chunks at once (could optimize with cursor for large streams)
        chunks = await self.client.lrange(stream_key, 0, -1)

        for chunk in chunks:
            # Skip completion marker
            if chunk == self._COMPLETE_CHUNK_MARKER:
                break
            yield chunk

    def _get_entry_ttl(self, request: Request) -> Optional[float]:
        """
        Return the time-to-live (TTL) to apply for a cache entry based on the request.
        
        Parameters:
            request (Request): The HTTP request whose metadata may contain a per-request TTL under the key "hishel_ttl".
        
        Returns:
            Optional[float]: The TTL in seconds from request.metadata["hishel_ttl"] if it exists and is numeric, otherwise the storage instance's configured default_ttl.
        """
        # Check for per-request TTL
        if "hishel_ttl" in request.metadata:
            ttl_value = request.metadata["hishel_ttl"]
            if isinstance(ttl_value, (int, float)):
                return float(ttl_value)

        # Use default TTL
        return self.default_ttl

    async def cleanup_expired(self) -> int:
        """
        Remove soft-deleted cache entries that are eligible for permanent deletion.
        
        Scans stored entries and permanently deletes those that are marked as soft-deleted and determined safe to hard-delete by the storage's policy.
        
        Returns:
            int: Number of entries permanently removed.
        """
        cleaned = 0

        # Scan for all entry keys
        pattern = f"{self.key_prefix}:entry:*"
        cursor = 0

        while True:
            cursor, keys = await self.client.scan(cursor, match=pattern, count=100)

            for key in keys:
                entry_hash = await self.client.hgetall(key)
                if not entry_hash or b"data" not in entry_hash:
                    continue

                entry = unpack(entry_hash[b"data"], kind="pair")
                if not isinstance(entry, Entry):
                    continue

                # Hard delete if safe to do so
                if self.is_soft_deleted(entry) and self.is_safe_to_hard_delete(entry):
                    await self._hard_delete_entry(entry.id)
                    cleaned += 1

            if cursor == 0:
                break

        return cleaned

    async def _hard_delete_entry(self, entry_id: uuid.UUID) -> None:
        """
        Permanently remove a cached entry, its stored response stream, and any index membership from Redis.
        
        Parameters:
            entry_id (uuid.UUID): UUID of the entry to hard-delete; if the entry is indexed by a cache key, its ID is removed from that index and the entry's Redis hash and stream list are deleted.
        """
        entry_key = self._entry_key(entry_id)
        stream_key = self._stream_key(entry_id)

        # Get cache key before deleting
        entry_hash = await self.client.hgetall(entry_key)
        if entry_hash and b"cache_key" in entry_hash:
            cache_key = entry_hash[b"cache_key"].decode("utf-8")
            index_key = self._index_key(cache_key)

            # Remove from index
            await self.client.srem(index_key, str(entry_id).encode("utf-8"))

        # Delete entry and stream
        pipe = self.client.pipeline()
        pipe.delete(entry_key)
        pipe.delete(stream_key)
        await pipe.execute()