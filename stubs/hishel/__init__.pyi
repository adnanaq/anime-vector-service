"""Type stubs for hishel HTTP caching library (v1.0)."""

import uuid
from typing import Any, Callable, List, Optional, Union

from hishel._core._storages._async_base import AsyncBaseStorage as AsyncBaseStorage

# Re-export storage from _core._storages
from hishel._core._storages._sync_base import SyncBaseStorage as SyncBaseStorage

# Re-export core models from _core.models
from hishel._core.models import Entry as Entry
from hishel._core.models import EntryMeta as EntryMeta
from hishel._core.models import Request as Request
from hishel._core.models import Response as Response

class CacheOptions:
    """Cache options for Hishel 1.0."""

    def __init__(self) -> None: """
Initialize a CacheOptions instance used to hold configuration for caching behavior.
"""
...

class SyncSqliteStorage(SyncBaseStorage):
    """Synchronous SQLite storage."""

    def __init__(
        self,
        *,
        connection: Optional[Any] = None,
        database_path: str = "hishel_cache.db",
        default_ttl: Optional[float] = None,
        refresh_ttl_on_access: bool = True,
    ) -> None: """
        Initialize the SQLite storage backend for cache entries.
        
        Parameters:
            connection (Optional[Any]): Existing DB connection to use; if omitted the backend will manage its own connection.
            database_path (str): Filesystem path to the SQLite database file.
            default_ttl (Optional[float]): Default time-to-live in seconds for new entries when not specified.
            refresh_ttl_on_access (bool): If True, refresh an entry's TTL when it is accessed.
        """
        ...
    def create_entry(
        self,
        request: Request,
        response: Response,
        key: str,
        id_: Optional[uuid.UUID] = None,
    ) -> Entry: """
        Create a cache entry that associates the given request and response with a cache key.
        
        Parameters:
            request (Request): The original request to be cached.
            response (Response): The response to store for the request.
            key (str): Cache key under which the entry will be stored.
            id_ (Optional[uuid.UUID]): Optional explicit identifier for the entry; a new id will be generated if omitted.
        
        Returns:
            Entry: The created cache entry.
        """
        ...
    def get_entries(self, key: str) -> List[Entry]: """
Retrieve all cache entries associated with the given cache key.

Parameters:
    key (str): Cache key that identifies the stored entries.

Returns:
    List[Entry]: Entries stored under the specified key (may be empty).
"""
...
    def update_entry(
        self,
        id: uuid.UUID,
        new_entry: Union[Entry, Callable[[Entry], Entry]],
    ) -> Optional[Entry]: """
        Update the cache entry identified by `id` using either a replacement entry or a mapper function.
        
        Parameters:
            id (uuid.UUID): Identifier of the entry to update.
            new_entry (Entry | Callable[[Entry], Entry]): Either a replacement Entry or a function that receives the current Entry and returns the updated Entry.
        
        Returns:
            Optional[Entry]: The updated Entry if the entry existed and was updated, `None` if no entry with `id` was found.
        """
        ...
    def remove_entry(self, id: uuid.UUID) -> None: """
Remove the cache entry identified by the given UUID.
"""
...

class AsyncSqliteStorage(AsyncBaseStorage):
    """Asynchronous SQLite storage."""

    def __init__(
        self,
        *,
        connection: Optional[Any] = None,
        database_path: str = "hishel_cache.db",
        default_ttl: Optional[float] = None,
        refresh_ttl_on_access: bool = True,
    ) -> None: """
        Initialize the SQLite storage backend for cache entries.
        
        Parameters:
            connection (Optional[Any]): Existing DB connection to use; if omitted the backend will manage its own connection.
            database_path (str): Filesystem path to the SQLite database file.
            default_ttl (Optional[float]): Default time-to-live in seconds for new entries when not specified.
            refresh_ttl_on_access (bool): If True, refresh an entry's TTL when it is accessed.
        """
        ...
    async def create_entry(
        self,
        request: Request,
        response: Response,
        key: str,
        id_: Optional[uuid.UUID] = None,
    ) -> Entry: """
        Create a cache entry for the given request and response under the specified key.
        
        Parameters:
            request (Request): The original request to be cached.
            response (Response): The response to store in the cache.
            key (str): The cache key grouping one or more entries.
            id_ (Optional[uuid.UUID]): Optional explicit entry identifier; if omitted, an identifier will be assigned.
        
        Returns:
            Entry: The created cache entry representing the stored request/response.
        """
        ...
    async def get_entries(self, key: str) -> List[Entry]: """
Retrieve all cache entries stored under the given cache key.

Parameters:
    key (str): Cache key used to look up stored entries.

Returns:
    List[Entry]: List of cache entries associated with `key`.
"""
...
    async def update_entry(
        self,
        id: uuid.UUID,
        new_entry: Union[Entry, Callable[[Entry], Entry]],
    ) -> Optional[Entry]: """
        Update an existing cache entry identified by `id`.
        
        Parameters:
            id (uuid.UUID): Identifier of the entry to update.
            new_entry (Entry | Callable[[Entry], Entry]): Either a replacement Entry or a function that takes the current Entry and returns the updated Entry.
        
        Returns:
            Optional[Entry]: The updated Entry if an entry with `id` existed and was updated, `None` otherwise.
        """
        ...
    async def remove_entry(self, id: uuid.UUID) -> None: """
Remove the cache entry identified by the given UUID.

Parameters:
    id (uuid.UUID): Unique identifier of the entry to remove.
"""
...

class CachePolicy:
    """Cache policy for Hishel 1.0."""

    ...

class SyncCacheProxy:
    """Synchronous cache proxy for requests library."""

    def __init__(
        self,
        request_sender: Any,
        storage: Optional[SyncBaseStorage] = None,
        policy: Optional[CachePolicy] = None,
    ) -> None: """
        Initialize the synchronous cache proxy that intercepts requests and serves and stores cached responses.
        
        Parameters:
            request_sender (Any): Callable or object used to perform the underlying HTTP request when a cache miss occurs.
            storage (Optional[SyncBaseStorage]): Storage backend for cache entries; if omitted, a default backend may be used.
            policy (Optional[CachePolicy]): Cache policy governing hit/miss decisions, TTLs, and refresh behavior.
        """
        ...
    def __call__(self, *args: Any, **kwargs: Any) -> Any: """
Invoke the cache proxy to process a request and return the proxied handler's result.

Returns:
    The proxied handler's result (typically an HTTP response).
"""
...

class AsyncCacheProxy:
    """Asynchronous cache proxy for aiohttp/httpx."""

    def __init__(
        self,
        client: Any,
        storage: AsyncBaseStorage,
        options: Optional[CacheOptions] = None,
    ) -> None: """
        Initialize an asynchronous cache proxy that wraps an HTTP client with an asynchronous storage backend.
        
        Parameters:
            client (Any): HTTP client instance used to perform requests.
            storage (AsyncBaseStorage): Storage backend used to persist and retrieve cache entries.
            options (Optional[CacheOptions]): Optional cache configuration. If omitted, defaults are used.
        """
        ...