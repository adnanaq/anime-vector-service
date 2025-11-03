"""
aiohttp caching adapter using Hishel's AsyncRedisStorage.

Provides a drop-in replacement for aiohttp.ClientSession that adds
HTTP caching via Redis backend.

**Note**: This is a simplified implementation that directly uses
AsyncRedisStorage. For async helpers (AniList, Kitsu, AniDB), we use
the cached session which wraps the original aiohttp.ClientSession.
"""

import hashlib
from types import TracebackType
from typing import Any, AsyncIterator, Dict, Optional, Type

import aiohttp
from hishel._core._headers import Headers
from hishel._core._storages._async_base import AsyncBaseStorage
from hishel._core.models import Request, Response
from multidict import CIMultiDict, CIMultiDictProxy
from yarl import URL


class _CachedResponse:
    """
    Mock aiohttp.ClientResponse for cached data.

    Provides minimal interface needed by enrichment helpers.
    """

    def __init__(
        self,
        status: int,
        headers: Dict[str, str],
        body: bytes,
        url: str,
        from_cache: bool = False,
    ) -> None:
        """
        Create a cached response object that mimics the minimal aiohttp.ClientResponse interface.
        
        Parameters:
            status: HTTP status code of the response.
            headers: Mapping of response header names to values (case-insensitive access will be provided).
            body: Raw response body bytes.
            url: Request URL returned with the response.
            from_cache: `True` if the response was served from the cache, `False` otherwise.
        """
        self.status = status
        # Create CIMultiDict for case-insensitive header access
        headers_multidict = CIMultiDict(headers)
        self.headers = CIMultiDictProxy(headers_multidict)
        self._body = body
        self.url = URL(url)
        self._released = False
        self.from_cache = from_cache

    async def read(self) -> bytes:
        """
        Return the cached response body.
        
        Returns:
            bytes: The raw response body stored for this cached response.
        """
        return self._body

    async def text(self, encoding: str = "utf-8") -> str:
        """
        Return the response body decoded using the specified encoding.
        
        Parameters:
            encoding (str): Character encoding to use for decoding the stored response body bytes. Defaults to "utf-8".
        
        Returns:
            str: Decoded text of the response body.
        """
        return self._body.decode(encoding)

    async def json(self, **kwargs: Any) -> Any:
        """
        Parse the stored response body as JSON (decoded using UTF-8).
        
        Returns:
            Any: The deserialized JSON value.
        """
        import json

        return json.loads(self._body.decode("utf-8"))

    def release(self) -> None:
        """
        Mark the cached response as released.
        
        Sets an internal flag indicating the response has been released; does not perform network or resource cleanup.
        """
        self._released = True

    def raise_for_status(self) -> None:
        """
        Raise ValueError when the response status indicates an HTTP error.
        
        Raises:
            ValueError: If the response status is between 400 and 599 (inclusive).
        """
        if 400 <= self.status < 600:
            raise ValueError(f"HTTP {self.status} error")

    async def __aenter__(self) -> "_CachedResponse":
        """
        Enter the async context and return this cached response instance.
        
        Returns:
            _CachedResponse: The same cached response instance to be used within the `async with` block.
        """
        return self

    async def __aexit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> None:
        """
        Exit the async context and release the underlying response.
        
        Releases the acquired response (if any) and performs cleanup. Does not suppress any exception raised in the context block.
        """
        self.release()


class _CachedRequestContextManager:
    """
    Async context manager for cached HTTP requests.

    This wrapper allows using `async with session.post()` syntax
    by implementing the async context manager protocol.
    """

    def __init__(
        self,
        coro: Any,  # Coroutine that returns aiohttp.ClientResponse
        session: "CachedAiohttpSession",
        method: str,
        url: str,
        kwargs: Dict[str, Any],
    ) -> None:
        """
        Create an async context manager that defers execution of a coroutine returning an aiohttp.ClientResponse for use with `async with`.
        
        Parameters:
            coro (Coroutine): Coroutine that, when awaited, yields an `aiohttp.ClientResponse`.
            session (CachedAiohttpSession): Parent cached session that created this context manager.
            method (str): HTTP method for the pending request (e.g., "GET", "POST").
            url (str): Request URL.
            kwargs (Dict[str, Any]): Keyword arguments that will be passed to the request when the coroutine is executed.
        """
        self._coro = coro
        self._session = session
        self._method = method
        self._url = url
        self._kwargs = kwargs
        self._response: Optional[aiohttp.ClientResponse] = None

    async def __aenter__(self) -> aiohttp.ClientResponse:
        """
        Execute the pending request coroutine and return the resulting response for use in an async with block.
        
        Returns:
            aiohttp.ClientResponse: The completed response object from the awaited request.
        """
        self._response = await self._coro
        return self._response

    async def __aexit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> None:
        """
        Exit the async context and release the underlying response if one was obtained.
        
        Releases the awaited response to free associated resources; does nothing if no response was acquired.
        """
        if self._response is not None:
            self._response.release()


class CachedAiohttpSession:
    """
    Wrapper around aiohttp.ClientSession that adds HTTP caching.

    Uses Hishel's AsyncRedisStorage for cache backend with service-specific TTLs.
    """

    def __init__(
        self,
        storage: AsyncBaseStorage,
        session: Optional[aiohttp.ClientSession] = None,
        **session_kwargs: Any,
    ) -> None:
        """
        Create a cached aiohttp session wrapper that uses the provided AsyncBaseStorage backend.
        
        Parameters:
            storage: AsyncBaseStorage used to read and write cached entries (e.g., AsyncRedisStorage).
            session: Optional existing aiohttp.ClientSession to wrap; if omitted, a new ClientSession is created using `session_kwargs`.
            **session_kwargs: Additional keyword arguments forwarded to aiohttp.ClientSession when a new session is created.
        """
        self.storage = storage
        self.session = session or aiohttp.ClientSession(**session_kwargs)

    def get(self, url: str, **kwargs: Any) -> _CachedRequestContextManager:
        """
        Create an async context manager for a GET request that uses the session's cache.
        
        Parameters:
            url (str): Request URL.
            **kwargs: Arguments forwarded to the underlying aiohttp request; values affecting the request body (e.g., `json`, `data`) are included when generating the cache key.
        
        Returns:
            _CachedRequestContextManager: An async context manager that yields a response-like object (either a cached response or a live aiohttp response) compatible with `async with` usage.
        """
        coro = self._request("GET", url, **kwargs)
        return _CachedRequestContextManager(coro, self, "GET", url, kwargs)

    def post(self, url: str, **kwargs: Any) -> _CachedRequestContextManager:
        """
        Create an async context manager that performs a POST request and returns a cached or live response.
        
        Parameters:
            url (str): Request URL.
            **kwargs: Additional aiohttp request arguments forwarded to the underlying session; these are also considered when generating the cache key (e.g., `json`, `data`, headers).
        
        Returns:
            _CachedRequestContextManager: Context manager that yields a response-like object (cached when available).
        """
        coro = self._request("POST", url, **kwargs)
        return _CachedRequestContextManager(coro, self, "POST", url, kwargs)

    async def _request(
        self, method: str, url: str, **kwargs: Any
    ) -> Any:  # Returns aiohttp.ClientResponse or _CachedResponse
        """
        Perform an HTTP request and return a cached response when available.
        
        If a cache entry exists for the request key, return a _CachedResponse constructed from stored data without making a network call. On a cache miss, perform the request via the underlying aiohttp session; if the response status is less than 400, store the response body in the cache and return a _CachedResponse wrapping the fresh response.
        
        Parameters:
            method (str): HTTP method (e.g., "GET", "POST").
            url (str): Request URL.
            **kwargs: Additional aiohttp request arguments; these are considered when generating the cache key and are forwarded to the underlying session.request call.
        
        Returns:
            _CachedResponse: The response wrapper; `from_cache` is `True` for cached responses and `False` for responses fetched from the network.
        """
        # Generate cache key
        cache_key = self._generate_cache_key(method, url, kwargs)

        # Check cache
        entries = await self.storage.get_entries(cache_key)
        if entries:
            # Cache hit - return cached response WITHOUT making HTTP request
            entry = entries[0]  # Get most recent entry
            if entry.response:
                # Read all chunks from cached stream
                body_chunks: list[bytes] = []
                if entry.response.stream:
                    # Handle both async and sync iterators
                    stream = entry.response.stream
                    if hasattr(stream, "__aiter__"):
                        # AsyncIterator
                        async for chunk in stream:
                            body_chunks.append(chunk)
                    else:
                        # Iterator - convert to async
                        for chunk in stream:
                            body_chunks.append(chunk)
                body = b"".join(body_chunks)

                # Extract headers - convert to simple dict
                if isinstance(entry.response.headers, Headers):
                    # Headers._headers is list-based with nested structure
                    # Convert to dict (taking last value for duplicate keys)
                    headers_dict = {}
                    try:
                        # Try direct iteration first
                        for item in entry.response.headers._headers:
                            if isinstance(item, (list, tuple)) and len(item) >= 2:
                                key, value = item[0], item[1]
                                headers_dict[key] = value
                    except (ValueError, TypeError):
                        # Fallback: headers might be already a dict
                        headers_dict = dict(entry.response.headers._headers)
                else:
                    headers_dict = entry.response.headers

                # Return cached response
                return _CachedResponse(
                    status=entry.response.status_code,
                    headers=headers_dict,
                    body=body,
                    url=url,
                    from_cache=True,
                )

        # Cache miss - make actual HTTP request
        response = await self.session.request(method, url, **kwargs)

        # Read response body to cache it
        body = await response.read()

        # Only cache successful responses (2xx and 3xx)
        # NEVER cache error responses (4xx, 5xx) as they are temporary
        if response.status < 400:
            await self._store_response_with_body(
                method, url, response, cache_key, kwargs, body
            )

        # Return cached response wrapper (allows multiple reads)
        return _CachedResponse(
            status=response.status,
            headers=dict(response.headers),
            body=body,
            url=str(response.url),
            from_cache=False,
        )

    def _generate_cache_key(self, method: str, url: str, kwargs: Dict[str, Any]) -> str:
        """
        Create a stable cache key for an HTTP request using the method, URL, and request body when present.
        
        Parameters:
            method (str): HTTP method (e.g., "GET", "POST").
            url (str): Request URL.
            kwargs (Dict[str, Any]): Request keyword arguments; if `json` or string/bytes `data` is present it will be included in the key.
        
        Returns:
            str: A cache key string combining the HTTP method and a hash of the relevant request parts.
        """
        # Include method, URL, and body (for POST) in cache key
        key_parts = [method, url]

        # Include request body for POST/PUT requests
        if "json" in kwargs:
            import json

            body = json.dumps(kwargs["json"], sort_keys=True)
            key_parts.append(body)
        elif "data" in kwargs:
            # For form data, include in key
            data = kwargs["data"]
            if isinstance(data, (str, bytes)):
                key_parts.append(str(data))

        # Hash to create stable key
        key_string = ":".join(key_parts)
        key_hash = hashlib.sha256(key_string.encode()).hexdigest()
        return f"{method}:{key_hash}"

    async def _store_response_with_body(
        self,
        method: str,
        url: str,
        response: aiohttp.ClientResponse,
        cache_key: str,
        request_kwargs: Dict[str, Any],
        body: bytes,
    ) -> None:
        """
        Store an HTTP response and its pre-read body into the configured cache storage.
        
        Stores a Hishel-compatible entry built from the provided response and request data under the given cache key and consumes the stored response stream so the storage backend persists the body.
        
        Parameters:
            method (str): HTTP method used for the request.
            url (str): Request URL.
            response (aiohttp.ClientResponse): Live aiohttp response whose headers and status will be recorded.
            cache_key (str): Key under which the entry will be stored.
            request_kwargs (Dict[str, Any]): Original request keyword arguments; `metadata` from this dict is included in the stored request.
            body (bytes): Pre-read response body to be stored and served from cache.
        """
        # Convert aiohttp response to Hishel Entry
        hishel_request = Request(
            method=method,
            url=str(response.url),
            headers=Headers(dict(response.request_info.headers)),
            stream=None,
            metadata=request_kwargs.get("metadata", {}),
        )

        # Create async iterator factory for body (can be called multiple times)
        def body_stream_factory() -> AsyncIterator[bytes]:
            """
            Create an asynchronous byte-stream iterator that yields the captured response body exactly once.
            
            Returns:
                AsyncIterator[bytes]: An async iterator that yields the pre-read response body as a single `bytes` chunk.
            """
            async def body_stream() -> AsyncIterator[bytes]:
                yield body

            return body_stream()

        hishel_response = Response(
            status_code=response.status,
            headers=Headers(dict(response.headers)),
            stream=body_stream_factory(),
            metadata={},
        )

        # Store in cache - this returns an Entry with wrapped stream
        entry = await self.storage.create_entry(
            hishel_request, hishel_response, cache_key
        )

        # IMPORTANT: Consume the wrapped stream to actually save to Redis
        # The storage wraps the stream with _save_stream which saves chunks as they're read
        if entry.response and entry.response.stream:
            stream = entry.response.stream
            if hasattr(stream, "__aiter__"):
                async for _ in stream:
                    pass  # Just consume, data already yielded from body_stream()
            else:
                for _ in stream:
                    pass  # Sync iterator fallback

    async def close(self) -> None:
        """Close the session and storage."""
        await self.session.close()
        await self.storage.close()

    async def __aenter__(self) -> "CachedAiohttpSession":
        """
        Enter the async context and return the cached aiohttp session.
        
        Returns:
            CachedAiohttpSession: The session instance.
        """
        return self

    async def __aexit__(self, *args: Any) -> None:
        """
        Exit the async context by closing the session and its storage.
        
        This awaits self.close(), ensuring the underlying aiohttp session and the cache storage are closed.
        """
        await self.close()