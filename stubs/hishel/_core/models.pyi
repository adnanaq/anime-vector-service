"""Type stubs for hishel._core.models module."""

import uuid
from typing import Any, AsyncIterator, Dict, Iterator, Optional, Union

from hishel._core._headers import Headers

class EntryMeta:
    """Cache entry metadata."""
    created_at: float
    deleted_at: Optional[float]

    def __init__(
        self, created_at: float, deleted_at: Optional[float] = None
    ) -> None: """
        Create cache entry metadata with creation and optional deletion timestamps.
        
        Parameters:
            created_at (float): Timestamp (seconds since the Unix epoch) when the entry was created.
            deleted_at (Optional[float]): Timestamp when the entry was deleted, or `None` if the entry has not been deleted.
        """
        ...

class Request:
    """HTTP request model."""
    method: str
    url: str
    headers: Union[Dict[str, str], Headers]
    stream: Optional[Iterator[bytes]]
    metadata: Dict[str, Any]

    def __init__(
        self,
        method: str,
        url: str,
        headers: Union[Dict[str, str], Headers],
        stream: Optional[Iterator[bytes]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None: """
        Initialize a Request model representing an HTTP request.
        
        Parameters:
            method (str): HTTP method (e.g., "GET", "POST").
            url (str): Request URL.
            headers (Union[Dict[str, str], Headers]): Request headers as a mapping or Headers object.
            stream (Optional[Iterator[bytes]]): Optional synchronous byte iterator for the request body.
            metadata (Optional[Dict[str, Any]]): Optional user-defined metadata associated with the request.
        """
        ...

class Response:
    """HTTP response model."""
    status_code: int
    headers: Union[Dict[str, str], Headers]
    stream: Optional[Union[Iterator[bytes], AsyncIterator[bytes]]]
    metadata: Dict[str, Any]

    def __init__(
        self,
        status_code: int,
        headers: Union[Dict[str, str], Headers],
        stream: Optional[Union[Iterator[bytes], AsyncIterator[bytes]]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None: """
        Initialize a Response model representing an HTTP response.
        
        Parameters:
            status_code (int): HTTP status code for the response.
            headers (Union[Dict[str, str], Headers]): Response headers as a mapping or a Headers instance.
            stream (Optional[Union[Iterator[bytes], AsyncIterator[bytes]]]): Optional iterator (synchronous or asynchronous) that yields the response body in bytes.
            metadata (Optional[Dict[str, Any]]): Optional dictionary of arbitrary metadata associated with the response.
        """
        ...

class Entry:
    """Cache entry model."""
    id: uuid.UUID
    request: Request
    response: Optional[Response]
    meta: EntryMeta
    cache_key: bytes

    def __init__(
        self,
        id: uuid.UUID,
        request: Request,
        response: Optional[Response],
        meta: EntryMeta,
        cache_key: bytes,
    ) -> None: """
        Initialize a cache Entry representing a stored HTTP request/response pair and its metadata.
        
        Parameters:
            id (uuid.UUID): Unique identifier for the cache entry.
            request (Request): The HTTP request associated with this entry.
            response (Optional[Response]): The HTTP response stored for the request, or None if absent.
            meta (EntryMeta): Metadata for the entry (creation and optional deletion timestamps).
            cache_key (bytes): Binary key used to index or look up the entry in the cache.
        """
        ...