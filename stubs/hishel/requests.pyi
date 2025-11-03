"""Type stubs for hishel.requests module (v1.0)."""

from typing import Optional

from requests.adapters import HTTPAdapter

from . import SyncBaseStorage

class CacheAdapter(HTTPAdapter):
    """Cache adapter for requests library."""

    def __init__(
        self, storage: Optional[SyncBaseStorage] = None, **kwargs: object
    ) -> None: """
        Initialize the CacheAdapter with an optional synchronous storage backend and adapter options.
        
        Parameters:
        	storage (Optional[SyncBaseStorage]): Storage backend used to persist cached responses. If `None`, no external storage is provided.
        	**kwargs (object): Additional keyword arguments forwarded to the underlying HTTPAdapter initializer.
        """
        ...