"""Type stubs for hishel._core._headers module."""

from typing import Dict, List, Tuple

class Headers:
    """HTTP headers wrapper."""

    _headers: List[Tuple[str, str]]  # List of (key, value) tuples for multivalue support

    def __init__(self, headers: Dict[str, str]) -> None:
        """
        Create a Headers container populated from a mapping of header names to values.
        
        Parameters:
            headers (Dict[str, str]): Mapping of header names to header values; each key/value pair is added as a header entry.
        """
        ...

    def __getitem__(self, key: str) -> str:
        """
        Retrieve the value for the given header name.
        
        Returns:
            The header value associated with `key`.
        """
        ...

    def __setitem__(self, key: str, value: str) -> None:
        """
        Set the header named `key` to `value`, replacing any existing values for that header.
        
        Parameters:
            key (str): Header name.
            value (str): Header value to assign.
        """
        ...

    def get(self, key: str, default: str | None = None) -> str | None:
        """
        Return the first value for the given header key or the provided default if the header is not present.
        
        Parameters:
            key (str): Header name to look up.
            default (str | None): Value to return if the header is not found. Defaults to `None`.
        
        Returns:
            str | None: The first header value for `key` if present, otherwise `default`.
        """
        ...