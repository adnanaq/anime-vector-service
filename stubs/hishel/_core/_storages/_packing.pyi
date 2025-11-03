"""Type stubs for hishel._core._storages._packing module."""

from typing import Any, Literal

def pack(obj: Any, kind: Literal["pair"] = "pair") -> bytes:
    """
    Serialize a cache entry to bytes using MessagePack.
    
    Parameters:
        obj: Object to serialize; typically an Entry.
        kind: Serialization kind; supported value: "pair" (default).
    
    Returns:
        Serialized bytes.
    """
    ...

def unpack(data: bytes, kind: Literal["pair"] = "pair") -> Any:
    """
    Deserialize cache entry bytes into the original Python object using MessagePack.
    
    Parameters:
        data (bytes): Serialized bytes representing a cache entry.
        kind (Literal["pair"]): Serialization format variant; defaults to "pair".
    
    Returns:
        Any: Deserialized object (typically an Entry).
    """
    ...