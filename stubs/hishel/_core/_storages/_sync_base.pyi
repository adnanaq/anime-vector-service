"""Type stubs for hishel._core._storages._sync_base module."""

import abc
import uuid
from typing import Callable, List, Optional, Union

# Import from parent module
from hishel import Entry, Request, Response

class SyncBaseStorage(abc.ABC):
    """Base class for synchronous storage backends."""

    @abc.abstractmethod
    def create_entry(
        self,
        request: Request,
        response: Response,
        key: str,
        id_: Optional[uuid.UUID] = None,
    ) -> Entry:
        """
        Create and store a cache entry for the given key.
        
        Parameters:
            request (Request): The original request that produced the response.
            response (Response): The response to store with the entry.
            key (str): Cache key under which the entry will be stored.
            id_ (Optional[uuid.UUID]): Optional UUID to assign to the new entry; if omitted, an implementation may generate one.
        
        Returns:
            Entry: The stored cache entry.
        """
        ...

    @abc.abstractmethod
    def get_entries(self, key: str) -> List[Entry]:
        """
        Retrieve all cache entries associated with the given key.
        
        Returns:
            List[Entry]: A list of `Entry` objects matching the provided key; an empty list if no entries exist.
        """
        ...

    @abc.abstractmethod
    def update_entry(
        self,
        id: uuid.UUID,
        new_entry: Union[Entry, Callable[[Entry], Entry]],
    ) -> Optional[Entry]:
        """
        Update an existing cache entry identified by its UUID.
        
        Parameters:
            id (uuid.UUID): UUID of the entry to update.
            new_entry (Union[Entry, Callable[[Entry], Entry]]): Either an `Entry` to replace the existing entry or a callable that takes the current `Entry` and returns an updated `Entry`.
        
        Returns:
            Optional[Entry]: The updated `Entry` if the update succeeded, `None` if no entry with the given `id` was found.
        """
        ...

    @abc.abstractmethod
    def remove_entry(self, id: uuid.UUID) -> None:
        """
        Mark a stored entry as removed without permanently deleting it.
        
        Parameters:
            id (uuid.UUID): UUID of the entry to mark as soft deleted.
        """
        ...

    def close(self) -> None:
        """
        Perform any necessary cleanup for the storage backend.
        
        Implementations may override this to close connections, flush buffers, or release other resources; the base implementation is a no-op.
        """
        ...

    def is_soft_deleted(self, entry: Entry) -> bool:
        """
        Determine whether the given cache entry has been marked as soft deleted.
        
        Parameters:
            entry (Entry): The cache entry to inspect.
        
        Returns:
            bool: `True` if the entry is marked as soft deleted, `False` otherwise.
        """
        ...

    def is_safe_to_hard_delete(self, entry: Entry) -> bool:
        """
        Determine whether the given cache entry is eligible for permanent (hard) deletion.
        
        Returns:
            `true` if the entry can be permanently removed, `false` otherwise.
        """
        ...

    def mark_pair_as_deleted(self, entry: Entry) -> Entry:
        """
        Mark the given entry as soft deleted.
        
        Returns:
            entry (Entry): The entry updated to reflect a soft delete.
        """
        ...