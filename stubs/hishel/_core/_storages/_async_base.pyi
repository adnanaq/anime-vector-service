"""Type stubs for hishel._core._storages._async_base module."""

import abc
import uuid
from typing import Callable, List, Optional, Union

# Import from parent module
from hishel import Entry, Request, Response

class AsyncBaseStorage(abc.ABC):
    """Base class for asynchronous storage backends."""

    @abc.abstractmethod
    async def create_entry(
        self,
        request: Request,
        response: Response,
        key: str,
        id_: Optional[uuid.UUID] = None,
    ) -> Entry:
        """Create and store a new cache entry."""
        ...

    @abc.abstractmethod
    async def get_entries(self, key: str) -> List[Entry]:
        """Retrieve all entries for a given cache key."""
        ...

    @abc.abstractmethod
    async def update_entry(
        self,
        id: uuid.UUID,
        new_entry: Union[Entry, Callable[[Entry], Entry]],
    ) -> Optional[Entry]:
        """
        Update an existing cache entry identified by its UUID.
        
        Parameters:
            id (uuid.UUID): UUID of the entry to update.
            new_entry (Union[Entry, Callable[[Entry], Entry]]): Either a replacement Entry or a callable that receives the current Entry and returns the updated Entry.
        
        Returns:
            Optional[Entry]: The updated Entry if the entry existed and was updated, `None` if no entry with the given UUID was found.
        """
        ...

    @abc.abstractmethod
    async def remove_entry(self, id: uuid.UUID) -> None:
        """
        Mark the cache entry identified by `id` as removed (soft delete).
        
        Parameters:
            id (uuid.UUID): UUID of the entry to soft-delete.
        """
        ...

    async def close(self) -> None:
        """
        Perform backend-specific cleanup such as closing connections or releasing resources.
        
        Implementations may override this optional hook; the base implementation performs no action.
        """
        ...

    def is_soft_deleted(self, entry: Entry) -> bool:
        """
        Determine whether the given cache entry has been marked as soft deleted.
        
        Parameters:
            entry (Entry): Cache entry to inspect.
        
        Returns:
            bool: `True` if the entry is marked as soft deleted, `False` otherwise.
        """
        ...

    def is_safe_to_hard_delete(self, entry: Entry) -> bool:
        """
        Determine whether the given cache entry is eligible for permanent (hard) deletion.
        
        Parameters:
            entry (Entry): The cache entry to evaluate.
        
        Returns:
            `true` if the entry may be permanently removed, `false` otherwise.
        """
        ...

    def mark_pair_as_deleted(self, entry: Entry) -> Entry:
        """
        Mark a cache entry as soft deleted.
        
        Parameters:
            entry (Entry): The entry to mark as soft deleted.
        
        Returns:
            Entry: The entry with soft-delete metadata applied.
        """
        ...