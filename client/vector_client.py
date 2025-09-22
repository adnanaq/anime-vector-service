"""
Vector Service Client for interacting with the Anime Vector Service.

This client provides a clean interface for applications to interact with
the vector database without directly coupling to the implementation.
"""

import asyncio
import json
import logging
from typing import Any, Dict, List, Optional

import aiohttp

logger = logging.getLogger(__name__)


class VectorServiceError(Exception):
    """Exception raised by vector service operations."""

    pass


class VectorServiceClient:
    """
    Client for interacting with the Anime Vector Service.

    Provides async methods for all vector operations including search,
    similarity, and administrative functions.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8002",
        timeout: int = 30,
        api_key: Optional[str] = None,
    ):
        """
        Initialize vector service client.

        Args:
            base_url: Base URL of the vector service
            timeout: Request timeout in seconds
            api_key: Optional API key for authentication
        """
        self.base_url = base_url.rstrip("/")
        self.timeout = aiohttp.ClientTimeout(total=timeout)
        self.api_key = api_key
        self._session = None

    async def __aenter__(self):
        """Async context manager entry."""
        await self._ensure_session()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self._session:
            await self._session.close()
            self._session = None

    async def _ensure_session(self):
        """Ensure HTTP session is created."""
        if not self._session:
            headers = {}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"

            self._session = aiohttp.ClientSession(timeout=self.timeout, headers=headers)

    async def _request(
        self,
        method: str,
        endpoint: str,
        json_data: Optional[Dict] = None,
        params: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """
        Make HTTP request to vector service.

        Args:
            method: HTTP method
            endpoint: API endpoint
            json_data: JSON data for request body
            params: Query parameters

        Returns:
            Response JSON data

        Raises:
            VectorServiceError: If request fails
        """
        await self._ensure_session()

        url = f"{self.base_url}{endpoint}"

        try:
            async with self._session.request(
                method, url, json=json_data, params=params
            ) as response:

                if response.status >= 400:
                    error_text = await response.text()
                    try:
                        error_data = await response.json()
                        error_message = error_data.get("detail", error_text)
                    except:
                        error_message = error_text

                    raise VectorServiceError(
                        f"Vector service error ({response.status}): {error_message}"
                    )

                return await response.json()

        except aiohttp.ClientError as e:
            raise VectorServiceError(f"Connection error: {str(e)}")
        except asyncio.TimeoutError:
            raise VectorServiceError("Request timeout")

    # Health and Status Methods

    async def health_check(self) -> Dict[str, Any]:
        """Check service health."""
        return await self._request("GET", "/health")

    async def get_stats(self) -> Dict[str, Any]:
        """Get database statistics."""
        return await self._request("GET", "/api/v1/admin/stats")

    # Search Methods

    async def search(
        self, query: str, limit: int = 20, filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Perform semantic text search.

        Args:
            query: Search query text
            limit: Maximum number of results
            filters: Optional search filters

        Returns:
            List of search results
        """
        request_data = {"query": query, "limit": limit}
        if filters:
            request_data["filters"] = filters

        response = await self._request("POST", "/api/v1/search", json_data=request_data)
        return response.get("results", [])

    async def search_by_image(
        self, image_data: str, limit: int = 10, use_hybrid_search: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Perform image-based search.

        Args:
            image_data: Base64 encoded image data
            limit: Maximum number of results
            use_hybrid_search: Use hybrid search for better accuracy

        Returns:
            List of search results
        """
        request_data = {
            "image_data": image_data,
            "limit": limit,
            "use_hybrid_search": use_hybrid_search,
        }

        response = await self._request(
            "POST", "/api/v1/search/image", json_data=request_data
        )
        return response.get("results", [])

    async def search_multimodal(
        self,
        query: str,
        image_data: Optional[str] = None,
        limit: int = 10,
        text_weight: float = 0.7,
    ) -> List[Dict[str, Any]]:
        """
        Perform combined text and image search.

        Args:
            query: Text search query
            image_data: Optional base64 encoded image data
            limit: Maximum number of results
            text_weight: Weight for text similarity (0.0-1.0)

        Returns:
            List of search results
        """
        request_data = {"query": query, "limit": limit, "text_weight": text_weight}
        if image_data:
            request_data["image_data"] = image_data

        response = await self._request(
            "POST", "/api/v1/search/multimodal", json_data=request_data
        )
        return response.get("results", [])

    async def get_by_id(self, anime_id: str) -> Optional[Dict[str, Any]]:
        """
        Get anime by ID.

        Args:
            anime_id: Anime ID to retrieve

        Returns:
            Anime data or None if not found
        """
        try:
            response = await self._request("GET", f"/api/v1/search/by-id/{anime_id}")
            return response.get("anime")
        except VectorServiceError as e:
            if "404" in str(e):
                return None
            raise

    # Similarity Methods

    async def find_similar(
        self, anime_id: str, limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Find similar anime based on content similarity.

        Args:
            anime_id: Reference anime ID
            limit: Maximum number of similar anime

        Returns:
            List of similar anime
        """
        response = await self._request(
            "GET", f"/api/v1/similarity/anime/{anime_id}", params={"limit": limit}
        )
        return response.get("similar_anime", [])

    async def find_visually_similar(
        self, anime_id: str, limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Find visually similar anime based on image similarity.

        Args:
            anime_id: Reference anime ID
            limit: Maximum number of similar anime

        Returns:
            List of visually similar anime
        """
        response = await self._request(
            "GET", f"/api/v1/similarity/visual/{anime_id}", params={"limit": limit}
        )
        return response.get("similar_anime", [])

    async def find_vector_similar(
        self, anime_id: str, limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Find similar anime using direct vector similarity.

        Args:
            anime_id: Reference anime ID
            limit: Maximum number of similar anime

        Returns:
            List of similar anime
        """
        response = await self._request(
            "GET", f"/api/v1/similarity/vector/{anime_id}", params={"limit": limit}
        )
        return response.get("similar_anime", [])

    async def batch_similarity(
        self, anime_ids: List[str], limit: int = 5, similarity_type: str = "semantic"
    ) -> Dict[str, Dict[str, Any]]:
        """
        Find similar anime for multiple references in batch.

        Args:
            anime_ids: List of reference anime IDs
            limit: Maximum similar anime per reference
            similarity_type: Type of similarity (semantic, visual, vector)

        Returns:
            Dictionary mapping anime IDs to their similar anime
        """
        request_data = anime_ids
        params = {"limit": limit, "similarity_type": similarity_type}

        response = await self._request(
            "POST", "/api/v1/similarity/batch", json_data=request_data, params=params
        )
        return response.get("batch_results", {})

    # Administrative Methods

    async def upsert_vectors(
        self, documents: List[Dict[str, Any]], batch_size: int = 100
    ) -> Dict[str, Any]:
        """
        Add or update vectors in the database.

        Args:
            documents: List of documents with embeddings and metadata
            batch_size: Batch size for processing

        Returns:
            Operation result
        """
        request_data = {"documents": documents, "batch_size": batch_size}

        return await self._request(
            "POST", "/api/v1/admin/vectors/upsert", json_data=request_data
        )

    async def get_collection_info(self) -> Dict[str, Any]:
        """Get detailed collection information."""
        return await self._request("GET", "/api/v1/admin/collection/info")

    async def reindex_collection(self) -> Dict[str, Any]:
        """
        Rebuild the vector index.

        WARNING: This will clear all existing data.
        """
        return await self._request("POST", "/api/v1/admin/reindex")

    # Convenience Methods

    async def is_healthy(self) -> bool:
        """Check if the service is healthy."""
        try:
            health = await self.health_check()
            return health.get("status") == "healthy"
        except:
            return False

    async def get_document_count(self) -> int:
        """Get total number of documents in the collection."""
        try:
            stats = await self.get_stats()
            return stats.get("total_documents", 0)
        except:
            return 0


# Singleton client instance for easy usage
_default_client = None


def get_client(
    base_url: str = "http://localhost:8002",
    timeout: int = 30,
    api_key: Optional[str] = None,
) -> VectorServiceClient:
    """
    Get a default client instance.

    Args:
        base_url: Base URL of the vector service
        timeout: Request timeout in seconds
        api_key: Optional API key for authentication

    Returns:
        Vector service client instance
    """
    global _default_client

    if _default_client is None:
        _default_client = VectorServiceClient(
            base_url=base_url, timeout=timeout, api_key=api_key
        )

    return _default_client

