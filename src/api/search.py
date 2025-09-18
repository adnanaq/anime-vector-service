"""
Search API endpoints for vector operations.

Provides semantic search, image search, and multimodal search capabilities.
"""

import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from ..config import get_settings

settings = get_settings()
logger = logging.getLogger(__name__)

router = APIRouter()


class SearchRequest(BaseModel):
    """Request model for text search."""

    query: str = Field(..., description="Search query text", min_length=1)
    limit: int = Field(
        default=20,
        ge=1,
        le=settings.max_search_limit,
        description="Maximum number of results",
    )
    filters: Optional[Dict[str, Any]] = Field(
        None, description="Optional search filters"
    )


class ImageSearchRequest(BaseModel):
    """Request model for image search."""

    image_data: str = Field(..., description="Base64 encoded image data")
    limit: int = Field(
        default=10,
        ge=1,
        le=settings.max_search_limit,
        description="Maximum number of results",
    )
    use_hybrid_search: bool = Field(
        default=True,
        description="Use hybrid search combining picture and thumbnail vectors",
    )


class MultimodalSearchRequest(BaseModel):
    """Request model for multimodal search."""

    query: str = Field(..., description="Text search query", min_length=1)
    image_data: Optional[str] = Field(
        None, description="Optional base64 encoded image data"
    )
    limit: int = Field(
        default=10,
        ge=1,
        le=settings.max_search_limit,
        description="Maximum number of results",
    )
    text_weight: float = Field(
        default=0.7, ge=0.0, le=1.0, description="Weight for text similarity (0.0-1.0)"
    )


class SearchResponse(BaseModel):
    """Response model for search operations."""

    results: List[Dict[str, Any]] = Field(..., description="Search results")
    total_found: int = Field(..., description="Total number of results found")
    query_info: Dict[str, Any] = Field(..., description="Query processing information")


@router.post("/search", response_model=SearchResponse)
async def search_anime(request: SearchRequest):
    """
    Perform semantic text search on anime database.

    Uses advanced text embeddings to find anime based on natural language queries.
    Supports filtering by genre, year, type, etc.
    """
    try:
        # Import here to avoid circular imports
        from ..main import qdrant_client

        if not qdrant_client:
            raise HTTPException(status_code=503, detail="Vector database not available")

        # Perform search
        results = await qdrant_client.search(
            query=request.query, limit=request.limit, filters=request.filters
        )

        return SearchResponse(
            results=results,
            total_found=len(results),
            query_info={
                "query": request.query,
                "search_type": "text",
                "filters_applied": request.filters is not None,
                "processing_time_ms": 0,  # Could add timing here
            },
        )

    except Exception as e:
        logger.error(f"Search failed: {e}")
        raise HTTPException(
            status_code=500, detail=f"Search operation failed: {str(e)}"
        )


@router.post("/search/image", response_model=SearchResponse)
async def search_by_image(request: ImageSearchRequest):
    """
    Perform image-based search on anime database.

    Uses visual similarity to find anime with similar artwork, covers, or screenshots.
    Supports hybrid search combining multiple image vectors for better accuracy.
    """
    try:
        from ..main import qdrant_client

        if not qdrant_client:
            raise HTTPException(status_code=503, detail="Vector database not available")

        # Validate image data
        if not request.image_data or not request.image_data.strip():
            raise HTTPException(status_code=400, detail="Image data is required")

        # Perform image search
        results = await qdrant_client.search_by_image(
            image_data=request.image_data,
            limit=request.limit,
            use_hybrid_search=request.use_hybrid_search,
        )

        return SearchResponse(
            results=results,
            total_found=len(results),
            query_info={
                "search_type": "image",
                "hybrid_search": request.use_hybrid_search,
                "image_processed": len(request.image_data) > 0,
                "processing_time_ms": 0,
            },
        )

    except Exception as e:
        logger.error(f"Image search failed: {e}")
        raise HTTPException(
            status_code=500, detail=f"Image search operation failed: {str(e)}"
        )


@router.post("/search/multimodal", response_model=SearchResponse)
async def search_multimodal(request: MultimodalSearchRequest):
    """
    Perform combined text and image search.

    Combines semantic text search with visual similarity for more accurate results.
    Allows weighting between text and image components.
    """
    try:
        from ..main import qdrant_client

        if not qdrant_client:
            raise HTTPException(status_code=503, detail="Vector database not available")

        # Perform multimodal search
        results = await qdrant_client.search_multimodal(
            query=request.query,
            image_data=request.image_data,
            limit=request.limit,
            text_weight=request.text_weight,
        )

        return SearchResponse(
            results=results,
            total_found=len(results),
            query_info={
                "query": request.query,
                "search_type": "multimodal",
                "has_image": request.image_data is not None,
                "text_weight": request.text_weight,
                "image_weight": 1.0 - request.text_weight,
                "processing_time_ms": 0,
            },
        )

    except Exception as e:
        logger.error(f"Multimodal search failed: {e}")
        raise HTTPException(
            status_code=500, detail=f"Multimodal search operation failed: {str(e)}"
        )


@router.get("/search/by-id/{anime_id}")
async def get_anime_by_id(anime_id: str):
    """
    Get anime details by ID.

    Retrieves full anime information from the vector database.
    """
    try:
        from ..main import qdrant_client

        if not qdrant_client:
            raise HTTPException(status_code=503, detail="Vector database not available")

        # Get anime by ID
        result = await qdrant_client.get_by_id(anime_id)

        if not result:
            raise HTTPException(status_code=404, detail=f"Anime not found: {anime_id}")

        return {"anime": result, "found": True, "anime_id": anime_id}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get by ID failed: {e}")
        raise HTTPException(
            status_code=500, detail=f"Retrieval operation failed: {str(e)}"
        )
