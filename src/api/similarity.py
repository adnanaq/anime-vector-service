"""
Similarity API endpoints for finding related anime.

Provides similarity search based on content, visual style, and vector similarity.
"""

import logging
from typing import Dict, List, Optional, Any

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from ..config import get_settings

settings = get_settings()
logger = logging.getLogger(__name__)

router = APIRouter()


class SimilarityResponse(BaseModel):
    """Response model for similarity operations."""
    reference_anime_id: str = Field(..., description="Reference anime ID")
    similar_anime: List[Dict[str, Any]] = Field(..., description="Similar anime results")
    similarity_type: str = Field(..., description="Type of similarity search performed")
    total_found: int = Field(..., description="Total number of similar anime found")


@router.get("/similarity/anime/{anime_id}", response_model=SimilarityResponse)
async def get_similar_anime(
    anime_id: str,
    limit: int = Query(default=10, ge=1, le=50, description="Maximum number of similar anime to return")
):
    """
    Find anime similar to the reference anime.
    
    Uses semantic similarity based on title, tags, synopsis, and metadata.
    Good for finding anime with similar themes, genres, or content.
    """
    try:
        from ..main import qdrant_client
        
        if not qdrant_client:
            raise HTTPException(status_code=503, detail="Vector database not available")
        
        # Find similar anime
        results = await qdrant_client.find_similar(
            anime_id=anime_id,
            limit=limit
        )
        
        if not results:
            # Check if reference anime exists
            reference = await qdrant_client.get_by_id(anime_id)
            if not reference:
                raise HTTPException(status_code=404, detail=f"Reference anime not found: {anime_id}")
        
        return SimilarityResponse(
            reference_anime_id=anime_id,
            similar_anime=results,
            similarity_type="semantic",
            total_found=len(results)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Similar anime search failed: {e}")
        raise HTTPException(status_code=500, detail=f"Similarity search failed: {str(e)}")


@router.get("/similarity/visual/{anime_id}", response_model=SimilarityResponse)
async def get_visually_similar_anime(
    anime_id: str,
    limit: int = Query(default=10, ge=1, le=50, description="Maximum number of visually similar anime to return")
):
    """
    Find anime with similar visual style to the reference anime.
    
    Uses image embeddings to find anime with similar artwork, art style, or visual appearance.
    Good for finding anime from the same studio or with similar artistic direction.
    """
    try:
        from ..main import qdrant_client
        
        if not qdrant_client:
            raise HTTPException(status_code=503, detail="Vector database not available")
        
        # Find visually similar anime
        results = await qdrant_client.find_visually_similar_anime(
            anime_id=anime_id,
            limit=limit
        )
        
        if not results:
            # Check if reference anime exists
            reference = await qdrant_client.get_by_id(anime_id)
            if not reference:
                raise HTTPException(status_code=404, detail=f"Reference anime not found: {anime_id}")
            
            # If anime exists but no visual similarity found, it might not have image embeddings
            return SimilarityResponse(
                reference_anime_id=anime_id,
                similar_anime=[],
                similarity_type="visual",
                total_found=0
            )
        
        return SimilarityResponse(
            reference_anime_id=anime_id,
            similar_anime=results,
            similarity_type="visual",
            total_found=len(results)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Visual similarity search failed: {e}")
        raise HTTPException(status_code=500, detail=f"Visual similarity search failed: {str(e)}")


@router.get("/similarity/vector/{anime_id}", response_model=SimilarityResponse)
async def get_vector_similar_anime(
    anime_id: str,
    limit: int = Query(default=10, ge=1, le=50, description="Maximum number of similar anime to return")
):
    """
    Find anime similar using direct vector similarity.
    
    Uses the stored embedding vectors to find the most similar anime.
    This is the most direct similarity measure based on the vector database.
    """
    try:
        from ..main import qdrant_client
        
        if not qdrant_client:
            raise HTTPException(status_code=503, detail="Vector database not available")
        
        # Find similar anime using vector similarity
        results = await qdrant_client.get_similar_anime(
            anime_id=anime_id,
            limit=limit
        )
        
        if not results:
            # Check if reference anime exists
            reference = await qdrant_client.get_by_id(anime_id)
            if not reference:
                raise HTTPException(status_code=404, detail=f"Reference anime not found: {anime_id}")
        
        return SimilarityResponse(
            reference_anime_id=anime_id,
            similar_anime=results,
            similarity_type="vector",
            total_found=len(results)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Vector similarity search failed: {e}")
        raise HTTPException(status_code=500, detail=f"Vector similarity search failed: {str(e)}")


@router.post("/similarity/batch")
async def get_batch_similarity(
    anime_ids: List[str] = Field(..., description="List of anime IDs to find similarities for"),
    limit: int = Query(default=5, ge=1, le=20, description="Maximum similar anime per reference"),
    similarity_type: str = Query(default="semantic", description="Type of similarity: semantic, visual, or vector")
):
    """
    Find similar anime for multiple reference anime in batch.
    
    More efficient than individual requests when finding similarities for multiple anime.
    """
    try:
        from ..main import qdrant_client
        
        if not qdrant_client:
            raise HTTPException(status_code=503, detail="Vector database not available")
        
        if len(anime_ids) > 10:
            raise HTTPException(status_code=400, detail="Maximum 10 anime IDs allowed in batch request")
        
        batch_results = {}
        
        for anime_id in anime_ids:
            try:
                if similarity_type == "visual":
                    results = await qdrant_client.find_visually_similar_anime(anime_id, limit)
                elif similarity_type == "vector":
                    results = await qdrant_client.get_similar_anime(anime_id, limit)
                else:  # semantic (default)
                    results = await qdrant_client.find_similar(anime_id, limit)
                
                batch_results[anime_id] = {
                    "similar_anime": results,
                    "total_found": len(results),
                    "similarity_type": similarity_type
                }
                
            except Exception as e:
                logger.warning(f"Failed to find similarities for {anime_id}: {e}")
                batch_results[anime_id] = {
                    "similar_anime": [],
                    "total_found": 0,
                    "error": str(e),
                    "similarity_type": similarity_type
                }
        
        return {
            "batch_results": batch_results,
            "total_processed": len(anime_ids),
            "similarity_type": similarity_type
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Batch similarity search failed: {e}")
        raise HTTPException(status_code=500, detail=f"Batch similarity search failed: {str(e)}")