"""Multi-Vector Embedding Manager for coordinated embedding generation.

This module coordinates the generation of all 13 vectors (12 text + 1 visual)
for the comprehensive anime search system with error handling and validation.
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional

from ..config import Settings
from ..models.anime import AnimeEntry
from .anime_field_mapper import AnimeFieldMapper
from .text_processor import TextProcessor
from .vision_processor import VisionProcessor

logger = logging.getLogger(__name__)


class MultiVectorEmbeddingManager:
    """Manager for coordinated generation of all 13 embedding vectors."""

    def __init__(self, settings: Optional[Settings] = None):
        """Initialize the multi-vector embedding manager.

        Args:
            settings: Configuration settings instance
        """
        if settings is None:
            from ..config import get_settings

            settings = get_settings()

        self.settings = settings

        # Initialize processors
        self.text_processor = TextProcessor(settings)
        self.vision_processor = VisionProcessor(settings)

        # Initialize field mapper
        self.field_mapper = AnimeFieldMapper()

        # Get vector configuration
        self.vector_names = list(settings.vector_names.keys())
        self.text_vector_names = [v for v in self.vector_names if v != "image_vector"]
        self.image_vector_name = "image_vector"

        logger.info(
            f"Initialized MultiVectorEmbeddingManager with {len(self.vector_names)} vectors"
        )

    async def process_anime_vectors(self, anime: AnimeEntry) -> Dict[str, Any]:
        """Process anime data to generate all embedding vectors.

        Args:
            anime: AnimeEntry instance with anime data

        Returns:
            Dictionary containing:
            - vectors: Dict with vector name -> embedding list mappings
            - payload: Dict with additional data for Qdrant storage
            - metadata: Dict with processing metadata
        """
        try:
            # Generate all text vectors
            text_vectors = await self._generate_text_vectors(anime)

            # Generate image vector
            image_vector = await self._generate_image_vector(anime)

            # Combine all vectors
            all_vectors = {**text_vectors}
            if image_vector is not None:
                all_vectors[self.image_vector_name] = image_vector

            # Generate payload data
            payload = self._generate_payload(anime)

            # Generate processing metadata
            metadata = self._generate_metadata(all_vectors, anime)

            return {"vectors": all_vectors, "payload": payload, "metadata": metadata}

        except Exception as e:
            logger.error(f"Failed to process anime vectors for {anime.title}: {e}")
            return {
                "vectors": {},
                "payload": {},
                "metadata": {"error": str(e), "processing_failed": True},
            }

    async def _generate_text_vectors(self, anime: AnimeEntry) -> Dict[str, List[float]]:
        """Generate all text-based embedding vectors.

        Args:
            anime: AnimeEntry instance

        Returns:
            Dictionary mapping text vector names to embeddings
        """
        try:
            # Use text processor's multi-vector method
            text_vectors = self.text_processor.process_anime_vectors(anime)

            # Filter out None vectors and log status
            valid_vectors = {}
            failed_vectors = []

            for vector_name in self.text_vector_names:
                if (
                    vector_name in text_vectors
                    and text_vectors[vector_name] is not None
                ):
                    valid_vectors[vector_name] = text_vectors[vector_name]
                else:
                    failed_vectors.append(vector_name)

            if failed_vectors:
                logger.warning(f"Failed to generate text vectors: {failed_vectors}")

            logger.debug(
                f"Generated {len(valid_vectors)}/{len(self.text_vector_names)} text vectors"
            )
            return valid_vectors

        except Exception as e:
            logger.error(f"Text vector generation failed: {e}")
            return {}

    async def _generate_image_vector(self, anime: AnimeEntry) -> Optional[List[float]]:
        """Generate image embedding vector.

        Args:
            anime: AnimeEntry instance

        Returns:
            Image embedding vector or None if generation fails
        """
        try:
            # Use vision processor's async method
            image_vector = await self.vision_processor.process_anime_image_vector(anime)

            if image_vector is not None:
                logger.debug("Successfully generated image vector")
            else:
                logger.debug(
                    "Image vector generation failed - will store URL in payload"
                )

            return image_vector

        except Exception as e:
            logger.error(f"Image vector generation failed: {e}")
            return None

    def _generate_payload(self, anime: AnimeEntry) -> Dict[str, Any]:
        """Generate payload data for Qdrant storage.

        Args:
            anime: AnimeEntry instance

        Returns:
            Payload dictionary for Qdrant
        """
        try:
            # Basic anime information
            payload = {
                "id": anime.id,
                "title": anime.title,
                "synopsis": anime.synopsis or "",
                "type": anime.type or "unknown",
                "status": anime.status or "unknown",
                "episodes": anime.episodes,
                "genres": anime.genres or [],
                "tags": anime.tags or [],
                "rating": anime.rating or "",
                "nsfw": anime.nsfw,
            }

            # Add duration for precise numerical filtering
            if anime.duration:
                payload["duration"] = anime.duration

            # Add anime_season for precise temporal filtering
            if anime.anime_season:
                payload["anime_season"] = {
                    "season": anime.anime_season.season,
                    "year": anime.anime_season.year
                }

            # Add aggregated score calculations if available
            if anime.score:
                payload["score"] = {
                    "arithmetic_geometric_mean": anime.score.arithmeticGeometricMean,
                    "arithmetic_mean": anime.score.arithmeticMean,
                    "median": anime.score.median,
                }

            # Add all platform statistics for indexing and filtering
            if anime.statistics:
                payload["statistics"] = {}
                for platform, stats in anime.statistics.items():
                    payload["statistics"][platform] = {
                        "score": getattr(stats, "score", None),
                        "scored_by": getattr(stats, "scored_by", None),
                        "popularity_rank": getattr(stats, "popularity_rank", None),
                        "members": getattr(stats, "members", None),
                        "favorites": getattr(stats, "favorites", None),
                        "rank": getattr(stats, "rank", None),
                    }

            # Add sources for platform availability filtering
            if anime.sources:
                payload["sources"] = anime.sources

            # Add enrichment metadata for data quality (non-indexed operational data)
            if hasattr(anime, "enrichment_metadata") and anime.enrichment_metadata:
                payload["enrichment_metadata"] = anime.enrichment_metadata

            # Add complete images structure for frontend use and fallback
            if hasattr(anime, "images") and anime.images:
                payload["images"] = anime.images

            return payload

        except Exception as e:
            logger.error(f"Payload generation failed: {e}")
            return {"error": str(e)}

    def _generate_metadata(
        self, vectors: Dict[str, List[float]], anime: AnimeEntry
    ) -> Dict[str, Any]:
        """Generate processing metadata.

        Args:
            vectors: Generated vectors dictionary
            anime: AnimeEntry instance

        Returns:
            Processing metadata dictionary
        """
        try:
            # Vector generation statistics
            total_expected = len(self.vector_names)
            total_generated = len(vectors)
            text_vectors_generated = len(
                [v for v in vectors.keys() if v != self.image_vector_name]
            )
            image_vector_generated = self.image_vector_name in vectors

            # Missing vectors
            missing_vectors = [v for v in self.vector_names if v not in vectors]

            metadata = {
                "processing_timestamp": None,  # Will be set by caller
                "total_vectors_expected": total_expected,
                "total_vectors_generated": total_generated,
                "text_vectors_generated": text_vectors_generated,
                "image_vector_generated": image_vector_generated,
                "missing_vectors": missing_vectors,
                "success_rate": (
                    total_generated / total_expected if total_expected > 0 else 0.0
                ),
                "anime_id": anime.id,
                "anime_title": anime.title,
                "processing_complete": len(missing_vectors) == 0,
            }

            # Add vector dimensions for validation
            vector_dimensions = {}
            for vector_name, vector_data in vectors.items():
                if vector_data:
                    vector_dimensions[vector_name] = len(vector_data)

            metadata["vector_dimensions"] = vector_dimensions

            return metadata

        except Exception as e:
            logger.error(f"Metadata generation failed: {e}")
            return {"error": str(e)}

    async def process_anime_batch(
        self, anime_list: List[AnimeEntry]
    ) -> List[Dict[str, Any]]:
        """Process multiple anime entries in batch.

        Args:
            anime_list: List of AnimeEntry instances

        Returns:
            List of processing results for each anime
        """
        try:
            logger.info(f"Processing batch of {len(anime_list)} anime")

            # Process all anime concurrently
            tasks = [self.process_anime_vectors(anime) for anime in anime_list]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Handle any exceptions in results
            processed_results = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(f"Batch processing error for anime {i}: {result}")
                    processed_results.append(
                        {
                            "vectors": {},
                            "payload": {},
                            "metadata": {
                                "error": str(result),
                                "processing_failed": True,
                            },
                        }
                    )
                else:
                    processed_results.append(result)

            # Log batch statistics
            successful = sum(
                1
                for r in processed_results
                if not r["metadata"].get("processing_failed", False)
            )
            logger.info(
                f"Batch processing complete: {successful}/{len(anime_list)} successful"
            )

            return processed_results

        except Exception as e:
            logger.error(f"Batch processing failed: {e}")
            return []

    def validate_vectors(self, vectors: Dict[str, List[float]]) -> Dict[str, Any]:
        """Validate generated vectors.

        Args:
            vectors: Dictionary of vector name -> embedding mappings

        Returns:
            Validation report dictionary
        """
        try:
            validation_report = {
                "valid": True,
                "errors": [],
                "warnings": [],
                "vector_stats": {},
            }

            expected_dimensions = self.settings.vector_names

            for vector_name, vector_data in vectors.items():
                if vector_name not in expected_dimensions:
                    validation_report["errors"].append(
                        f"Unexpected vector: {vector_name}"
                    )
                    validation_report["valid"] = False
                    continue

                expected_dim = expected_dimensions[vector_name]
                actual_dim = len(vector_data) if vector_data else 0

                if actual_dim != expected_dim:
                    validation_report["errors"].append(
                        f"{vector_name}: dimension mismatch (expected {expected_dim}, got {actual_dim})"
                    )
                    validation_report["valid"] = False

                # Check for invalid values
                if vector_data and any(
                    not isinstance(x, (int, float)) for x in vector_data
                ):
                    validation_report["errors"].append(
                        f"{vector_name}: contains non-numeric values"
                    )
                    validation_report["valid"] = False

                # Statistics
                validation_report["vector_stats"][vector_name] = {
                    "dimension": actual_dim,
                    "non_zero_values": (
                        sum(1 for x in vector_data if x != 0) if vector_data else 0
                    ),
                    "magnitude": (
                        sum(x * x for x in vector_data) ** 0.5 if vector_data else 0.0
                    ),
                }

            # Check for missing critical vectors
            missing_critical = []
            critical_vectors = self.settings.vector_priorities.get("high", [])
            for critical_vector in critical_vectors:
                if critical_vector not in vectors:
                    missing_critical.append(critical_vector)

            if missing_critical:
                validation_report["warnings"].append(
                    f"Missing critical vectors: {missing_critical}"
                )

            return validation_report

        except Exception as e:
            logger.error(f"Vector validation failed: {e}")
            return {"valid": False, "errors": [str(e)]}

    def get_embedding_stats(self) -> Dict[str, Any]:
        """Get embedding manager statistics.

        Returns:
            Statistics dictionary
        """
        try:
            text_model_info = self.text_processor.get_model_info()
            vision_model_info = self.vision_processor.get_model_info()

            return {
                "text_processor": text_model_info,
                "vision_processor": vision_model_info,
                "vector_configuration": {
                    "total_vectors": len(self.vector_names),
                    "text_vectors": len(self.text_vector_names),
                    "image_vectors": 1,
                    "vector_names": self.vector_names,
                    "vector_dimensions": dict(self.settings.vector_names),
                    "vector_priorities": dict(self.settings.vector_priorities),
                },
                "cache_stats": {"image_cache": self.vision_processor.get_cache_stats()},
            }

        except Exception as e:
            logger.error(f"Failed to get embedding stats: {e}")
            return {"error": str(e)}
