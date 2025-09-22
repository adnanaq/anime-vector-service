"""Qdrant Vector Database Client for Anime Search

Provides high-performance vector search capabilities optimized for anime data
with advanced filtering, cross-platform ID lookups, and hybrid search.
"""

import asyncio
import hashlib
import logging
from typing import Any, Dict, List, Optional, TypeGuard

# fastembed import moved to _init_encoder method for lazy loading
from qdrant_client import QdrantClient as QdrantSDK
from qdrant_client.models import (  # Qdrant optimization models; Multi-vector search models
    BinaryQuantization,
    BinaryQuantizationConfig,
    CollectionParams,
    Distance,
    FieldCondition,
    Filter,
    Fusion,
    FusionQuery,
    HnswConfig,
    HnswConfigDiff,
    MatchAny,
    MatchValue,
    NamedVector,
    NearestQuery,
    OptimizersConfig,
    OptimizersConfigDiff,
    PayloadSchemaType,
    PointStruct,
    Prefetch,
    ProductQuantization,
    QuantizationConfig,
    Query,
    QueryRequest,
    Range,
    ScalarQuantization,
    ScalarQuantizationConfig,
    ScalarType,
    VectorParams,
    WalConfig,
    WalConfigDiff,
)

from ...config import Settings
from ...models.anime import AnimeEntry
from ..processors.embedding_manager import MultiVectorEmbeddingManager

logger = logging.getLogger(__name__)


def is_float_vector(vector: Any) -> TypeGuard[List[float]]:
    """Type guard to check if vector is a List[float]."""
    return (
        isinstance(vector, list)
        and len(vector) > 0
        and all(isinstance(x, (int, float)) for x in vector)
    )


class QdrantClient:
    """Qdrant client wrapper optimized for anime search operations."""

    def __init__(
        self,
        url: Optional[str] = None,
        collection_name: Optional[str] = None,
        settings: Optional[Settings] = None,
    ):
        """Initialize Qdrant client with FastEmbed and configuration.

        Args:
            url: Qdrant server URL (optional, uses settings if not provided)
            collection_name: Name of the anime collection (optional, uses settings if not provided)
            settings: Configuration settings instance (optional, will import default if not provided)
        """
        # Use provided settings or import default settings
        if settings is None:
            from ...config.settings import Settings

            settings = Settings()

        self.settings = settings
        self.url = url or settings.qdrant_url
        self.collection_name = collection_name or settings.qdrant_collection_name

        # Initialize Qdrant client with API key if provided
        if settings.qdrant_api_key:
            self.client = QdrantSDK(url=self.url, api_key=settings.qdrant_api_key)
        else:
            self.client = QdrantSDK(url=self.url)

        self._distance_metric = settings.qdrant_distance_metric

        # Initialize embedding manager
        self.embedding_manager = MultiVectorEmbeddingManager(settings)

        # Initialize processors
        self._init_processors()

        # Create collection if it doesn't exist
        self._initialize_collection()

    def _init_processors(self) -> None:
        """Initialize embedding processors."""
        try:
            # Import processors
            from ..processors.text_processor import TextProcessor
            from ..processors.vision_processor import VisionProcessor

            # Initialize text processor
            self.text_processor = TextProcessor(self.settings)

            # Initialize vision processor
            self.vision_processor = VisionProcessor(self.settings)

            # Update vector sizes based on modern models
            text_info = self.text_processor.get_model_info()
            vision_info = self.vision_processor.get_model_info()

            self._vector_size = text_info.get("embedding_size", 384)
            self._image_vector_size = vision_info.get("embedding_size", 512)

            logger.info(
                f"Initialized processors - Text: {text_info['model_name']} ({self._vector_size}), "
                f"Vision: {vision_info['model_name']} ({self._image_vector_size})"
            )

        except Exception as e:
            logger.error(f"Failed to initialize processors: {e}")
            raise

    def _initialize_collection(self) -> None:
        """Initialize and validate anime collection with 14-vector architecture and performance optimization."""
        try:
            # Check if collection exists and validate its configuration
            collections = self.client.get_collections().collections
            collection_exists = any(
                col.name == self.collection_name for col in collections
            )

            if not collection_exists:
                # Create collection with current vector architecture
                logger.info(f"Creating optimized collection: {self.collection_name}")
                vectors_config = self._create_multi_vector_config()

                # Validate vector configuration before creation
                self._validate_vector_config(vectors_config)

                # Add performance optimization configurations
                quantization_config = self._create_quantization_config()
                optimizers_config = self._create_optimized_optimizers_config()
                wal_config = self._create_wal_config()

                # Create collection with optimization
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=vectors_config,
                    quantization_config=quantization_config,
                    optimizers_config=optimizers_config,
                    wal_config=wal_config,
                )

                # Configure payload indexing for faster filtering
                if getattr(self.settings, "qdrant_enable_payload_indexing", True):
                    self._setup_payload_indexing()

                logger.info(
                    f"Successfully created collection with {len(vectors_config)} vectors"
                )
            else:
                # Validate existing collection compatibility
                if not self._validate_collection_compatibility():
                    logger.warning(
                        f"Collection {self.collection_name} exists but may have compatibility issues"
                    )
                    logger.info(
                        "Continuing with existing collection configuration for backward compatibility"
                    )
                else:
                    logger.info(
                        f"Collection {self.collection_name} validated successfully"
                    )

        except Exception as e:
            logger.error(f"Failed to ensure collection exists: {e}")
            raise

    def _validate_collection_compatibility(self) -> bool:
        """Validate existing collection compatibility with current vector architecture."""
        try:
            collection_info = self.client.get_collection(self.collection_name)
            existing_vectors = collection_info.config.params.vectors

            # Check if collection has expected vector configurations
            expected_vectors = set(self.settings.vector_names.keys())

            if isinstance(existing_vectors, dict):
                existing_vector_names = set(existing_vectors.keys())

                # Check if we have the current semantic vectors
                has_current_vectors = expected_vectors.issubset(existing_vector_names)

                if has_current_vectors:
                    logger.info("Collection has complete vector configuration")
                    return True
                else:
                    logger.warning(
                        f"Collection missing expected vectors. Expected: {expected_vectors}, Found: {existing_vector_names}"
                    )
                    return False
            else:
                logger.warning(
                    "Collection uses single vector configuration, not compatible with multi-vector architecture"
                )
                return False

        except Exception as e:
            logger.error(f"Failed to validate collection compatibility: {e}")
            return False

    def _validate_vector_config(self, vectors_config: Dict[str, VectorParams]) -> None:
        """Validate vector configuration before collection creation."""
        if not vectors_config:
            raise ValueError("Vector configuration is empty")

        expected_count = len(self.settings.vector_names)
        actual_count = len(vectors_config)

        if actual_count != expected_count:
            raise ValueError(
                f"Vector count mismatch: expected {expected_count}, got {actual_count}"
            )

        # Validate vector dimensions
        for vector_name, vector_params in vectors_config.items():
            expected_dim = self.settings.vector_names.get(vector_name)
            if expected_dim and vector_params.size != expected_dim:
                raise ValueError(
                    f"Vector {vector_name} dimension mismatch: expected {expected_dim}, got {vector_params.size}"
                )

        logger.info(
            f"Vector configuration validated: {actual_count} vectors with correct dimensions"
        )

    # Distance mapping constant
    _DISTANCE_MAPPING = {
        "cosine": Distance.COSINE,
        "euclid": Distance.EUCLID,
        "dot": Distance.DOT,
    }

    def _create_multi_vector_config(self) -> Dict[str, VectorParams]:
        """Create 14-vector configuration with priority-based optimization."""
        distance = self._DISTANCE_MAPPING.get(self._distance_metric, Distance.COSINE)

        # Use new 14-vector architecture from settings
        vector_params = {}
        for vector_name, dimension in self.settings.vector_names.items():
            priority = self._get_vector_priority(vector_name)
            vector_params[vector_name] = VectorParams(
                size=dimension,
                distance=distance,
                hnsw_config=self._get_hnsw_config(priority),
                quantization_config=self._get_quantization_config(priority),
            )

        logger.info(
            f"Created 14-vector configuration with {len(vector_params)} vectors"
        )
        return vector_params

    # NEW: Priority-Based Configuration Methods for Million-Query Optimization

    def _get_quantization_config(self, priority: str) -> Optional[QuantizationConfig]:
        """Get quantization config based on vector priority."""
        config = self.settings.quantization_config.get(priority, {})
        if config.get("type") == "scalar":
            scalar_config = ScalarQuantizationConfig(
                type=ScalarType.INT8, always_ram=config.get("always_ram", False)
            )
            return ScalarQuantization(scalar=scalar_config)
        elif config.get("type") == "binary":
            binary_config = BinaryQuantizationConfig(
                always_ram=config.get("always_ram", False)
            )
            return BinaryQuantization(binary=binary_config)
        return None

    def _get_hnsw_config(self, priority: str) -> HnswConfigDiff:
        """Get HNSW config based on vector priority."""
        config = self.settings.hnsw_config.get(priority, {})
        return HnswConfigDiff(
            ef_construct=config.get("ef_construct", 200), m=config.get("m", 48)
        )

    def _get_vector_priority(self, vector_name: str) -> str:
        """Determine priority level for vector."""
        for priority, vectors in self.settings.vector_priorities.items():
            if vector_name in vectors:
                return str(priority)
        return "medium"  # default

    def _create_optimized_optimizers_config(self) -> Optional[OptimizersConfigDiff]:
        """Create optimized optimizers configuration for million-query scale."""
        try:
            return OptimizersConfigDiff(
                default_segment_number=4,
                indexing_threshold=20000,
                memmap_threshold=self.settings.memory_mapping_threshold_mb * 1024,
            )
        except Exception as e:
            logger.error(f"Failed to create optimized optimizers config: {e}")
            return None

    def _create_quantization_config(
        self,
    ) -> Optional[BinaryQuantization | ScalarQuantization | ProductQuantization]:
        """Create quantization configuration for performance optimization."""
        if not getattr(self.settings, "qdrant_enable_quantization", False):
            return None

        quantization_type = getattr(self.settings, "qdrant_quantization_type", "scalar")
        always_ram = getattr(self.settings, "qdrant_quantization_always_ram", None)

        try:
            if quantization_type == "binary":
                binary_config = BinaryQuantizationConfig(always_ram=always_ram)
                logger.info("Enabling binary quantization for 40x speedup potential")
                return BinaryQuantization(binary=binary_config)
            elif quantization_type == "scalar":
                scalar_config = ScalarQuantizationConfig(
                    type=ScalarType.INT8,  # 8-bit quantization for good balance
                    always_ram=always_ram,
                )
                logger.info("Enabling scalar quantization for memory optimization")
                return ScalarQuantization(scalar=scalar_config)
            elif quantization_type == "product":
                from qdrant_client.models import (
                    CompressionRatio,
                    ProductQuantizationConfig,
                )

                product_config = ProductQuantizationConfig(
                    compression=CompressionRatio.X16
                )
                logger.info("Enabling product quantization for storage optimization")
                return ProductQuantization(product=product_config)
            else:
                logger.warning(f"Unknown quantization type: {quantization_type}")
                return None
        except Exception as e:
            logger.error(f"Failed to create quantization config: {e}")
            return None

    def _create_optimizers_config(self) -> Optional[OptimizersConfig]:
        """Create optimizers configuration for indexing performance."""
        try:
            optimizer_params = {}

            # Task #116: Configure memory mapping threshold
            memory_threshold = getattr(
                self.settings, "qdrant_memory_mapping_threshold", None
            )
            if memory_threshold:
                optimizer_params["memmap_threshold"] = memory_threshold

            # Configure indexing threads if specified
            indexing_threads = getattr(
                self.settings, "qdrant_hnsw_max_indexing_threads", None
            )
            if indexing_threads:
                optimizer_params["indexing_threshold"] = 0  # Start indexing immediately

            if optimizer_params:
                logger.info(f"Applying optimizer configuration: {optimizer_params}")
                return OptimizersConfig(**optimizer_params)
            return None
        except Exception as e:
            logger.error(f"Failed to create optimizers config: {e}")
            return None

    def _create_wal_config(self) -> Optional[WalConfigDiff]:
        """Create Write-Ahead Logging configuration."""
        enable_wal = getattr(self.settings, "qdrant_enable_wal", None)
        if enable_wal is not None:
            try:
                config = WalConfigDiff(wal_capacity_mb=32, wal_segments_ahead=0)
                logger.info(f"WAL configuration: enabled={enable_wal}")
                return config
            except Exception as e:
                logger.error(f"Failed to create WAL config: {e}")
        return None

    def _setup_payload_indexing(self) -> None:
        """Setup payload field indexing for faster filtering.

        Creates indexes only for searchable metadata fields while keeping
        operational data (like enrichment_metadata) non-indexed for storage efficiency.

        Indexed fields include:
        - Core searchable fields: id, title, type, status, episodes, rating, nsfw
        - Categorical fields: genres, tags, demographics, content_warnings
        - Temporal fields: anime_season, duration
        - Platform fields: sources
        - Statistics for numerical filtering: statistics, score

        Non-indexed operational data:
        - enrichment_metadata: Development/debugging data not needed for search
        """
        indexed_fields = getattr(self.settings, "qdrant_indexed_payload_fields", {})
        if not indexed_fields:
            logger.info("No payload indexing configured")
            return

        try:
            logger.info(
                f"Setting up payload indexing for {len(indexed_fields)} searchable fields with optimized types"
            )
            logger.info(
                "Indexed fields enable fast filtering on: core metadata, genres, temporal data, platform stats"
            )
            logger.info(
                "Non-indexed fields (enrichment_metadata) stored for debugging but don't impact search performance"
            )

            # Map string types to PayloadSchemaType enums
            type_mapping = {
                "keyword": PayloadSchemaType.KEYWORD,
                "integer": PayloadSchemaType.INTEGER,
                "float": PayloadSchemaType.FLOAT,
                "bool": PayloadSchemaType.BOOL,
                "text": PayloadSchemaType.TEXT,
                "geo": PayloadSchemaType.GEO,
                "datetime": PayloadSchemaType.DATETIME,
                "uuid": PayloadSchemaType.UUID,
            }

            for field_name, field_type in indexed_fields.items():
                # Get the appropriate schema type
                schema_type = type_mapping.get(field_type.lower(), PayloadSchemaType.KEYWORD)

                # Create index for each field with its specific type
                self.client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name=field_name,
                    field_schema=schema_type,
                )
                logger.debug(f"✓ Created {field_type.upper()} index for field: {field_name}")

            logger.info(
                f"Successfully indexed {len(indexed_fields)} searchable payload fields with optimized types"
            )
            logger.info(
                "Payload optimization complete: type-specific indexing enabled for better filtering performance"
            )

        except Exception as e:
            logger.warning(f"Failed to setup payload indexing: {e}")
            # Don't fail collection creation if indexing fails

    async def health_check(self) -> bool:
        """Check if Qdrant is healthy and reachable."""
        try:
            # Run in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            # Simple health check by getting collections
            collections = await loop.run_in_executor(
                None, lambda: self.client.get_collections()
            )
            return True
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False

    async def get_stats(self) -> Dict[str, Any]:
        """Get collection statistics."""
        try:
            loop = asyncio.get_event_loop()

            # Get collection info
            collection_info = await loop.run_in_executor(
                None, lambda: self.client.get_collection(self.collection_name)
            )

            # Count total points
            count_result = await loop.run_in_executor(
                None,
                lambda: self.client.count(
                    collection_name=self.collection_name, count_filter=None, exact=True
                ),
            )

            return {
                "collection_name": self.collection_name,
                "total_documents": count_result.count,
                "vector_size": self._vector_size,
                "distance_metric": "cosine",
                "status": collection_info.status,
                "optimizer_status": collection_info.optimizer_status,
                "indexed_vectors_count": collection_info.indexed_vectors_count,
                "points_count": collection_info.points_count,
            }

        except Exception as e:
            logger.error(f"Failed to get stats: {e}")
            return {"error": str(e)}

    def _generate_point_id(self, anime_id: str) -> str:
        """Generate unique point ID from anime ID."""
        return hashlib.md5(anime_id.encode()).hexdigest()

    async def add_documents(
        self, documents: List[AnimeEntry], batch_size: int = 100
    ) -> bool:
        """Add anime documents to the collection using the 13-vector architecture.

        Args:
            documents: List of AnimeEntry objects
            batch_size: Number of documents to process per batch

        Returns:
            True if successful, False otherwise
        """
        try:
            total_docs = len(documents)
            logger.info(f"Adding {total_docs} documents in batches of {batch_size}")

            for i in range(0, total_docs, batch_size):
                batch_documents = documents[i : i + batch_size]

                # Process batch to get vectors and payloads
                processed_batch = await self.embedding_manager.process_anime_batch(
                    batch_documents
                )

                points = []
                for doc_data in processed_batch:
                    if doc_data["metadata"].get("processing_failed"):
                        logger.warning(
                            f"Skipping failed document: {doc_data['metadata'].get('anime_title')}"
                        )
                        continue

                    point_id = self._generate_point_id(doc_data["payload"]["id"])

                    point = PointStruct(
                        id=point_id,
                        vector=doc_data["vectors"],
                        payload=doc_data["payload"],
                    )
                    points.append(point)

                if points:
                    # Upsert batch to Qdrant
                    self.client.upsert(
                        collection_name=self.collection_name, points=points, wait=True
                    )
                    logger.info(
                        f"Uploaded batch {i//batch_size + 1}/{(total_docs-1)//batch_size + 1} ({len(points)} points)"
                    )

            logger.info(f"Successfully added {total_docs} documents")
            return True

        except Exception as e:
            logger.error(f"Failed to add documents: {e}")
            return False

    # async def search(
    #     self, query: str, limit: int = 20, filters: Optional[Dict] = None
    # ) -> List[Dict[str, Any]]:
    #     """Perform semantic search on anime collection.

    #     Args:
    #         query: Search query text
    #         limit: Maximum number of results
    #         filters: Optional filters for metadata

    #     Returns:
    #         List of search results with anime data and scores
    #     """
    #     try:
    #         # Create query embedding using the embedding manager's text processor
    #         query_embedding = self.embedding_manager.text_processor.encode_text(query)
    #         if query_embedding is None:
    #             logger.warning("Failed to create embedding for search query.")
    #             return []

    #         # Build filter if provided
    #         qdrant_filter = None
    #         if filters:
    #             qdrant_filter = self._build_filter(filters)

    #         loop = asyncio.get_event_loop()

    #         # Perform search using named vector for multi-vector collection
    #         search_result = await loop.run_in_executor(
    #             None,
    #             lambda: self.client.search(
    #                 collection_name=self.collection_name,
    #                 query_vector=NamedVector(name="title_vector", vector=query_embedding),
    #                 query_filter=qdrant_filter,
    #                 limit=limit,
    #                 with_payload=True,
    #                 with_vectors=False,
    #             ),
    #         )

    #         # Format results
    #         results = []
    #         for hit in search_result:
    #             result = dict(hit.payload)
    #             result["_score"] = hit.score
    #             result["_id"] = hit.id
    #             results.append(result)

    #         return results

    #     except Exception as e:
    #         logger.error(f"Search failed: {e}")
    #         return []

    async def get_similar_anime(
        self, anime_id: str, limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Find similar anime based on vector similarity.

        Args:
            anime_id: ID of the reference anime
            limit: Maximum number of similar anime to return

        Returns:
            List of similar anime with similarity scores
        """
        try:
            # Find the reference anime
            point_id = self._generate_point_id(anime_id)

            loop = asyncio.get_event_loop()

            # Get the reference point
            reference_point = await loop.run_in_executor(
                None,
                lambda: self.client.retrieve(
                    collection_name=self.collection_name,
                    ids=[point_id],
                    with_vectors=True,
                ),
            )

            if not reference_point:
                logger.warning(f"Anime not found: {anime_id}")
                return []

            # Use the reference vector to find similar anime
            reference_vectors = reference_point[0].vector

            # Search excluding the reference anime itself
            filter_out_self = Filter(
                must_not=[FieldCondition(key="id", match=MatchValue(value=anime_id))]
            )

            # Use title_vector for similarity search
            if (
                isinstance(reference_vectors, dict)
                and "title_vector" in reference_vectors
            ):
                title_vector = reference_vectors["title_vector"]
                if is_float_vector(title_vector):
                    query_vector = NamedVector(
                        name="title_vector",
                        vector=title_vector,
                    )
                else:
                    logger.warning(
                        f"title_vector is not a valid float list: {type(title_vector)}"
                    )
                    return []
            else:
                logger.warning(f"No 'title_vector' found for anime: {anime_id}")
                return []

            search_result = await loop.run_in_executor(
                None,
                lambda: self.client.search(
                    collection_name=self.collection_name,
                    query_vector=query_vector,
                    query_filter=filter_out_self,
                    limit=limit,
                    with_payload=True,
                    with_vectors=False,
                ),
            )

            # Format results
            results = []
            for hit in search_result:
                result = dict(hit.payload) if hit.payload else {}
                result["similarity_score"] = hit.score
                results.append(result)

            return results

        except Exception as e:
            logger.error(f"Similar anime search failed: {e}")
            return []

    def _build_filter(self, filters: Dict[str, Any]) -> Optional[Filter]:
        """Build Qdrant filter from filter dictionary.

        Args:
            filters: Dictionary with filter conditions

        Returns:
            Qdrant Filter object
        """
        if not filters:
            return None

        conditions = []

        for key, value in filters.items():
            # Skip None values and empty collections
            if value is None:
                continue
            if isinstance(value, (list, tuple)) and len(value) == 0:
                continue
            if isinstance(value, dict) and len(value) == 0:
                continue

            if isinstance(value, dict):
                # Range filter
                if "gte" in value or "lte" in value or "gt" in value or "lt" in value:
                    conditions.append(FieldCondition(key=key, range=Range(**value)))
                # Match any filter
                elif "any" in value:
                    any_values = value["any"]
                    # Skip empty any values
                    if any_values and len(any_values) > 0:
                        conditions.append(
                            FieldCondition(key=key, match=MatchAny(any=any_values))
                        )
            elif isinstance(value, list):
                # Match any from list - only add if list is not empty
                if len(value) > 0:
                    conditions.append(
                        FieldCondition(key=key, match=MatchAny(any=value))
                    )
            else:
                # Exact match - only add if value is not None and is a valid type
                if value is not None and isinstance(value, (str, int, bool)):
                    conditions.append(
                        FieldCondition(key=key, match=MatchValue(value=value))
                    )

        return Filter(must=conditions) if conditions else None  # type: ignore[arg-type]

    async def get_by_id(self, anime_id: str) -> Optional[Dict[str, Any]]:
        """Get anime by ID.

        Args:
            anime_id: The anime ID to retrieve

        Returns:
            Anime data dictionary or None if not found
        """
        try:
            loop = asyncio.get_event_loop()
            point_id = self._generate_point_id(anime_id)

            # Retrieve point by ID
            points = await loop.run_in_executor(
                None,
                lambda: self.client.retrieve(
                    collection_name=self.collection_name,
                    ids=[point_id],
                    with_payload=True,
                    with_vectors=False,
                ),
            )

            if points:
                return dict(points[0].payload) if points[0].payload else {}
            return None

        except Exception as e:
            logger.error(f"Failed to get anime by ID {anime_id}: {e}")
            return None

    async def get_point(self, point_id: str) -> Optional[Dict[str, Any]]:
        """Get point by Qdrant point ID including vectors and payload.

        Args:
            point_id: The Qdrant point ID to retrieve

        Returns:
            Point data dictionary with vectors and payload or None if not found
        """
        try:
            loop = asyncio.get_event_loop()

            # Retrieve point by ID with vectors
            points = await loop.run_in_executor(
                None,
                lambda: self.client.retrieve(
                    collection_name=self.collection_name,
                    ids=[point_id],
                    with_vectors=True,
                    with_payload=True
                ),
            )

            if points:
                point = points[0]
                return {
                    "id": str(point.id),
                    "vector": point.vector,
                    "payload": dict(point.payload) if point.payload else {}
                }
            return None

        except Exception as e:
            logger.error(f"Failed to get point by ID {point_id}: {e}")
            return None

    async def find_similar(
        self, anime_id: str, limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Find similar anime using vector similarity.

        Args:
            anime_id: Reference anime ID
            limit: Number of similar anime to return

        Returns:
            List of similar anime with similarity scores
        """
        try:
            # First get the reference anime
            reference_anime = await self.get_by_id(anime_id)
            if not reference_anime:
                logger.warning(f"Reference anime not found: {anime_id}")
                return []

            # Use the title and tags for similarity search
            search_text = reference_anime.get("title", "")
            if reference_anime.get("tags"):
                search_text += " " + " ".join(reference_anime["tags"][:5])  # Limit tags

            # Perform similarity search
            loop = asyncio.get_event_loop()
            embedding = self.text_processor.encode_text(search_text)

            # Validate embedding
            if not embedding:
                logger.warning(
                    f"Failed to create embedding for search text: {search_text}"
                )
                return []

            # Filter to exclude the reference anime itself
            filter_out_self = Filter(
                must_not=[
                    FieldCondition(key="anime_id", match=MatchValue(value=anime_id))
                ]
            )

            # Use named vector search for multi-vector collection
            search_result = await loop.run_in_executor(
                None,
                lambda: self.client.search(
                    collection_name=self.collection_name,
                    query_vector=NamedVector(name="text", vector=embedding),
                    query_filter=filter_out_self,
                    limit=limit,
                    with_payload=True,
                    with_vectors=False,
                ),
            )

            # Format results
            results = []
            for hit in search_result:
                result = dict(hit.payload) if hit.payload else {}
                result["similarity_score"] = hit.score
                results.append(result)

            return results

        except Exception as e:
            logger.error(f"Failed to find similar anime for {anime_id}: {e}")
            return []

    async def clear_index(self) -> bool:
        """Clear all points from the collection (for fresh re-indexing)."""
        try:
            # Delete and recreate collection for clean state
            delete_success = await self.delete_collection()
            if not delete_success:
                return False

            create_success = await self.create_collection()
            if not create_success:
                return False

            logger.info(f"Cleared and recreated collection: {self.collection_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to clear index: {e}")
            return False

    async def delete_collection(self) -> bool:
        """Delete the anime collection (for testing/reset)."""
        try:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None, lambda: self.client.delete_collection(self.collection_name)
            )
            logger.info(f"Deleted collection: {self.collection_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete collection: {e}")
            return False

    async def create_collection(self) -> bool:
        """Create the anime collection."""
        try:
            self._initialize_collection()
            return True
        except Exception as e:
            logger.error(f"Failed to create collection: {e}")
            return False

    # async def search_by_image(
    #     self, image_data: str, limit: int = 10, use_hybrid_search: bool = True
    # ) -> List[Dict[str, Any]]:
    #     """Search for anime by image similarity using optimized hybrid search.

    #     Args:
    #         image_data: Base64 encoded image data
    #         limit: Maximum number of results
    #         use_hybrid_search: If True, uses modern hybrid search API for efficiency

    #     Returns:
    #         List of anime with visual similarity scores
    #     """
    #     try:
    #         # Create image embedding
    #         image_embedding = self.embedding_manager.vision_processor.encode_image(image_data)
    #         if image_embedding is None:
    #             logger.error("Failed to create image embedding")
    #             return []

    #         loop = asyncio.get_event_loop()

    #         # Use image_vector for search
    #         search_result = await loop.run_in_executor(
    #             None,
    #             lambda: self.client.search(
    #                 collection_name=self.collection_name,
    #                 query_vector=NamedVector(name="image_vector", vector=image_embedding),
    #                 limit=limit,
    #                 with_payload=True,
    #                 with_vectors=False,
    #             ),
    #         )

    #         # Format results
    #         results = []
    #         for hit in search_result:
    #             result = dict(hit.payload)
    #             result["visual_similarity_score"] = hit.score
    #             result["_id"] = hit.id
    #             results.append(result)

    #         return results

    #     except Exception as e:
    #         logger.error(f"Image search failed: {e}")
    #         return []

    async def search_multi_vector(
        self,
        vector_queries: List[Dict[str, Any]],
        limit: int = 10,
        fusion_method: str = "rrf",
        filters: Optional[Filter] = None,
    ) -> List[Dict[str, Any]]:
        """Search across multiple vectors using Qdrant's native multi-vector API.

        Args:
            vector_queries: List of vector query dicts with keys:
                - vector_name: Name of the vector to search (e.g., "title_vector")
                - vector_data: The query vector (list of floats)
                - weight: Optional weight for fusion (default: 1.0)
            limit: Maximum number of results to return
            fusion_method: Fusion algorithm - "rrf" or "dbsf"
            filters: Optional Qdrant filter conditions

        Returns:
            List of search results with fusion scores
        """
        try:
            if not vector_queries:
                raise ValueError("vector_queries cannot be empty")

            # Create prefetch queries for each vector
            prefetch_queries = []
            for query_config in vector_queries:
                vector_name = query_config["vector_name"]
                vector_data = query_config["vector_data"]

                # Create a Prefetch query for this vector
                prefetch_query = Prefetch(
                    using=vector_name,
                    query=vector_data,
                    limit=limit * 2,  # Get more results for better fusion
                    filter=filters,
                )
                prefetch_queries.append(prefetch_query)

            # Determine fusion method
            if fusion_method.lower() == "rrf":
                fusion = Fusion.RRF
            elif fusion_method.lower() == "dbsf":
                fusion = Fusion.DBSF
            else:
                logger.warning(f"Unknown fusion method {fusion_method}, using RRF")
                fusion = Fusion.RRF

            # Create the fusion query
            fusion_query = FusionQuery(fusion=fusion)

            # Execute the multi-vector search using query_points
            response = self.client.query_points(
                collection_name=self.collection_name,
                prefetch=prefetch_queries,
                query=fusion_query,
                limit=limit,
                with_payload=True,
                with_vectors=False,
            )

            # Convert response to our format
            results = []
            for point in response.points:
                payload = point.payload if point.payload else {}
                result = {
                    "id": str(point.id),
                    "anime_id": str(point.id),
                    "_id": str(point.id),
                    **payload,
                    # Search scores override any payload scores
                    "score": point.score,
                    "_score": point.score,
                    "fusion_score": point.score,
                }
                results.append(result)

            logger.info(
                f"Multi-vector search returned {len(results)} results using {fusion_method.upper()}"
            )
            return results

        except Exception as e:
            logger.error(f"Multi-vector search failed: {e}")
            raise

    async def search_text_comprehensive(
        self,
        query: str,
        limit: int = 10,
        fusion_method: str = "rrf",
        filters: Optional[Filter] = None,
    ) -> List[Dict[str, Any]]:
        """Search across all 12 text vectors using native Qdrant fusion.

        Args:
            query: Text search query
            limit: Maximum number of results
            fusion_method: Fusion algorithm - "rrf" or "dbsf"
            filters: Optional Qdrant filter conditions

        Returns:
            List of search results with comprehensive text similarity scores
        """
        try:
            # Generate text embedding once
            query_embedding = self.embedding_manager.text_processor.encode_text(query)
            if query_embedding is None:
                logger.warning(
                    "Failed to create embedding for comprehensive text search"
                )
                return []

            # All 12 text vectors for comprehensive search
            text_vector_names = [
                "title_vector",
                "character_vector",
                "genre_vector",
                "technical_vector",
                "staff_vector",
                "review_vector",
                "temporal_vector",
                "streaming_vector",
                "related_vector",
                "franchise_vector",
                "episode_vector",
                "identifiers_vector",
            ]

            # Create vector queries for all text vectors
            vector_queries = []
            for vector_name in text_vector_names:
                vector_queries.append(
                    {"vector_name": vector_name, "vector_data": query_embedding}
                )

            # Use native multi-vector search
            results = await self.search_multi_vector(
                vector_queries=vector_queries,
                limit=limit,
                fusion_method=fusion_method,
                filters=filters,
            )

            logger.info(
                f"Comprehensive text search returned {len(results)} results across {len(text_vector_names)} vectors"
            )
            return results

        except Exception as e:
            logger.error(f"Comprehensive text search failed: {e}")
            return []

    async def search_visual_comprehensive(
        self,
        image_data: str,
        limit: int = 10,
        fusion_method: str = "rrf",
        filters: Optional[Filter] = None,
    ) -> List[Dict[str, Any]]:
        """Search across both image vectors using native Qdrant fusion.

        Args:
            image_data: Base64 encoded image data
            limit: Maximum number of results
            fusion_method: Fusion algorithm - "rrf" or "dbsf"
            filters: Optional Qdrant filter conditions

        Returns:
            List of search results with comprehensive visual similarity scores
        """
        try:
            # Generate image embedding once
            image_embedding = self.embedding_manager.vision_processor.encode_image(
                image_data
            )
            if image_embedding is None:
                logger.error(
                    "Failed to create image embedding for comprehensive visual search"
                )
                return []

            # Both image vectors for comprehensive visual search
            image_vector_names = ["image_vector", "character_image_vector"]

            # Create vector queries for both image vectors
            vector_queries = []
            for vector_name in image_vector_names:
                vector_queries.append(
                    {"vector_name": vector_name, "vector_data": image_embedding}
                )

            # Use native multi-vector search
            results = await self.search_multi_vector(
                vector_queries=vector_queries,
                limit=limit,
                fusion_method=fusion_method,
                filters=filters,
            )

            logger.info(
                f"Comprehensive visual search returned {len(results)} results across {len(image_vector_names)} vectors"
            )
            return results

        except Exception as e:
            logger.error(f"Comprehensive visual search failed: {e}")
            return []

    async def search_complete(
        self,
        query: str,
        image_data: Optional[str] = None,
        limit: int = 10,
        fusion_method: str = "rrf",
        filters: Optional[Filter] = None,
    ) -> List[Dict[str, Any]]:
        """Search across all 14 vectors (12 text + 2 image) using native Qdrant fusion.

        Args:
            query: Text search query
            image_data: Optional base64 encoded image data
            limit: Maximum number of results
            fusion_method: Fusion algorithm - "rrf" or "dbsf"
            filters: Optional Qdrant filter conditions

        Returns:
            List of search results with complete multi-modal similarity scores
        """
        try:
            vector_queries = []

            # Generate text embedding for all 12 text vectors
            query_embedding = self.embedding_manager.text_processor.encode_text(query)
            if query_embedding is None:
                logger.warning("Failed to create text embedding for complete search")
            else:
                # All 12 text vectors
                text_vector_names = [
                    "title_vector",
                    "character_vector",
                    "genre_vector",
                    "technical_vector",
                    "staff_vector",
                    "review_vector",
                    "temporal_vector",
                    "streaming_vector",
                    "related_vector",
                    "franchise_vector",
                    "episode_vector",
                    "identifiers_vector",
                ]

                for vector_name in text_vector_names:
                    vector_queries.append(
                        {"vector_name": vector_name, "vector_data": query_embedding}
                    )

            # Add image vectors if image provided
            if image_data:
                image_embedding = self.embedding_manager.vision_processor.encode_image(
                    image_data
                )
                if image_embedding is None:
                    logger.warning(
                        "Failed to create image embedding for complete search"
                    )
                else:
                    # Both image vectors
                    image_vector_names = ["image_vector", "character_image_vector"]

                    for vector_name in image_vector_names:
                        vector_queries.append(
                            {"vector_name": vector_name, "vector_data": image_embedding}
                        )

            if not vector_queries:
                logger.error("No valid embeddings generated for complete search")
                return []

            # Use native multi-vector search across all vectors
            results = await self.search_multi_vector(
                vector_queries=vector_queries,
                limit=limit,
                fusion_method=fusion_method,
                filters=filters,
            )

            logger.info(
                f"Complete search returned {len(results)} results across {len(vector_queries)} vectors"
            )
            return results

        except Exception as e:
            logger.error(f"Complete search failed: {e}")
            return []

    async def search_characters(
        self,
        query: str,
        limit: int = 10,
        fusion_method: str = "rrf",
        filters: Optional[Filter] = None,
    ) -> List[Dict[str, Any]]:
        """Search specifically for character-related content.

        Args:
            query: Text search query focused on characters
            limit: Maximum number of results
            fusion_method: Fusion algorithm - "rrf" or "dbsf"
            filters: Optional Qdrant filter conditions

        Returns:
            List of search results focused on character similarity
        """
        try:
            # Generate text embedding
            query_embedding = self.embedding_manager.text_processor.encode_text(query)
            if query_embedding is None:
                logger.warning("Failed to create embedding for character search")
                return []

            # Character-focused vectors
            vector_queries = [
                {"vector_name": "character_vector", "vector_data": query_embedding},
                {
                    "vector_name": "title_vector",
                    "vector_data": query_embedding,
                },  # For context
            ]

            results = await self.search_multi_vector(
                vector_queries=vector_queries,
                limit=limit,
                fusion_method=fusion_method,
                filters=filters,
            )

            logger.info(f"Character search returned {len(results)} results")
            return results

        except Exception as e:
            logger.error(f"Character search failed: {e}")
            return []

    async def find_visually_similar_anime(
        self, anime_id: str, limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Find anime with similar visual style to reference anime.

        Args:
            anime_id: Reference anime ID
            limit: Number of similar anime to return

        Returns:
            List of visually similar anime with similarity scores
        """
        try:
            # Get reference anime's image vector
            point_id = self._generate_point_id(anime_id)
            loop = asyncio.get_event_loop()

            # Retrieve reference point with vectors
            reference_point = await loop.run_in_executor(
                None,
                lambda: self.client.retrieve(
                    collection_name=self.collection_name,
                    ids=[point_id],
                    with_vectors=True,
                ),
            )

            if not reference_point:
                logger.warning(f"Reference anime not found: {anime_id}")
                return []

            # Extract image vector
            reference_vectors = reference_point[0].vector
            if (
                isinstance(reference_vectors, dict)
                and "image_vector" in reference_vectors
            ):
                image_vector = reference_vectors["image_vector"]
                # Ensure image_vector is a list of floats
                if not is_float_vector(image_vector):
                    logger.warning(
                        f"image_vector is not a valid float list: {type(image_vector)}"
                    )
                    return []
                # Check if image vector is all zeros (no image processed)
                if all(v == 0.0 for v in image_vector):
                    logger.warning(
                        f"No image embeddings available for anime: {anime_id} (all zeros)"
                    )
                    return []
            else:
                logger.warning(f"No image_vector found for anime: {anime_id}")
                return []

            # Filter to exclude the reference anime itself
            filter_out_self = Filter(
                must_not=[FieldCondition(key="id", match=MatchValue(value=anime_id))]
            )

            # Search using image_vector
            search_result = await loop.run_in_executor(
                None,
                lambda: self.client.search(
                    collection_name=self.collection_name,
                    query_vector=NamedVector(name="image_vector", vector=image_vector),
                    query_filter=filter_out_self,
                    limit=limit,
                    with_payload=True,
                    with_vectors=False,
                ),
            )

            # Format results
            results = []
            for hit in search_result:
                result = dict(hit.payload) if hit.payload else {}
                result["visual_similarity_score"] = hit.score
                results.append(result)

            return results

        except Exception as e:
            logger.error(f"Visual similarity search failed: {e}")
            # Fallback to text-based similarity
            return await self.find_similar(anime_id, limit)
