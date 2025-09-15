"""Qdrant Vector Database Client for Anime Search

Provides high-performance vector search capabilities optimized for anime data
with advanced filtering, cross-platform ID lookups, and hybrid search.
"""

import asyncio
import hashlib
import logging
from typing import Any, Dict, List, Optional

# fastembed import moved to _init_encoder method for lazy loading
from qdrant_client import QdrantClient as QdrantSDK
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchAny,
    MatchValue,
    NamedVector,
    PointStruct,
    Range,
    VectorParams,
    # Qdrant optimization models
    BinaryQuantization,
    BinaryQuantizationConfig,
    ScalarQuantization,
    ScalarQuantizationConfig,
    ProductQuantization,
    QuantizationConfig,
    HnswConfig,
    HnswConfigDiff,
    OptimizersConfig,
    OptimizersConfigDiff,
    WalConfig,
    CollectionParams,
    PayloadSchemaType,
    ScalarType,
)

from ..config import Settings

logger = logging.getLogger(__name__)


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
            from ..config import get_settings

            settings = get_settings()

        self.settings = settings
        self.url = url or settings.qdrant_url
        self.collection_name = collection_name or settings.qdrant_collection_name
        self.client = QdrantSDK(url=self.url)
        self._vector_size = settings.qdrant_vector_size
        self._distance_metric = settings.qdrant_distance_metric
        self._image_vector_size = getattr(settings, "image_vector_size", 512)
        
        # Embedding processors
        self.text_processor = None
        self.vision_processor = None

        # Initialize processors
        self._init_processors()

        # Create collection if it doesn't exist
        self._initialize_collection()

    def _init_processors(self):
        """Initialize embedding processors."""
        try:
            # Import processors
            from .text_processor import TextProcessor
            from .vision_processor import VisionProcessor
            
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
    

    def _initialize_collection(self):
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
                optimizers_config = self._create_optimized_optimizers_config() or self._create_optimizers_config()
                wal_config = self._create_wal_config()

                # Create collection with optimization
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=vectors_config,
                    quantization_config=quantization_config,
                    optimizers_config=optimizers_config,
                    wal_config=wal_config
                )

                # Configure payload indexing for faster filtering
                if getattr(self.settings, "qdrant_enable_payload_indexing", True):
                    self._setup_payload_indexing()

                logger.info(f"Successfully created collection with {len(vectors_config)} vectors")
            else:
                # Validate existing collection compatibility
                if not self._validate_collection_compatibility():
                    logger.warning(f"Collection {self.collection_name} exists but may have compatibility issues")
                    logger.info("Continuing with existing collection configuration for backward compatibility")
                else:
                    logger.info(f"Collection {self.collection_name} validated successfully")

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
                    logger.warning(f"Collection missing expected vectors. Expected: {expected_vectors}, Found: {existing_vector_names}")
                    return False
            else:
                logger.warning("Collection uses single vector configuration, not compatible with multi-vector architecture")
                return False

        except Exception as e:
            logger.error(f"Failed to validate collection compatibility: {e}")
            return False

    def _validate_vector_config(self, vectors_config: Dict[str, VectorParams]):
        """Validate vector configuration before collection creation."""
        if not vectors_config:
            raise ValueError("Vector configuration is empty")

        expected_count = len(self.settings.vector_names)
        actual_count = len(vectors_config)

        if actual_count != expected_count:
            raise ValueError(f"Vector count mismatch: expected {expected_count}, got {actual_count}")

        # Validate vector dimensions
        for vector_name, vector_params in vectors_config.items():
            expected_dim = self.settings.vector_names.get(vector_name)
            if expected_dim and vector_params.size != expected_dim:
                raise ValueError(f"Vector {vector_name} dimension mismatch: expected {expected_dim}, got {vector_params.size}")

        logger.info(f"Vector configuration validated: {actual_count} vectors with correct dimensions")

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
                quantization_config=self._get_quantization_config(priority)
            )

        logger.info(f"Created 14-vector configuration with {len(vector_params)} vectors")
        return vector_params

    # NEW: Priority-Based Configuration Methods for Million-Query Optimization

    def _get_quantization_config(self, priority: str) -> Optional[QuantizationConfig]:
        """Get quantization config based on vector priority."""
        config = self.settings.quantization_config.get(priority, {})
        if config.get("type") == "scalar":
            scalar_config = ScalarQuantizationConfig(
                type=ScalarType.INT8,
                always_ram=config.get("always_ram", False)
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
            ef_construct=config.get("ef_construct", 200),
            m=config.get("m", 48)
        )

    def _get_vector_priority(self, vector_name: str) -> str:
        """Determine priority level for vector."""
        for priority, vectors in self.settings.vector_priorities.items():
            if vector_name in vectors:
                return priority
        return "medium"  # default

    def _create_optimized_optimizers_config(self) -> Optional[OptimizersConfigDiff]:
        """Create optimized optimizers configuration for million-query scale."""
        try:
            return OptimizersConfigDiff(
                default_segment_number=4,
                indexing_threshold=20000,
                memmap_threshold_kb=self.settings.memory_mapping_threshold_mb * 1024
            )
        except Exception as e:
            logger.error(f"Failed to create optimized optimizers config: {e}")
            return None

    def _create_quantization_config(self) -> Optional[QuantizationConfig]:
        """Create quantization configuration for performance optimization."""
        if not getattr(self.settings, "qdrant_enable_quantization", False):
            return None
            
        quantization_type = getattr(self.settings, "qdrant_quantization_type", "scalar")
        always_ram = getattr(self.settings, "qdrant_quantization_always_ram", None)
        
        try:
            if quantization_type == "binary":
                quantization = BinaryQuantization(always_ram=always_ram)
                logger.info("Enabling binary quantization for 40x speedup potential")
            elif quantization_type == "scalar":
                quantization = ScalarQuantization(
                    type="int8",  # 8-bit quantization for good balance
                    always_ram=always_ram
                )
                logger.info("Enabling scalar quantization for memory optimization")
            elif quantization_type == "product":
                quantization = ProductQuantization(
                    compression=16,  # Good compression ratio for anime vectors
                    always_ram=always_ram
                )
                logger.info("Enabling product quantization for storage optimization")
            else:
                logger.warning(f"Unknown quantization type: {quantization_type}")
                return None
                
            return QuantizationConfig(quantization)
        except Exception as e:
            logger.error(f"Failed to create quantization config: {e}")
            return None

    def _create_optimizers_config(self) -> Optional[OptimizersConfig]:
        """Create optimizers configuration for indexing performance."""
        try:
            optimizer_params = {}
            
            # Task #116: Configure memory mapping threshold
            memory_threshold = getattr(self.settings, "qdrant_memory_mapping_threshold", None)
            if memory_threshold:
                optimizer_params["memmap_threshold"] = memory_threshold
                
            # Configure indexing threads if specified
            indexing_threads = getattr(self.settings, "qdrant_hnsw_max_indexing_threads", None)
            if indexing_threads:
                optimizer_params["indexing_threshold"] = 0  # Start indexing immediately
                
            if optimizer_params:
                logger.info(f"Applying optimizer configuration: {optimizer_params}")
                return OptimizersConfig(**optimizer_params)
            return None
        except Exception as e:
            logger.error(f"Failed to create optimizers config: {e}")
            return None

    def _create_wal_config(self) -> Optional[WalConfig]:
        """Create Write-Ahead Logging configuration."""
        enable_wal = getattr(self.settings, "qdrant_enable_wal", None)
        if enable_wal is not None:
            try:
                config = WalConfig(wal_capacity_mb=32, wal_segments_ahead=0)
                logger.info(f"WAL configuration: enabled={enable_wal}")
                return config
            except Exception as e:
                logger.error(f"Failed to create WAL config: {e}")
        return None

    def _setup_payload_indexing(self):
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
        indexed_fields = getattr(self.settings, "qdrant_indexed_payload_fields", [])
        if not indexed_fields:
            logger.info("No payload indexing configured")
            return

        try:
            logger.info(f"Setting up payload indexing for {len(indexed_fields)} searchable fields")
            logger.info("Indexed fields enable fast filtering on: core metadata, genres, temporal data, platform stats")
            logger.info("Non-indexed fields (enrichment_metadata) stored for debugging but don't impact search performance")

            for field in indexed_fields:
                # Create index for each field to optimize filtering
                self.client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name=field,
                    field_schema=PayloadSchemaType.KEYWORD  # Most anime fields are keywords/strings
                )
                logger.debug(f"✓ Created payload index for searchable field: {field}")

            logger.info(f"Successfully indexed {len(indexed_fields)} searchable payload fields")
            logger.info("Payload optimization complete: fast filtering enabled, operational metadata preserved")

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

    def _create_image_embedding(self, image_data: str) -> Optional[List[float]]:
        """Create image embedding vector using modern vision processor.

        Args:
            image_data: Base64 encoded image data

        Returns:
            Image embedding vector or None if processing fails
        """
        try:
            if not image_data or not image_data.strip():
                logger.warning("Empty image data provided for embedding")
                return [0.0] * self._image_vector_size

            # Use modern vision processor
            embedding = self.vision_processor.encode_image(image_data)

            if embedding is None:
                logger.warning("No embedding generated for image")
                return [0.0] * self._image_vector_size

            # Ensure correct dimensions
            if len(embedding) != self._image_vector_size:
                logger.warning(
                    f"Image embedding size mismatch: got {len(embedding)}, expected {self._image_vector_size}"
                )
                # Pad or truncate to match expected size
                if len(embedding) < self._image_vector_size:
                    embedding.extend([0.0] * (self._image_vector_size - len(embedding)))
                else:
                    embedding = embedding[: self._image_vector_size]

            return embedding

        except Exception as e:
            logger.error(f"Failed to create image embedding: {e}")
            # Return zero vector on error to prevent pipeline failure
            return [0.0] * self._image_vector_size

    def _create_embedding(self, text: str) -> List[float]:
        """Create semantic embedding vector from text using modern text processor."""
        try:
            if not text or not text.strip():
                logger.warning("Empty text provided for embedding")
                # Return zero vector for empty text
                return [0.0] * self._vector_size

            # Use modern text processor
            embedding = self.text_processor.encode_text(text.strip())
            
            if embedding is None:
                logger.warning(f"No embedding generated for text: {text[:100]}...")
                return [0.0] * self._vector_size

            # Ensure correct dimensions
            if len(embedding) != self._vector_size:
                logger.warning(
                    f"Embedding size mismatch: got {len(embedding)}, expected {self._vector_size}"
                )
                # Pad or truncate to match expected size
                if len(embedding) < self._vector_size:
                    embedding.extend([0.0] * (self._vector_size - len(embedding)))
                else:
                    embedding = embedding[: self._vector_size]

            return embedding

        except Exception as e:
            logger.error(f"Failed to create embedding for text '{text[:100]}...': {e}")
            # Return zero vector on error to prevent pipeline failure
            return [0.0] * self._vector_size

    def _generate_point_id(self, anime_id: str) -> str:
        """Generate unique point ID from anime ID."""
        return hashlib.md5(anime_id.encode()).hexdigest()

    async def add_documents(
        self, documents: List[Dict[str, Any]], batch_size: int = 100
    ) -> bool:
        """Add anime documents to the collection.

        Args:
            documents: List of anime documents
            batch_size: Number of documents to process per batch

        Returns:
            True if successful, False otherwise
        """
        try:
            loop = asyncio.get_event_loop()
            total_docs = len(documents)

            logger.info(f"Adding {total_docs} documents in batches of {batch_size}")

            for i in range(0, total_docs, batch_size):
                batch = documents[i : i + batch_size]
                points = []

                for doc in batch:
                    try:
                        # Create text embedding from embedding_text
                        embedding_text = doc.get("embedding_text", "")
                        if not embedding_text:
                            logger.warning(
                                f"Empty embedding_text for anime_id: {doc.get('anime_id')}"
                            )
                            continue

                        text_embedding = self._create_embedding(embedding_text)
                        point_id = self._generate_point_id(doc["anime_id"])

                        # Prepare payload (exclude processed fields)
                        payload = {
                            k: v
                            for k, v in doc.items()
                            if k
                            not in ("embedding_text", "picture_data", "thumbnail_data")
                        }

                        # Create multi-vector point with text + picture + thumbnail vectors
                        vectors = {"text": text_embedding}

                        # Add picture vector if picture data available
                        picture_data = doc.get("picture_data")
                        if picture_data:
                            picture_embedding = self._create_image_embedding(
                                picture_data
                            )
                            if picture_embedding:
                                vectors["picture"] = picture_embedding
                            else:
                                vectors["picture"] = [0.0] * self._image_vector_size
                        else:
                            # Use zero vector for missing picture
                            vectors["picture"] = [0.0] * self._image_vector_size

                        # Add thumbnail vector if thumbnail data available
                        thumbnail_data = doc.get("thumbnail_data")
                        if thumbnail_data:
                            thumbnail_embedding = self._create_image_embedding(
                                thumbnail_data
                            )
                            if thumbnail_embedding:
                                vectors["thumbnail"] = thumbnail_embedding
                            else:
                                vectors["thumbnail"] = [0.0] * self._image_vector_size
                        else:
                            # Use zero vector for missing thumbnail
                            vectors["thumbnail"] = [0.0] * self._image_vector_size

                        point = PointStruct(
                            id=point_id, vector=vectors, payload=payload
                        )

                        points.append(point)

                    except Exception as e:
                        logger.error(
                            f"Failed to process document {doc.get('anime_id')}: {e}"
                        )
                        continue

                if points:
                    # Upload batch
                    await loop.run_in_executor(
                        None,
                        lambda: self.client.upsert(
                            collection_name=self.collection_name, points=points
                        ),
                    )

                    logger.info(
                        f"Uploaded batch {i//batch_size + 1}/{(total_docs-1)//batch_size + 1} ({len(points)} points)"
                    )

            logger.info(f"Successfully added {total_docs} documents")
            return True

        except Exception as e:
            logger.error(f"Failed to add documents: {e}")
            return False

    async def search(
        self, query: str, limit: int = 20, filters: Optional[Dict] = None
    ) -> List[Dict[str, Any]]:
        """Perform semantic search on anime collection.

        Args:
            query: Search query text
            limit: Maximum number of results
            filters: Optional filters for metadata

        Returns:
            List of search results with anime data and scores
        """
        try:
            # Create query embedding
            query_embedding = self._create_embedding(query)

            # Build filter if provided
            qdrant_filter = None
            if filters:
                qdrant_filter = self._build_filter(filters)
            print(f"Qdrant filter: {qdrant_filter}")
            loop = asyncio.get_event_loop()

            # Perform search using named vector for multi-vector collection
            search_result = await loop.run_in_executor(
                None,
                lambda: self.client.search(
                    collection_name=self.collection_name,
                    query_vector=NamedVector(name="text", vector=query_embedding),
                    query_filter=qdrant_filter,
                    limit=limit,
                    with_payload=True,
                    with_vectors=False,
                ),
            )

            # Format results
            results = []
            for hit in search_result:
                result = dict(hit.payload)
                result["_score"] = hit.score
                result["_id"] = hit.id
                results.append(result)

            return results

        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []

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
            reference_vector = reference_point[0].vector

            # Search excluding the reference anime itself
            filter_out_self = Filter(
                must_not=[
                    FieldCondition(key="anime_id", match=MatchValue(value=anime_id))
                ]
            )

            # Use text vector for similarity search (multi-vector collection)
            if isinstance(reference_vector, dict):
                query_vector = NamedVector(
                    name="text",
                    vector=reference_vector.get(
                        "text", reference_vector.get("picture", [])
                    ),
                )
            else:
                # Legacy case - shouldn't happen with multi-vector
                logger.warning("Found non-dict vector in multi-vector collection")
                query_vector = reference_vector

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
                result = dict(hit.payload)
                result["similarity_score"] = hit.score
                results.append(result)

            return results

        except Exception as e:
            logger.error(f"Similar anime search failed: {e}")
            return []

    def _build_filter(self, filters: Dict) -> Filter:
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
                # Exact match - only add if value is not None
                if value is not None:
                    conditions.append(
                        FieldCondition(key=key, match=MatchValue(value=value))
                    )

        return Filter(must=conditions) if conditions else None

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
                return dict(points[0].payload)
            return None

        except Exception as e:
            logger.error(f"Failed to get anime by ID {anime_id}: {e}")
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
            embedding = self._create_embedding(search_text)

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
                result = dict(hit.payload)
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

    async def search_by_image(
        self, image_data: str, limit: int = 10, use_hybrid_search: bool = True
    ) -> List[Dict[str, Any]]:
        """Search for anime by image similarity using optimized hybrid search.

        Args:
            image_data: Base64 encoded image data
            limit: Maximum number of results
            use_hybrid_search: If True, uses modern hybrid search API for efficiency

        Returns:
            List of anime with visual similarity scores
        """
        try:
            # Create image embedding
            image_embedding = self._create_image_embedding(image_data)
            if not image_embedding:
                logger.error("Failed to create image embedding")
                return []

            loop = asyncio.get_event_loop()

            if use_hybrid_search:
                # Task #116: Use modern hybrid search API (single request vs multiple)
                try:
                    # Use batch search for multiple vectors in single request
                    search_requests = [
                        {
                            "vector": NamedVector(name="picture", vector=image_embedding),
                            "limit": limit,
                            "with_payload": True,
                            "with_vectors": False,
                        },
                        {
                            "vector": NamedVector(name="thumbnail", vector=image_embedding),
                            "limit": limit,
                            "with_payload": True,
                            "with_vectors": False,
                        }
                    ]
                    
                    # Execute hybrid search
                    hybrid_results = await loop.run_in_executor(
                        None,
                        lambda: self.client.search_batch(
                            collection_name=self.collection_name,
                            requests=search_requests
                        ),
                    )

                    # Process hybrid results
                    picture_results = hybrid_results[0] if len(hybrid_results) > 0 else []
                    thumbnail_results = hybrid_results[1] if len(hybrid_results) > 1 else []

                    # Combine results with weighted scoring (favor picture over thumbnail)
                    picture_scores = {
                        hit.payload.get("anime_id", hit.id): hit.score
                        for hit in picture_results
                    }
                    thumbnail_scores = {
                        hit.payload.get("anime_id", hit.id): hit.score
                        for hit in thumbnail_results
                    }

                    # Weighted combination: 70% picture, 30% thumbnail
                    combined_results = {}
                    all_anime_ids = set(picture_scores.keys()) | set(thumbnail_scores.keys())

                    for anime_id in all_anime_ids:
                        picture_score = picture_scores.get(anime_id, 0.0)
                        thumbnail_score = thumbnail_scores.get(anime_id, 0.0)

                        # Combined score with higher weight on picture
                        if picture_score > 0 or thumbnail_score > 0:
                            combined_score = 0.7 * picture_score + 0.3 * thumbnail_score
                            combined_results[anime_id] = combined_score

                    # Sort by combined score and get top results
                    sorted_anime_ids = sorted(
                        combined_results.keys(),
                        key=lambda x: combined_results[x],
                        reverse=True,
                    )[:limit]

                    # Get full anime data for results
                    results = []
                    for anime_id in sorted_anime_ids:
                        # Find the anime data from either result set
                        anime_data = None
                        for hit in picture_results + thumbnail_results:
                            if hit.payload.get("anime_id", hit.id) == anime_id:
                                anime_data = dict(hit.payload)
                                break

                        if anime_data:
                            anime_data["visual_similarity_score"] = combined_results[anime_id]
                            anime_data["picture_score"] = picture_scores.get(anime_id, 0.0)
                            anime_data["thumbnail_score"] = thumbnail_scores.get(anime_id, 0.0)
                            anime_data["_id"] = anime_id
                            results.append(anime_data)

                    logger.info(
                        f"Hybrid image search: {len(picture_results)} picture + {len(thumbnail_results)} thumbnail = {len(results)} final results"
                    )
                    return results

                except Exception as hybrid_error:
                    logger.warning(f"Hybrid search failed, falling back to individual searches: {hybrid_error}")
                    # Fall back to individual searches if hybrid search not available

            # Fallback: Individual searches (legacy mode or hybrid search unavailable)
            picture_results = await loop.run_in_executor(
                None,
                lambda: self.client.search(
                    collection_name=self.collection_name,
                    query_vector=NamedVector(name="picture", vector=image_embedding),
                    limit=limit,
                    with_payload=True,
                    with_vectors=False,
                ),
            )

            # Format results
            results = []
            for hit in picture_results:
                result = dict(hit.payload)
                result["visual_similarity_score"] = hit.score
                result["_id"] = hit.id
                results.append(result)

            return results

        except Exception as e:
            logger.error(f"Image search failed: {e}")
            return []

    async def search_multimodal(
        self,
        query: str,
        image_data: Optional[str] = None,
        limit: int = 10,
        text_weight: float = 0.7,
    ) -> List[Dict[str, Any]]:
        """Search using both text and image queries with weighted combination.

        Args:
            query: Text search query
            image_data: Optional base64 encoded image data
            limit: Maximum number of results
            text_weight: Weight for text similarity (0.0-1.0), image gets (1-text_weight)

        Returns:
            List of anime with combined similarity scores
        """
        try:
            # If no image provided, just do text search
            if not image_data:
                logger.info("No image provided, performing text-only search")
                return await self.search(query, limit)

            # Get text results
            text_results = await self.search(query, limit * 2)  # Get more for fusion
            text_scores = {
                r.get("anime_id", r.get("_id", "")): r.get("_score", 0.0)
                for r in text_results
            }

            # Get image results
            image_results = await self.search_by_image(image_data, limit * 2)
            image_scores = {
                r.get("anime_id", r.get("_id", "")): r.get(
                    "visual_similarity_score", r.get("score", 0.0)
                )
                for r in image_results
            }

            # Debug logging
            logger.info(f"Text search returned {len(text_results)} results")
            logger.info(f"Image search returned {len(image_results)} results")

            # Combine results with weighted scoring
            combined_results = {}
            all_anime_ids = set(text_scores.keys()) | set(image_scores.keys())
            all_anime_ids.discard("")  # Remove empty IDs

            for anime_id in all_anime_ids:
                text_score = text_scores.get(anime_id, 0.0)
                image_score = image_scores.get(anime_id, 0.0)

                # Weighted combination - ensure at least one score exists
                if text_score > 0 or image_score > 0:
                    combined_score = (
                        text_weight * text_score + (1 - text_weight) * image_score
                    )
                    combined_results[anime_id] = combined_score

            logger.info(f"Combined {len(combined_results)} unique results")

            # Sort by combined score and get top results
            if not combined_results:
                logger.warning("No combined results, falling back to text search")
                return await self.search(query, limit)

            sorted_anime_ids = sorted(
                combined_results.keys(), key=lambda x: combined_results[x], reverse=True
            )[:limit]

            # Fetch full anime data for top results
            final_results = []
            for anime_id in sorted_anime_ids:
                # Try to get data from existing results first to avoid extra queries
                anime_data = None
                for result in text_results + image_results:
                    if (
                        result.get("anime_id") == anime_id
                        or result.get("_id") == anime_id
                    ):
                        anime_data = dict(result)
                        break

                # If not found in results, query directly
                if not anime_data:
                    anime_data = await self.get_by_id(anime_id)

                if anime_data:
                    anime_data["multimodal_score"] = combined_results[anime_id]
                    anime_data["text_score"] = text_scores.get(anime_id, 0.0)
                    anime_data["image_score"] = image_scores.get(anime_id, 0.0)
                    anime_data["_score"] = combined_results[anime_id]  # For consistency
                    final_results.append(anime_data)

            logger.info(f"Returning {len(final_results)} multimodal results")
            return final_results

        except Exception as e:
            logger.error(f"Multimodal search failed: {e}")
            # Fallback to text-only search
            return await self.search(query, limit)

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
            if isinstance(reference_vectors, dict) and "picture" in reference_vectors:
                image_vector = reference_vectors["picture"]
                # Check if image vector is all zeros (no image processed)
                if all(v == 0.0 for v in image_vector):
                    logger.warning(
                        f"No image embeddings available for anime: {anime_id} (all zeros)"
                    )
                    return []
            else:
                logger.warning(f"No picture vector found for anime: {anime_id}")
                return []

            # Filter to exclude the reference anime itself
            filter_out_self = Filter(
                must_not=[
                    FieldCondition(key="anime_id", match=MatchValue(value=anime_id))
                ]
            )

            # Search using picture vector
            search_result = await loop.run_in_executor(
                None,
                lambda: self.client.search(
                    collection_name=self.collection_name,
                    query_vector=NamedVector(name="picture", vector=image_vector),
                    query_filter=filter_out_self,
                    limit=limit,
                    with_payload=True,
                    with_vectors=False,
                ),
            )

            # Format results
            results = []
            for hit in search_result:
                result = dict(hit.payload)
                result["visual_similarity_score"] = hit.score
                results.append(result)

            return results

        except Exception as e:
            logger.error(f"Visual similarity search failed: {e}")
            # Fallback to text-based similarity
            return await self.find_similar(anime_id, limit)
