"""Vector Service Configuration Settings."""

from functools import lru_cache
from typing import List, Optional

from pydantic import ConfigDict, Field, field_validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Vector service settings with validation and type safety."""

    model_config = ConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )

    # Vector Service Configuration
    vector_service_host: str = Field(
        default="0.0.0.0", 
        description="Vector service host address"
    )
    vector_service_port: int = Field(
        default=8002, 
        ge=1, 
        le=65535, 
        description="Vector service port"
    )
    debug: bool = Field(
        default=True, 
        description="Enable debug mode"
    )

    # Qdrant Vector Database Configuration
    qdrant_url: str = Field(
        default="http://localhost:6333", 
        description="Qdrant server URL"
    )
    qdrant_collection_name: str = Field(
        default="anime_database", 
        description="Qdrant collection name"
    )
    qdrant_vector_size: int = Field(
        default=1024,
        description="Vector embedding dimensions for text vectors (BGE-M3)"
    )
    qdrant_distance_metric: str = Field(
        default="cosine", 
        description="Distance metric for similarity"
    )

    # Multi-Vector Configuration
    image_vector_size: int = Field(
        default=1024,
        description="Image embedding dimensions (JinaCLIP v2: 1024, CLIP: 512)"
    )

    # 14-Vector Semantic Architecture Configuration
    vector_names: dict = Field(
        default={
            "title_vector": 1024,
            "character_vector": 1024,
            "genre_vector": 1024,
            "technical_vector": 1024,
            "staff_vector": 1024,
            "review_vector": 1024,
            "temporal_vector": 1024,
            "streaming_vector": 1024,
            "related_vector": 1024,
            "franchise_vector": 1024,
            "episode_vector": 1024,
            "identifiers_vector": 1024,
            "image_vector": 1024,
            "character_image_vector": 1024
        },
        description="14-vector semantic architecture with named vectors and dimensions (BGE-M3: 1024-dim, JinaCLIP v2: 1024-dim)"
    )

    # Vector Priority Classification for Optimization
    vector_priorities: dict = Field(
        default={
            "high": ["title_vector", "character_vector", "genre_vector", "review_vector", "image_vector", "character_image_vector"],
            "medium": ["technical_vector", "staff_vector", "temporal_vector", "streaming_vector"],
            "low": ["related_vector", "franchise_vector", "episode_vector", "identifiers_vector"]
        },
        description="Vector priority classification for performance optimization"
    )

    # Modern Embedding Configuration
    text_embedding_provider: str = Field(
        default="huggingface", 
        description="Text embedding provider: fastembed, huggingface, sentence-transformers"
    )
    text_embedding_model: str = Field(
        default="BAAI/bge-m3", 
        description="Modern text embedding model name"
    )
    
    image_embedding_provider: str = Field(
        default="jinaclip", 
        description="Image embedding provider: clip, siglip, jinaclip"
    )
    image_embedding_model: str = Field(
        default="jinaai/jina-clip-v2", 
        description="Modern image embedding model name"
    )

    # Model-Specific Configuration
    bge_model_version: str = Field(
        default="m3", 
        description="BGE model version: v1.5, m3, reranker"
    )
    bge_model_size: str = Field(
        default="base", 
        description="BGE model size: small, base, large"
    )
    bge_max_length: int = Field(
        default=8192, 
        description="BGE maximum input sequence length"
    )
    
    jinaclip_input_resolution: int = Field(
        default=512, 
        description="JinaCLIP input image resolution"
    )
    jinaclip_text_max_length: int = Field(
        default=77, 
        description="JinaCLIP maximum text sequence length"
    )
    
    model_cache_dir: Optional[str] = Field(
        default=None, 
        description="Custom cache directory for embedding models"
    )
    model_warm_up: bool = Field(
        default=False, 
        description="Pre-load and warm up models during initialization"
    )

    # Qdrant Performance Optimization
    qdrant_enable_quantization: bool = Field(
        default=False,
        description="Enable quantization for performance"
    )
    qdrant_quantization_type: str = Field(
        default="scalar",
        description="Quantization type: binary, scalar, product"
    )
    qdrant_quantization_always_ram: Optional[bool] = Field(
        default=None,
        description="Keep quantized vectors in RAM"
    )

    # Advanced Quantization Configuration per Vector Priority
    quantization_config: dict = Field(
        default={
            "high": {
                "type": "scalar",
                "scalar_type": "int8",
                "always_ram": True
            },
            "medium": {
                "type": "scalar",
                "scalar_type": "int8",
                "always_ram": False
            },
            "low": {
                "type": "binary",
                "always_ram": False
            }
        },
        description="Quantization configuration per vector priority for memory optimization"
    )
    
    # HNSW Configuration
    qdrant_hnsw_ef_construct: Optional[int] = Field(
        default=None,
        description="HNSW ef_construct parameter"
    )
    qdrant_hnsw_m: Optional[int] = Field(
        default=None,
        description="HNSW M parameter"
    )
    qdrant_hnsw_max_indexing_threads: Optional[int] = Field(
        default=None,
        description="Maximum indexing threads"
    )

    # Anime-Optimized HNSW Parameters per Vector Priority
    hnsw_config: dict = Field(
        default={
            "high": {
                "ef_construct": 256,
                "m": 64,
                "ef": 128
            },
            "medium": {
                "ef_construct": 200,
                "m": 48,
                "ef": 64
            },
            "low": {
                "ef_construct": 128,
                "m": 32,
                "ef": 32
            }
        },
        description="Anime-optimized HNSW parameters per vector priority for similarity matching"
    )
    
    # Memory and Storage Configuration
    qdrant_memory_mapping_threshold: Optional[int] = Field(
        default=None,
        description="Memory mapping threshold in KB"
    )

    # Advanced Memory Management for Million-Query Optimization
    memory_mapping_threshold_mb: int = Field(
        default=50,
        description="Memory mapping threshold in MB for large collection optimization"
    )
    qdrant_enable_wal: Optional[bool] = Field(
        default=None, 
        description="Enable Write-Ahead Logging"
    )
    
    # Payload Indexing
    qdrant_enable_payload_indexing: bool = Field(
        default=True, 
        description="Enable payload field indexing"
    )
    qdrant_indexed_payload_fields: List[str] = Field(
        default=[
            # Core searchable fields
            "id", "title", "type", "status", "episodes", "rating", "nsfw",
            # Categorical fields
            "genres", "tags", "demographics", "content_warnings",
            # Temporal fields
            "anime_season", "duration",
            # Platform fields
            "sources",
            # Statistics for numerical filtering
            "statistics", "score"
            # Note: enrichment_metadata intentionally excluded (non-indexed operational data)
        ],
        description="Payload fields to index for search filtering (excludes operational metadata)"
    )

    # API Configuration
    api_title: str = Field(
        default="Anime Vector Service", 
        description="API title"
    )
    api_version: str = Field(
        default="1.0.0", 
        description="API version"
    )
    api_description: str = Field(
        default="Microservice for anime vector database operations", 
        description="API description"
    )

    # Batch Processing Configuration
    default_batch_size: int = Field(
        default=100, 
        ge=1, 
        le=1000, 
        description="Default batch size for operations"
    )
    max_batch_size: int = Field(
        default=500, 
        ge=1, 
        le=2000, 
        description="Maximum allowed batch size"
    )

    # Request Limits
    max_search_limit: int = Field(
        default=100, 
        ge=1, 
        le=1000, 
        description="Maximum search results limit"
    )
    request_timeout: int = Field(
        default=30, 
        ge=1, 
        le=300, 
        description="Request timeout in seconds"
    )

    # Logging Configuration
    log_level: str = Field(
        default="INFO", 
        description="Logging level"
    )
    log_format: str = Field(
        default="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        description="Log format string"
    )

    # CORS Configuration
    allowed_origins: List[str] = Field(
        default=["*"], 
        description="Allowed CORS origins"
    )
    allowed_methods: List[str] = Field(
        default=["*"], 
        description="Allowed HTTP methods"
    )
    allowed_headers: List[str] = Field(
        default=["*"], 
        description="Allowed HTTP headers"
    )

    @field_validator("qdrant_distance_metric")
    @classmethod
    def validate_distance_metric(cls, v: str) -> str:
        """Validate distance metric."""
        valid_metrics = ["cosine", "euclid", "dot"]
        if v.lower() not in valid_metrics:
            raise ValueError(f"Distance metric must be one of: {valid_metrics}")
        return v.lower()

    @field_validator("text_embedding_provider")
    @classmethod
    def validate_text_provider(cls, v: str) -> str:
        """Validate text embedding provider."""
        valid_providers = ["fastembed", "huggingface", "sentence-transformers"]
        if v.lower() not in valid_providers:
            raise ValueError(f"Text embedding provider must be one of: {valid_providers}")
        return v.lower()

    @field_validator("image_embedding_provider")
    @classmethod
    def validate_image_provider(cls, v: str) -> str:
        """Validate image embedding provider."""
        valid_providers = ["clip", "siglip", "jinaclip"]
        if v.lower() not in valid_providers:
            raise ValueError(f"Image embedding provider must be one of: {valid_providers}")
        return v.lower()

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        """Validate log level."""
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in valid_levels:
            raise ValueError(f"Log level must be one of: {valid_levels}")
        return v.upper()


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()