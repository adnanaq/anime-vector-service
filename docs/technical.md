# Technical Documentation

## Development Environment and Stack

### Technology Choices

#### Core Framework
- **FastAPI 0.115+**: Chosen for high-performance async capabilities, automatic OpenAPI documentation, and excellent type safety with Pydantic
- **Python 3.12+**: Latest Python for performance improvements, improved type hints, and modern language features
- **Uvicorn**: ASGI server for production-grade async request handling

#### Vector Database
- **Qdrant 1.14+**: Selected for its superior performance with HNSW indexing, multi-vector support, and production-ready features
- **HNSW Algorithm**: Hierarchical Navigable Small World for fast approximate nearest neighbor search
- **Quantization Support**: Binary and scalar quantization for memory optimization

#### AI/ML Stack
- **BGE-M3 (BAAI/bge-m3)**: State-of-the-art multilingual embedding model with 384 dimensions, supporting 8192 token context
- **JinaCLIP v2 (jinaai/jina-clip-v2)**: Vision-language model for 512-dimensional image embeddings
- **Sentence Transformers 5.0+**: Simplified embedding pipeline with HuggingFace integration
- **PyTorch 2.0+**: Backend ML framework with optimized inference

### Development Setup

#### Prerequisites
**System Requirements:**
- Python 3.12+ for modern language features
- Docker and Docker Compose for containerization
- Git for version control
- 8GB+ RAM required for model loading
- GPU optional but recommended for image processing

#### Local Development
**Setup Process:**
1. Clone repository from version control
2. Install Python dependencies via requirements.txt
3. Start Qdrant database using Docker Compose
4. Run service using Python module execution
5. Access API documentation at localhost:8002/docs

#### Docker Development
**Container-based Development:**
- Full stack deployment using docker compose
- Service health verification via health endpoint
- Isolated development environment

### Configuration Management

#### Environment Variables
The service uses Pydantic Settings for type-safe configuration:

**Vector Service Configuration:**
- VECTOR_SERVICE_HOST: Service host address (default: 0.0.0.0)
- VECTOR_SERVICE_PORT: Service port (default: 8002)
- DEBUG: Enable debug mode (default: true)

**Qdrant Database Configuration:**
- QDRANT_URL: Database server URL (default: http://localhost:6333)
- QDRANT_COLLECTION_NAME: Collection name (default: anime_database)

**Embedding Models Configuration:**
- TEXT_EMBEDDING_MODEL: Text model (default: BAAI/bge-m3)
- IMAGE_EMBEDDING_MODEL: Image model (default: jinaai/jina-clip-v2)

**Performance Tuning:**
- QDRANT_ENABLE_QUANTIZATION: Enable quantization (default: false)
- MODEL_WARM_UP: Pre-load models (default: false)

#### Configuration Validation
- **Field Validation**: Pydantic validators ensure valid distance metrics, embedding providers, and log levels
- **Type Safety**: All configuration fields are strictly typed
- **Environment Override**: Settings can be overridden via environment variables or `.env` file

### Key Technical Decisions

#### Multi-Vector Architecture
- **Decision**: Store text, picture, and thumbnail vectors separately in same collection
- **Rationale**: Enables targeted search types while maintaining data locality
- **Implementation**: Named vectors in Qdrant with different dimensions

#### Embedding Model Selection
- **BGE-M3**: Chosen for multilingual support, large context window, and state-of-the-art performance
- **JinaCLIP v2**: Selected for superior vision-language understanding compared to OpenAI CLIP
- **Model Caching**: HuggingFace cache directory for faster subsequent loads

#### Async Architecture
- **FastAPI Async**: All endpoints are async for non-blocking I/O
- **Qdrant Async Client**: Ensures database operations don't block request handling
- **Lifespan Management**: Proper async initialization and cleanup

#### Error Handling Strategy
- **HTTP Exceptions**: Proper status codes with detailed error messages
- **Validation Errors**: Pydantic automatically handles request validation
- **Database Errors**: Graceful degradation when Qdrant is unavailable
- **Logging**: Structured logging with configurable levels

### Design Patterns in Use

#### Dependency Injection
- **Settings**: Cached settings instance using `@lru_cache`
- **Qdrant Client**: Global instance initialized during lifespan
- **Router Dependencies**: Future support for authentication/authorization

#### Factory Pattern
- **Client Creation**: QdrantClient factory with configuration-based initialization
- **Embedding Processors**: Factory methods for different embedding providers

#### Repository Pattern
- **Vector Operations**: Abstracted through QdrantClient interface
- **Data Models**: Pydantic models for request/response validation
- **Configuration**: Settings class encapsulates all configuration logic

#### Observer Pattern (Future)
- **Health Monitoring**: Health check observers for different components
- **Metrics Collection**: Performance metric observers

### Performance Optimization

#### Vector Database Optimization
- **HNSW Parameters**: Configurable `ef_construct` and `M` parameters for index tuning
- **Quantization**: Optional scalar/binary quantization for memory efficiency
- **Payload Indexing**: Indexed fields for fast metadata filtering
- **Memory Mapping**: Configurable threshold for disk vs memory storage

#### Model Performance
- **Model Warming**: Optional pre-loading during service startup
- **Cache Management**: HuggingFace model cache with configurable directory
- **Batch Processing**: Efficient batch embedding generation

#### API Performance
- **Async Processing**: Non-blocking request handling
- **Connection Pooling**: Efficient Qdrant client connection management
- **Response Compression**: Automatic FastAPI compression
- **CORS Optimization**: Configurable CORS settings

### Technical Constraints

#### Memory Constraints
- **Model Loading**: BGE-M3 + JinaCLIP v2 require ~4GB RAM combined
- **Vector Storage**: 384-dim + 2�512-dim vectors per anime = ~5KB per document
- **Index Memory**: HNSW index requires additional memory proportional to dataset size

#### Performance Constraints
- **Embedding Generation**: Text: ~50ms, Image: ~200ms per item
- **Vector Search**: ~10ms for 100K vectors with HNSW
- **Concurrent Limits**: ~100 simultaneous requests before degradation

#### Storage Constraints
- **Vector Size**: 100K anime � 5KB vectors = ~500MB vector storage
- **Payload Size**: Metadata adds ~2KB per anime document
- **Index Overhead**: HNSW index adds ~30% storage overhead

### Development Tools

#### Code Quality
- **Black**: Code formatting (configured in pyproject.toml)
- **isort**: Import sorting
- **autoflake**: Unused import removal
- **mypy**: Static type checking (future)

#### Testing Framework
- **pytest**: Unit and integration testing
- **pytest-asyncio**: Async test support
- **httpx**: HTTP client for API testing
- **pytest-mock**: Mocking for isolated tests

#### API Documentation
- **FastAPI OpenAPI**: Automatic API documentation
- **Swagger UI**: Interactive API explorer at `/docs`
- **ReDoc**: Alternative documentation at `/redoc`

#### Monitoring and Observability
- **Structured Logging**: JSON-formatted logs with timestamps
- **Health Endpoints**: `/health` for service and database status
- **Error Tracking**: Exception logging with context
- **Performance Metrics**: Response time logging (future Prometheus integration)

### Deployment Considerations

#### Container Optimization
- **Multi-stage Build**: Separate build and runtime stages
- **Layer Caching**: Optimized Dockerfile layer ordering
- **Security**: Non-root user, minimal base image
- **Size Optimization**: .dockerignore for build context reduction

#### Production Settings
- **Debug Mode**: Disabled in production
- **Logging**: INFO level with structured format
- **CORS**: Restricted origins for security
- **Health Checks**: Docker health check configuration

#### Scaling Considerations
- **Stateless Design**: No local state, suitable for horizontal scaling
- **Database Sharing**: Multiple instances can share same Qdrant cluster
- **Load Balancing**: Standard HTTP load balancing compatible
- **Resource Requirements**: 2 CPU cores, 4GB RAM per instance recommended

### Security Considerations

#### API Security
- **Input Validation**: Pydantic models prevent injection attacks
- **CORS Configuration**: Configurable origin restrictions
- **Error Information**: Careful error message exposure
- **Request Limits**: Configurable batch and search limits

#### Data Security
- **No Sensitive Data**: Only public anime metadata stored
- **TLS Termination**: HTTPS recommended for production
- **Access Logging**: Request logging for audit trails

#### Infrastructure Security
- **Container Security**: Non-root user, minimal privileges
- **Network Security**: Internal Qdrant communication
- **Secret Management**: Environment variable configuration
- **Update Strategy**: Regular security updates for dependencies

### Future Technical Enhancements

#### Phase 2 Technical Improvements
- **Redis Caching**: Query result caching layer
- **Prometheus Metrics**: Detailed performance monitoring
- **Authentication**: JWT-based API authentication
- **Rate Limiting**: Per-client request rate limiting

#### Phase 3 Advanced Features
- **Model Fine-tuning**: LoRA adaptation for anime-specific improvements
- **Distributed Deployment**: Multi-region deployment support
- **Stream Processing**: Real-time data pipeline integration
- **Advanced Analytics**: Query pattern analysis and optimization