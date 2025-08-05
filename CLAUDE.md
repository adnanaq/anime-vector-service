# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Rules System - Compressed Protocols (ALWAYS Follow)

### Mode Tokens
- `@PLAN_MODE` → Load: docs/{architecture,product_requirement_docs,technical}.md + tasks/{active_context,tasks_plan}.md → Clarify → Strategy → Validate → Document
- `@CODE_MODE` → Load: core files + src/ context → Analyze deps → Plan changes → Simulate → Test → Document

### Protocol Tokens  
- `@PRE_IMPL` → Read docs/ + tasks/ + get src/ context + dependency analysis + flow analysis
- `@ARCH_VALID` → Parse mermaid from docs/architecture.md → validate boundaries/flow/interfaces → STOP if missing/fail
- `@SIM_TEST` → Dry run changes → validate no breakage → generate feedback → fix before implement
- `@MEM_UPDATE` → Review 7 core files → update active_context.md + tasks_plan.md → lessons-learned.md + error-documentation.md

### Intelligence Tokens
- `@LESSONS` → Apply patterns from rules/lessons-learned.md (async-first, config-driven, multi-vector, graceful degradation)
- `@ERRORS` → Check rules/error-documentation.md for similar issues → apply known resolutions
- `@ANTI_PATTERNS` → Avoid: premature optimization, config sprawl, monolithic loading, feature creep

### Execution Pattern
```
User Request → Mode Detection → Load @[MODE]_MODE → Apply @PRE_IMPL → @ARCH_VALID → 
Execute with @LESSONS + @ERRORS → @SIM_TEST → Implement → @MEM_UPDATE
```

**Usage**: Reference tokens (e.g., `@PLAN_MODE`, `@PRE_IMPL`) trigger full protocol expansion from rules/ files.

## Repository Overview

This is a specialized microservice for semantic search over anime content using vector embeddings and Qdrant database. The service provides text, image, and multimodal search capabilities with production-ready features including health checks, monitoring, and CORS support.

## Development Commands

### Local Development Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Start Qdrant database only
docker compose -f docker/docker-compose.yml up -d qdrant

# Run service locally for development
python -m src.main
```

### Docker Development (Recommended)
```bash
# Start full stack (service + database)
docker compose -f docker/docker-compose.yml up -d

# View logs
docker compose -f docker/docker-compose.yml logs -f vector-service

# Stop services
docker compose -f docker/docker-compose.yml down
```

### Testing and Quality
```bash
# Run tests
pytest

# Run specific test file
pytest test/test_filename.py

# Run tests with coverage
pytest --cov=src

# Code formatting
black src/
isort src/
autoflake --remove-all-unused-imports --in-place --recursive src/
```

### Service Health Checks
```bash
# Check service health
curl http://localhost:8002/health

# Check Qdrant health
curl http://localhost:6333/health

# Get database statistics
curl http://localhost:8002/api/v1/admin/stats
```

## Architecture Overview

### Core Architecture Pattern
The service follows a layered microservice architecture with clear separation of concerns:

**API Layer** (`src/api/`) → **Processing Layer** (`src/vector/`) → **Database Layer** (Qdrant)

### Key Architectural Components

#### 1. FastAPI Application (`src/main.py`)
- Async application with lifespan management
- Global Qdrant client initialization with health checks
- CORS middleware and structured logging
- Graceful startup/shutdown with dependency validation

#### 2. Configuration System (`src/config/settings.py`)
- Pydantic-based settings with environment variable support
- Comprehensive validation for all configuration parameters
- Support for multiple embedding providers and models
- Performance tuning parameters (quantization, HNSW, batch sizes)

#### 3. Multi-Vector Processing (`src/vector/`)
- **QdrantClient**: Advanced vector database operations with quantization support
- **TextProcessor**: BGE-M3 embeddings for semantic text search (384-dim)
- **VisionProcessor**: JinaCLIP v2 embeddings for image search (512-dim)
- **Fine-tuning modules**: Character recognition, art style classification, genre enhancement

#### 4. API Endpoints (`src/api/`)
- **Search Router**: Text, image, and multimodal search endpoints
- **Similarity Router**: Content-based and visual similarity operations
- **Admin Router**: Database management, statistics, and reindexing

#### 5. Data Enrichment Pipeline (`src/enrichment/`)
- **API Helpers**: Integration with 6+ external anime APIs (AniList, Kitsu, AniDB, etc.)
- **Scrapers**: Web scraping with Cloudflare bypass capabilities
- **Multi-stage AI Pipeline**: Modular prompt system for data enhancement

### Multi-Vector Collection Design
The service uses a single Qdrant collection with named vectors:
- `text`: 384-dimensional BGE-M3 embeddings for semantic search
- `picture`: 512-dimensional JinaCLIP v2 embeddings for cover art
- `thumbnail`: 512-dimensional JinaCLIP v2 embeddings for thumbnails

This design enables efficient multimodal search while maintaining data locality and reducing storage overhead.

### Configuration-Driven Model Selection
The service supports multiple embedding providers through configuration:
- **Text Models**: BGE-M3, BGE-small/base/large-v1.5, custom HuggingFace models
- **Vision Models**: JinaCLIP v2, CLIP ViT-B/32, SigLIP-384
- **Provider Flexibility**: Easy switching between embedding providers per modality

### Performance Optimization Features
- **Vector Quantization**: Binary/Scalar/Product quantization for 40x speedup potential
- **HNSW Tuning**: Optimized parameters for anime-specific search patterns
- **Payload Indexing**: Fast filtering on genre, year, type, status fields
- **Hybrid Search**: Single-request API for combined text+image queries
- **GPU Acceleration**: Support for GPU-accelerated model inference

## Environment Variables

### Critical Configuration
- `QDRANT_URL`: Vector database URL (default: http://localhost:6333)
- `QDRANT_COLLECTION_NAME`: Collection name (default: anime_database)
- `TEXT_EMBEDDING_MODEL`: Text model (default: BAAI/bge-m3)
- `IMAGE_EMBEDDING_MODEL`: Image model (default: jinaai/jina-clip-v2)

### Performance Tuning
- `QDRANT_ENABLE_QUANTIZATION`: Enable vector quantization (default: false)
- `QDRANT_QUANTIZATION_TYPE`: Quantization type (scalar, binary, product)
- `MODEL_WARM_UP`: Pre-load models during startup (default: false)
- `MAX_BATCH_SIZE`: Maximum batch size for operations (default: 500)

### Service Configuration
- `VECTOR_SERVICE_PORT`: Service port (default: 8002)
- `DEBUG`: Enable debug mode (default: true)
- `LOG_LEVEL`: Logging level (default: INFO)

## Memory Files System

This repository uses a comprehensive Memory Files system for project documentation. Always consult these files before making architectural changes or planning new features.

## Development Patterns

### Async-First Architecture
All I/O operations use async/await patterns. The service is designed for high concurrency with proper async context management.

### Configuration-Driven Features
Most functionality is configurable through environment variables rather than code changes. This enables zero-downtime configuration updates.

### Multi-Vector Design Philosophy
Named vectors in the same collection outperform separate collections for multi-modal data. This architecture provides 40% better search performance.

### Error Handling Strategy
The service implements graceful degradation - it continues operating with reduced functionality when dependencies fail rather than complete service failure.

## Integration Points

### External Dependencies
- **Qdrant Database**: Primary vector storage (required)
- **HuggingFace Models**: Text and image embeddings (cached locally)
- **External APIs**: Optional enrichment from anime platforms

### Client Integration
- Python client library available in `client/` directory
- REST API with comprehensive OpenAPI documentation at `/docs`
- Health check endpoint at `/health` for load balancer integration