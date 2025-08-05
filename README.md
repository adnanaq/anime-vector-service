# Anime Vector Service

A microservice for handling vector database operations, extracted from the main Anime MCP Server for better scalability and maintainability.

## Overview

This service provides:
- Semantic search capabilities using Qdrant vector database
- Multi-modal search (text + image) with CLIP embeddings
- FastEmbed text embeddings (BGE, Sentence Transformers)
- Batch vector operations for efficient data processing
- RESTful API for easy integration

## Architecture

```
anime-vector-service/
├── src/
│   ├── api/           # REST API endpoints
│   ├── vector/        # Vector processing logic (moved from main repo)
│   ├── models/        # Pydantic data models
│   ├── config/        # Configuration management
│   └── main.py        # FastAPI application entry point
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml
└── requirements.txt
```

## Key Features

- **Multi-Vector Support**: Text, picture, and thumbnail embeddings in single collection
- **Advanced Search**: Semantic, similarity, image-based, and multimodal search
- **Performance Optimized**: Quantization, HNSW indexing, batch processing
- **Production Ready**: Health checks, statistics, error handling

## API Endpoints

### Core Vector Operations
- `POST /api/v1/vectors/search` - Semantic text search
- `POST /api/v1/vectors/similar` - Find similar anime by ID
- `POST /api/v1/vectors/image-search` - Image-based visual search
- `POST /api/v1/vectors/multimodal` - Combined text+image search

### Data Management
- `POST /api/v1/vectors/upsert` - Add/update vectors
- `POST /api/v1/vectors/batch-upsert` - Batch operations
- `DELETE /api/v1/vectors/{id}` - Remove vectors

### Administration
- `GET /api/v1/health` - Service health check
- `GET /api/v1/stats` - Database statistics
- `POST /api/v1/admin/reindex` - Rebuild vector index

## Configuration

Key environment variables:
```env
# Vector Service
VECTOR_SERVICE_HOST=0.0.0.0
VECTOR_SERVICE_PORT=8002

# Qdrant Configuration
QDRANT_URL=http://localhost:6333
QDRANT_COLLECTION_NAME=anime_database

# Embedding Models
TEXT_EMBEDDING_MODEL=BAAI/bge-m3
IMAGE_EMBEDDING_MODEL=jinaai/jina-clip-v2
```

## Development

```bash
# Install dependencies
pip install -r requirements.txt

# Start Qdrant
docker compose up -d qdrant

# Run service
python -m src.main
```

## Migration Status

This service was extracted from the main anime-mcp-server repository to improve:
- **Scalability**: Independent scaling of vector operations
- **Maintainability**: Clear separation of concerns
- **Performance**: Dedicated resources for embedding computations
- **Reusability**: Multiple services can share the same vector backend

## Data Migration

When moving to production:
1. Export vectors from main service
2. Import to this service's Qdrant instance
3. Update main service to use vector service API
4. Validate search functionality

## Dependencies

- FastAPI: REST API framework
- Qdrant: Vector database
- FastEmbed: Text embeddings
- CLIP/JinaCLIP: Image embeddings
- Sentence Transformers: Alternative text embeddings
- PyTorch: Deep learning framework