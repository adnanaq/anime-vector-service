# Anime Vector Service

A microservice for semantic search over anime content using vector embeddings and Qdrant database.

## Features

- **Semantic Search**: Text-based search using BGE-M3 embeddings
- **Visual Search**: Image-to-image similarity with JinaCLIP v2
- **Multimodal Search**: Combined text and image queries
- **Similarity Analysis**: Find visually or contextually similar anime
- **Batch Operations**: Efficient bulk vector processing
- **Production Ready**: Health checks, monitoring, CORS support

## Quick Start

### Using Docker (Recommended)

```bash
# Start services
docker compose up -d

# Service available at http://localhost:8002
curl http://localhost:8002/health
```

### Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Start Qdrant database
docker compose up -d qdrant

# Run service
python -m src.main
```

## API Endpoints

### Search Operations
- `POST /api/v1/search` - Semantic text search
- `POST /api/v1/search/image` - Image-based search
- `POST /api/v1/search/multimodal` - Combined text+image search

### Similarity Operations  
- `POST /api/v1/similarity/anime/{anime_id}` - Find similar anime
- `POST /api/v1/similarity/visual/{anime_id}` - Find visually similar anime

### Administration
- `GET /health` - Service health status
- `GET /api/v1/admin/stats` - Database statistics
- `POST /api/v1/admin/reindex` - Rebuild search index

## Configuration

Environment variables:

```env
# Service
VECTOR_SERVICE_HOST=0.0.0.0
VECTOR_SERVICE_PORT=8002

# Database
QDRANT_URL=http://localhost:6333
QDRANT_COLLECTION_NAME=anime_database

# Embedding Models
TEXT_EMBEDDING_MODEL=BAAI/bge-m3
IMAGE_EMBEDDING_MODEL=jinaai/jina-clip-v2
```

## Technology Stack

- **FastAPI**: REST API framework
- **Qdrant**: Vector database with HNSW indexing
- **BGE-M3**: Multi-lingual text embeddings
- **JinaCLIP v2**: Vision-language model for images
- **Docker**: Containerized deployment