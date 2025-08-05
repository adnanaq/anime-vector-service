# Vector Service Migration Guide

This guide explains how to extract the vector service from the main anime-mcp-server repository and deploy it as a separate microservice.

## 🏗️ **Complete Architecture**

The vector service has been designed as a standalone microservice with:

### **✅ Components Ready**
- **FastAPI Application** (`src/main.py`) - REST API server
- **Vector Processing** (`src/vector/`) - All vector components copied
- **API Endpoints** (`src/api/`) - Search, similarity, admin endpoints
- **Configuration** (`src/config/`) - Centralized settings management
- **Docker Setup** (`docker/`) - Containerized deployment
- **Client Library** (`client/`) - Easy integration for main app

### **🔧 Key Features**
- **Multi-Vector Search**: Text + picture + thumbnail embeddings
- **Advanced Models**: BGE-m3, JinaCLIP v2, configurable providers
- **Performance Optimized**: Quantization, HNSW, payload indexing
- **Fine-tuning Ready**: Character recognition, art style, genre enhancement
- **Production Ready**: Health checks, error handling, logging

## 📋 **Migration Steps**

### **Phase 1: Move to Separate Repository**

1. **Create new repository** `anime-vector-service`:
   ```bash
   cd /path/to/your/code/directory
   git clone https://github.com/your-org/anime-vector-service.git
   cd anime-vector-service
   ```

2. **Copy the vector service directory**:
   ```bash
   cp -r /path/to/anime-mcp-server/anime-vector-service/* .
   ```

3. **Initialize the repository**:
   ```bash
   git add .
   git commit -m "Initial vector service extraction from anime-mcp-server"
   git push origin main
   ```

### **Phase 2: Data Migration**

1. **Copy Qdrant data** (309MB):
   ```bash
   cp -r /path/to/anime-mcp-server/data/qdrant_storage ./data/
   ```

2. **Copy processed vectors** (1.9GB):
   ```bash
   cp -r /path/to/anime-mcp-server/data/processed ./data/
   ```

3. **Set up environment**:
   ```bash
   cp .env.example .env
   # Edit .env with your specific configuration
   ```

### **Phase 3: Deploy Vector Service**

1. **Using Docker** (recommended):
   ```bash
   cd docker
   docker-compose up -d
   ```

2. **Manual deployment**:
   ```bash
   pip install -r requirements.txt
   python -m src.main
   ```

3. **Verify deployment**:
   ```bash
   curl http://localhost:8002/health
   curl http://localhost:8002/api/v1/admin/stats
   ```

### **Phase 4: Update Main Application**

1. **Install vector service client**:
   ```bash
   # In main anime-mcp-server repo
   pip install requests aiohttp  # Add to requirements.txt
   ```

2. **Replace direct imports** in main application:

   **Before:**
   ```python
   from .vector.qdrant_client import QdrantClient
   ```

   **After:**
   ```python
   from anime_vector_service.client import VectorServiceClient
   ```

3. **Update configuration** in main app:
   ```python
   # In src/config.py
   vector_service_url: str = "http://localhost:8002"
   vector_service_timeout: int = 30
   vector_service_api_key: Optional[str] = None
   ```

4. **Update service initialization** in main app:

   **Before:**
   ```python
   qdrant_client = QdrantClient(url=settings.qdrant_url, ...)
   ```

   **After:**
   ```python
   vector_client = VectorServiceClient(
       base_url=settings.vector_service_url,
       timeout=settings.vector_service_timeout
   )
   ```

## 🔄 **API Migration Mapping**

### **Search Operations**
```python
# Before (direct Qdrant)
results = await qdrant_client.search(query, limit, filters)

# After (vector service)
results = await vector_client.search(query, limit, filters)
```

### **Image Search**
```python
# Before
results = await qdrant_client.search_by_image(image_data, limit)

# After
results = await vector_client.search_by_image(image_data, limit)
```

### **Similarity Search**
```python
# Before
results = await qdrant_client.find_similar(anime_id, limit)

# After
results = await vector_client.find_similar(anime_id, limit)
```

## 📁 **Files to Update in Main Application**

Based on the analysis, these **12 files** need updates:

1. `src/main.py` - Replace global qdrant_client
2. `src/api/search.py` - Use vector service client
3. `src/api/admin.py` - Update admin operations
4. `src/services/update_service.py` - Batch operations via API
5. `src/langgraph/agents/search_agent.py` - Update search agent
6. `src/anime_mcp/handlers/anime_handler.py` - MCP handler updates
7. `src/anime_mcp/handlers/base_handler.py` - Base handler updates
8. `src/anime_mcp/tools/tier4_comprehensive_tools.py` - MCP tools
9. `src/anime_mcp/tools/semantic_tools.py` - Semantic tools
10. `src/integrations/modern_service_manager.py` - Service manager
11. `src/anime_mcp/modern_server.py` - Modern MCP server
12. `src/services/iterative_ai_enrichment.py` - AI enrichment service

## 🚀 **Deployment Options**

### **Option 1: Docker Compose (Recommended)**
```yaml
# In main application's docker-compose.yml
services:
  vector-service:
    image: anime-vector-service:latest
    ports:
      - "8002:8002"
    environment:
      - QDRANT_URL=http://qdrant:6333
    depends_on:
      - qdrant
```

### **Option 2: Kubernetes**
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: anime-vector-service
spec:
  replicas: 2
  selector:
    matchLabels:
      app: anime-vector-service
  template:
    spec:
      containers:
      - name: vector-service
        image: anime-vector-service:latest
        ports:
        - containerPort: 8002
```

### **Option 3: Standalone Service**
```bash
# Start Qdrant
docker run -p 6333:6333 qdrant/qdrant:latest

# Start vector service
cd anime-vector-service
python -m src.main
```

## 🎯 **Benefits After Migration**

1. **Scalability**: Independent scaling of vector operations
2. **Maintainability**: Clear separation of concerns
3. **Performance**: Dedicated resources for embedding computations
4. **Reusability**: Multiple services can share the vector backend
5. **Development**: Faster development cycles for vector features
6. **Testing**: Isolated testing of vector functionality

## ⚠️ **Important Considerations**

1. **Network Latency**: API calls introduce network overhead
2. **Data Consistency**: Ensure proper synchronization between services
3. **Error Handling**: Implement robust retry and fallback mechanisms
4. **Authentication**: Consider API keys for production deployment
5. **Monitoring**: Set up health checks and metrics collection

## 🧪 **Testing the Separation**

1. **Start vector service**:
   ```bash
   cd anime-vector-service
   docker-compose up -d
   ```

2. **Test API endpoints**:
   ```bash
   # Health check
   curl http://localhost:8002/health
   
   # Search test
   curl -X POST http://localhost:8002/api/v1/search \
     -H "Content-Type: application/json" \
     -d '{"query": "dragon ball", "limit": 5}'
   
   # Stats check
   curl http://localhost:8002/api/v1/admin/stats
   ```

3. **Test client library**:
   ```python
   from anime_vector_service.client import VectorServiceClient
   
   async def test_client():
       async with VectorServiceClient() as client:
           results = await client.search("one piece", limit=5)
           print(f"Found {len(results)} results")
   ```

## 📊 **Expected Performance**

- **Startup Time**: ~30-60 seconds (model loading)
- **Search Latency**: <100ms for text search, <200ms for image search
- **Throughput**: ~100 searches/second with proper scaling
- **Memory Usage**: ~2-4GB (depending on models loaded)
- **Storage**: 2.3GB for data + model cache

## 🔧 **Troubleshooting**

**Common Issues:**
- **Import Errors**: Check that all vector dependencies are installed
- **Model Loading**: Ensure sufficient memory for embedding models
- **Qdrant Connection**: Verify Qdrant is running and accessible
- **Port Conflicts**: Ensure port 8002 is available

**Debug Commands:**
```bash
# Check service logs
docker-compose logs vector-service

# Check Qdrant logs  
docker-compose logs qdrant

# Test individual components
python -c "from src.vector.qdrant_client import QdrantClient; print('Import OK')"
```

This migration transforms your monolithic architecture into a scalable microservices design while preserving all vector functionality!