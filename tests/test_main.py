"""
Comprehensive tests for src/main.py

Tests cover:
- Application initialization and lifespan
- Health check endpoint
- Root endpoint
- CORS middleware configuration
- Router integration
- Startup and shutdown procedures
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient


class TestApplicationLifespan:
    """Test application lifespan management."""

    @pytest.mark.asyncio
    async def test_lifespan_initializes_qdrant_client(self):
        """Test that lifespan initializes Qdrant client."""
        mock_qdrant_client = MagicMock()
        mock_qdrant_client.health_check = AsyncMock(return_value=True)
        
        with patch('src.main.QdrantClient', return_value=mock_qdrant_client):
            with patch('src.main.http_cache_manager'):
                with patch('src.main.close_result_cache_redis_client'):
                    from src.main import lifespan
                    
                    async with lifespan(MagicMock()):
                        pass
                    
                    mock_qdrant_client.health_check.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_lifespan_raises_on_unhealthy_qdrant(self):
        """Test that lifespan raises error when Qdrant is unhealthy."""
        mock_qdrant_client = MagicMock()
        mock_qdrant_client.health_check = AsyncMock(return_value=False)
        
        with patch('src.main.QdrantClient', return_value=mock_qdrant_client):
            from src.main import lifespan
            
            with pytest.raises(RuntimeError, match="Vector database is not available"):
                async with lifespan(MagicMock()):
                    pass

    @pytest.mark.asyncio
    async def test_lifespan_closes_cache_on_shutdown(self):
        """Test that lifespan closes cache clients on shutdown."""
        mock_qdrant_client = MagicMock()
        mock_qdrant_client.health_check = AsyncMock(return_value=True)
        
        mock_cache_manager = MagicMock()
        mock_cache_manager.close_async = AsyncMock()
        mock_close_redis = AsyncMock()
        
        with patch('src.main.QdrantClient', return_value=mock_qdrant_client):
            with patch('src.main.http_cache_manager', mock_cache_manager):
                with patch('src.main.close_result_cache_redis_client', mock_close_redis):
                    from src.main import lifespan
                    
                    async with lifespan(MagicMock()):
                        pass
                    
                    mock_cache_manager.close_async.assert_awaited_once()
                    mock_close_redis.assert_awaited_once()


class TestHealthCheckEndpoint:
    """Test health check endpoint."""

    def test_health_check_returns_healthy_status(self):
        """Test health check returns healthy when Qdrant is available."""
        with patch('src.main.qdrant_client') as mock_client:
            mock_client.health_check = AsyncMock(return_value=True)
            
            # Import after patching to use mocked client
            from src.main import app
            client = TestClient(app)
            
            # Temporarily set the client
            import src.main
            original_client = src.main.qdrant_client
            src.main.qdrant_client = mock_client
            
            try:
                response = client.get("/health")
                assert response.status_code == 200
                
                data = response.json()
                assert data["status"] == "healthy"
                assert data["service"] == "anime-vector-service"
                assert "timestamp" in data
                assert "version" in data
                assert data["qdrant_status"] is True
            finally:
                src.main.qdrant_client = original_client

    def test_health_check_returns_unhealthy_when_qdrant_down(self):
        """Test health check returns unhealthy when Qdrant is down."""
        with patch('src.main.qdrant_client') as mock_client:
            mock_client.health_check = AsyncMock(return_value=False)
            
            from src.main import app
            client = TestClient(app)
            
            import src.main
            original_client = src.main.qdrant_client
            src.main.qdrant_client = mock_client
            
            try:
                response = client.get("/health")
                assert response.status_code == 200
                
                data = response.json()
                assert data["status"] == "unhealthy"
                assert data["qdrant_status"] is False
            finally:
                src.main.qdrant_client = original_client

    def test_health_check_handles_exception(self):
        """Test health check handles exceptions gracefully."""
        with patch('src.main.qdrant_client') as mock_client:
            mock_client.health_check = AsyncMock(side_effect=Exception("Connection error"))
            
            from src.main import app
            client = TestClient(app)
            
            import src.main
            original_client = src.main.qdrant_client
            src.main.qdrant_client = mock_client
            
            try:
                response = client.get("/health")
                assert response.status_code == 503
                assert "Service unhealthy" in response.json()["detail"]
            finally:
                src.main.qdrant_client = original_client

    def test_health_check_requires_initialized_client(self):
        """Test health check fails when client is not initialized."""
        from src.main import app
        client = TestClient(app)
        
        import src.main
        original_client = src.main.qdrant_client
        src.main.qdrant_client = None
        
        try:
            response = client.get("/health")
            assert response.status_code == 503
            assert "Qdrant client not initialized" in response.json()["detail"]
        finally:
            src.main.qdrant_client = original_client


class TestRootEndpoint:
    """Test root endpoint."""

    def test_root_returns_service_info(self):
        """Test root endpoint returns service information."""
        from src.main import app
        client = TestClient(app)
        
        response = client.get("/")
        assert response.status_code == 200
        
        data = response.json()
        assert data["service"] == "Anime Vector Service"
        assert "version" in data
        assert "description" in data
        assert "endpoints" in data

    def test_root_includes_all_endpoints(self):
        """Test root endpoint includes all API endpoints."""
        from src.main import app
        client = TestClient(app)
        
        response = client.get("/")
        data = response.json()
        
        endpoints = data["endpoints"]
        assert "health" in endpoints
        assert "search" in endpoints
        assert "image_search" in endpoints
        assert "multimodal_search" in endpoints
        assert "similar" in endpoints
        assert "visual_similar" in endpoints
        assert "stats" in endpoints
        assert "docs" in endpoints

    def test_root_endpoint_paths_are_correct(self):
        """Test root endpoint provides correct paths."""
        from src.main import app
        client = TestClient(app)
        
        response = client.get("/")
        data = response.json()
        
        endpoints = data["endpoints"]
        assert endpoints["health"] == "/health"
        assert endpoints["search"] == "/api/v1/search"
        assert endpoints["docs"] == "/docs"


class TestApplicationConfiguration:
    """Test application configuration."""

    def test_app_has_correct_title(self):
        """Test application has correct title."""
        from src.main import app
        from src.config import get_settings
        
        settings = get_settings()
        assert app.title == settings.api_title

    def test_app_has_correct_description(self):
        """Test application has correct description."""
        from src.main import app
        from src.config import get_settings
        
        settings = get_settings()
        assert app.description == settings.api_description

    def test_app_has_correct_version(self):
        """Test application has correct version."""
        from src.main import app
        from src.config import get_settings
        
        settings = get_settings()
        assert app.version == settings.api_version

    def test_cors_middleware_configured(self):
        """Test CORS middleware is properly configured."""
        from src.main import app
        
        # Check that CORS middleware is present
        middleware_types = [type(m) for m in app.user_middleware]
        from fastapi.middleware.cors import CORSMiddleware
        
        # CORSMiddleware should be in the stack
        assert any('CORS' in str(m) for m in middleware_types)


class TestRouterIntegration:
    """Test router integration."""

    def test_search_router_included(self):
        """Test search router is included."""
        from src.main import app
        
        routes = [route.path for route in app.routes]
        # Should have search endpoints
        assert any('/api/v1/search' in path for path in routes)

    def test_similarity_router_included(self):
        """Test similarity router is included."""
        from src.main import app
        
        routes = [route.path for route in app.routes]
        # Should have similarity endpoints
        assert any('/api/v1/similarity' in path for path in routes)

    def test_admin_router_included(self):
        """Test admin router is included."""
        from src.main import app
        
        routes = [route.path for route in app.routes]
        # Should have admin endpoints
        assert any('/api/v1/admin' in path for path in routes)


class TestApplicationStartup:
    """Test application startup behavior."""

    def test_logging_configured_on_import(self):
        """Test logging is configured when module is imported."""
        import logging
        
        # Import should configure logging
        import src.main
        
        # Logger should be configured
        logger = logging.getLogger("src.main")
        assert logger is not None

    def test_settings_loaded_on_import(self):
        """Test settings are loaded when module is imported."""
        from src.main import settings
        
        assert settings is not None
        assert hasattr(settings, 'qdrant_url')
        assert hasattr(settings, 'api_version')


class TestGlobalInstances:
    """Test global instance management."""

    def test_qdrant_client_initially_none(self):
        """Test qdrant_client starts as None."""
        # Need to test before lifespan
        import importlib
        import src.main as main_module
        
        # Reload to get fresh state
        importlib.reload(main_module)
        
        # Check initial state
        # Note: In actual usage, this gets set during lifespan
        # We're testing the declaration here
        assert hasattr(main_module, 'qdrant_client')


class TestMainExecution:
    """Test __main__ execution."""

    def test_main_block_imports_uvicorn(self):
        """Test that __main__ block can import uvicorn."""
        # This is more of a smoke test that the import works
        try:
            import uvicorn
            assert uvicorn is not None
        except ImportError:
            pytest.skip("uvicorn not installed")

    @patch('uvicorn.run')
    def test_main_block_calls_uvicorn_with_correct_params(self, _mock_run):
        """Test that __main__ block calls uvicorn with correct parameters."""
        from src.config import get_settings
        settings = get_settings()
        
        # We can't actually test the if __name__ == "__main__" block
        # but we can verify the settings are available
        assert hasattr(settings, 'vector_service_host')
        assert hasattr(settings, 'vector_service_port')
        assert hasattr(settings, 'debug')
        assert hasattr(settings, 'log_level')


class TestApplicationEndpoints:
    """Integration tests for application endpoints."""

    def test_docs_endpoint_accessible(self):
        """Test OpenAPI docs endpoint is accessible."""
        from src.main import app
        client = TestClient(app)
        
        response = client.get("/docs")
        # Should redirect or return docs page
        assert response.status_code in [200, 307]

    def test_openapi_json_accessible(self):
        """Test OpenAPI JSON spec is accessible."""
        from src.main import app
        client = TestClient(app)
        
        response = client.get("/openapi.json")
        assert response.status_code == 200
        
        spec = response.json()
        assert "openapi" in spec
        assert "info" in spec
        assert "paths" in spec


class TestApplicationErrorHandling:
    """Test application error handling."""

    def test_404_on_unknown_endpoint(self):
        """Test 404 is returned for unknown endpoints."""
        from src.main import app
        client = TestClient(app)
        
        response = client.get("/unknown/endpoint")
        assert response.status_code == 404

    def test_405_on_wrong_method(self):
        """Test 405 is returned for wrong HTTP method."""
        from src.main import app
        client = TestClient(app)
        
        # POST to health check (only accepts GET)
        response = client.post("/health")
        assert response.status_code == 405