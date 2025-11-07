"""
Comprehensive tests for src/cache_manager/instance.py

Tests cover:
- Singleton instance creation and initialization
- Module-level configuration loading
- HTTPCacheManager instance attributes
- Configuration from environment variables
"""

from unittest.mock import MagicMock, patch

import pytest

from src.cache_manager.config import CacheConfig


class TestCacheManagerInstance:
    """Test cache manager singleton instance."""

    def test_module_imports_successfully(self):
        """Test that the instance module can be imported."""
        # This validates the module initialization doesn't raise errors
        from src.cache_manager import instance
        assert instance is not None

    def test_singleton_instance_created(self):
        """Test that singleton instance is created on module load."""
        from src.cache_manager.instance import http_cache_manager
        
        assert http_cache_manager is not None
        from src.cache_manager.manager import HTTPCacheManager
        assert isinstance(http_cache_manager, HTTPCacheManager)

    def test_singleton_instance_has_config(self):
        """Test that singleton instance has valid configuration."""
        from src.cache_manager.instance import http_cache_manager
        
        assert http_cache_manager.config is not None
        assert isinstance(http_cache_manager.config, CacheConfig)

    def test_same_instance_returned_on_multiple_imports(self):
        """Test that multiple imports return the same instance."""
        from src.cache_manager.instance import http_cache_manager as instance1
        from src.cache_manager.instance import http_cache_manager as instance2
        
        assert instance1 is instance2

    @patch('src.cache_manager.instance.get_cache_config')
    def test_instance_uses_get_cache_config(self, mock_get_config):
        """Test that instance creation uses get_cache_config."""
        mock_config = MagicMock(spec=CacheConfig)
        mock_get_config.return_value = mock_config
        
        # Reload module to trigger instance creation with mock
        import importlib
        from src.cache_manager import instance
        importlib.reload(instance)
        
        # Verify get_cache_config was called
        mock_get_config.assert_called()

    def test_instance_config_matches_get_cache_config(self):
        """Test that instance config matches get_cache_config result."""
        from src.cache_manager.config import get_cache_config
        from src.cache_manager.instance import http_cache_manager
        
        expected_config = get_cache_config()
        
        # Compare key attributes
        assert http_cache_manager.config.enabled == expected_config.enabled
        assert http_cache_manager.config.storage_type == expected_config.storage_type
        assert http_cache_manager.config.redis_url == expected_config.redis_url

    def test_instance_initialization_without_errors(self):
        """Test that instance initializes without errors."""
        # Should not raise any exceptions
        from src.cache_manager.instance import http_cache_manager
        
        # Basic validation that object is usable
        assert hasattr(http_cache_manager, 'config')
        assert hasattr(http_cache_manager, 'get_aiohttp_session')
        assert hasattr(http_cache_manager, 'close')

    def test_module_exports_correct_instance(self):
        """Test that module exports the correct instance name."""
        import src.cache_manager.instance as instance_module
        
        assert hasattr(instance_module, 'http_cache_manager')
        assert instance_module.http_cache_manager is not None

    def test_instance_config_respects_environment(self):
        """Test that instance respects environment configuration."""
        from src.cache_manager.instance import http_cache_manager
        
        # Instance should have been created with environment-based config
        config = http_cache_manager.config
        
        # These should match the environment or defaults
        assert isinstance(config.enabled, bool)
        assert config.storage_type in ['sqlite', 'redis']
        assert isinstance(config.redis_url, str)

    @patch.dict('os.environ', {'CACHE_ENABLED': 'false'})
    def test_instance_respects_disabled_cache_env(self):
        """Test that instance respects CACHE_ENABLED=false."""
        # Reload to pick up new environment
        import importlib
        from src.cache_manager import config as config_module
        importlib.reload(config_module)
        
        from src.cache_manager.config import get_cache_config
        test_config = get_cache_config()
        
        assert test_config.enabled is False

    def test_instance_storage_initialized_based_on_config(self):
        """Test that storage is initialized according to configuration."""
        from src.cache_manager.instance import http_cache_manager
        
        if http_cache_manager.config.enabled:
            # Storage should be initialized
            assert http_cache_manager._storage is not None or \
                   http_cache_manager._async_redis_client is not None
        else:
            # Storage should be None when disabled
            assert http_cache_manager._storage is None
            assert http_cache_manager._async_redis_client is None


class TestInstanceDocumentation:
    """Test documentation and module-level attributes."""

    def test_module_has_docstring(self):
        """Test that instance module has proper documentation."""
        import src.cache_manager.instance as instance_module
        
        assert instance_module.__doc__ is not None
        assert 'singleton' in instance_module.__doc__.lower()

    def test_instance_purpose_documented(self):
        """Test that singleton purpose is documented."""
        import src.cache_manager.instance as instance_module
        
        doc = instance_module.__doc__
        assert 'shared' in doc.lower() or 'singleton' in doc.lower()
        assert 'cache' in doc.lower()


class TestInstanceIntegration:
    """Test instance integration with other modules."""

    def test_instance_can_be_imported_from_other_modules(self):
        """Test that instance can be imported by other modules."""
        # This simulates how other parts of the codebase use it
        from src.cache_manager.instance import http_cache_manager as cache_mgr
        
        assert cache_mgr is not None
        assert hasattr(cache_mgr, 'get_aiohttp_session')

    def test_instance_compatible_with_manager_interface(self):
        """Test that instance implements HTTPCacheManager interface."""
        from src.cache_manager.instance import http_cache_manager
        
        # Should have all public methods of HTTPCacheManager
        assert callable(getattr(http_cache_manager, 'get_aiohttp_session', None))
        assert callable(getattr(http_cache_manager, 'close', None))
        assert callable(getattr(http_cache_manager, 'close_async', None))

    def test_instance_config_is_cache_config_instance(self):
        """Test that instance.config is proper CacheConfig."""
        from src.cache_manager.config import CacheConfig
        from src.cache_manager.instance import http_cache_manager
        
        assert isinstance(http_cache_manager.config, CacheConfig)

    @pytest.mark.asyncio
    async def test_instance_session_creation_works(self):
        """Test that instance can create sessions."""
        from src.cache_manager.instance import http_cache_manager
        
        # Should not raise errors
        session = http_cache_manager.get_aiohttp_session("test_service")
        assert session is not None
        
        # Cleanup
        await session.close()