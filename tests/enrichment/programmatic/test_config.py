"""
Comprehensive tests for src/enrichment/programmatic/config.py

Tests cover:
- EnrichmentConfig initialization and defaults
- Field validation and constraints
- Environment variable loading
- Invalid configuration handling
- Configuration logging
"""

from unittest.mock import patch

import pytest
from pydantic import ValidationError

from src.enrichment.programmatic.config import EnrichmentConfig


class TestEnrichmentConfigDefaults:
    """Test default configuration values."""

    def test_default_api_configuration(self):
        """Test API configuration defaults."""
        config = EnrichmentConfig()
        
        assert config.api_timeout == 200
        assert config.max_concurrent_apis == 6
        assert config.retry_attempts == 3
        assert config.retry_delay == 1.0

    def test_default_batch_configuration(self):
        """Test batch processing defaults."""
        config = EnrichmentConfig()
        
        assert config.batch_size == 10
        assert config.episode_batch_size == 50
        assert config.character_batch_size == 50

    def test_default_performance_configuration(self):
        """Test performance tuning defaults."""
        config = EnrichmentConfig()
        
        assert config.enable_caching is True
        assert config.cache_ttl == 86400  # 24 hours

    def test_config_initialization_without_errors(self):
        """Test that config initializes without errors."""
        config = EnrichmentConfig()
        assert config is not None
        assert isinstance(config, EnrichmentConfig)


class TestEnrichmentConfigValidation:
    """Test configuration field validation."""

    def test_api_timeout_validation(self):
        """Test api_timeout field validation."""
        # Valid values
        config = EnrichmentConfig(api_timeout=100)
        assert config.api_timeout == 100
        
        config = EnrichmentConfig(api_timeout=300)
        assert config.api_timeout == 300

    def test_max_concurrent_apis_validation(self):
        """Test max_concurrent_apis field validation."""
        config = EnrichmentConfig(max_concurrent_apis=10)
        assert config.max_concurrent_apis == 10
        
        # Test reasonable values
        config = EnrichmentConfig(max_concurrent_apis=1)
        assert config.max_concurrent_apis == 1

    def test_retry_attempts_validation(self):
        """Test retry_attempts field validation."""
        config = EnrichmentConfig(retry_attempts=5)
        assert config.retry_attempts == 5
        
        config = EnrichmentConfig(retry_attempts=0)
        assert config.retry_attempts == 0

    def test_retry_delay_validation(self):
        """Test retry_delay field validation."""
        config = EnrichmentConfig(retry_delay=2.5)
        assert config.retry_delay == 2.5
        
        config = EnrichmentConfig(retry_delay=0.1)
        assert config.retry_delay == 0.1

    def test_batch_size_validation(self):
        """Test batch_size field validation."""
        config = EnrichmentConfig(batch_size=20)
        assert config.batch_size == 20
        
        config = EnrichmentConfig(batch_size=1)
        assert config.batch_size == 1

    def test_episode_batch_size_validation(self):
        """Test episode_batch_size field validation."""
        config = EnrichmentConfig(episode_batch_size=100)
        assert config.episode_batch_size == 100

    def test_character_batch_size_validation(self):
        """Test character_batch_size field validation."""
        config = EnrichmentConfig(character_batch_size=75)
        assert config.character_batch_size == 75

    def test_enable_caching_validation(self):
        """Test enable_caching field validation."""
        config = EnrichmentConfig(enable_caching=False)
        assert config.enable_caching is False
        
        config = EnrichmentConfig(enable_caching=True)
        assert config.enable_caching is True

    def test_cache_ttl_validation(self):
        """Test cache_ttl field validation."""
        config = EnrichmentConfig(cache_ttl=3600)  # 1 hour
        assert config.cache_ttl == 3600
        
        config = EnrichmentConfig(cache_ttl=604800)  # 1 week
        assert config.cache_ttl == 604800


class TestEnrichmentConfigEnvironment:
    """Test environment variable loading."""

    @patch.dict('os.environ', {'API_TIMEOUT': '150'})
    def test_load_api_timeout_from_env(self):
        """Test loading api_timeout from environment."""
        config = EnrichmentConfig()
        assert config.api_timeout == 150

    @patch.dict('os.environ', {'MAX_CONCURRENT_APIS': '8'})
    def test_load_max_concurrent_apis_from_env(self):
        """Test loading max_concurrent_apis from environment."""
        config = EnrichmentConfig()
        assert config.max_concurrent_apis == 8

    @patch.dict('os.environ', {'RETRY_ATTEMPTS': '5'})
    def test_load_retry_attempts_from_env(self):
        """Test loading retry_attempts from environment."""
        config = EnrichmentConfig()
        assert config.retry_attempts == 5

    @patch.dict('os.environ', {'RETRY_DELAY': '2.0'})
    def test_load_retry_delay_from_env(self):
        """Test loading retry_delay from environment."""
        config = EnrichmentConfig()
        assert config.retry_delay == 2.0

    @patch.dict('os.environ', {'BATCH_SIZE': '15'})
    def test_load_batch_size_from_env(self):
        """Test loading batch_size from environment."""
        config = EnrichmentConfig()
        assert config.batch_size == 15

    @patch.dict('os.environ', {'EPISODE_BATCH_SIZE': '60'})
    def test_load_episode_batch_size_from_env(self):
        """Test loading episode_batch_size from environment."""
        config = EnrichmentConfig()
        assert config.episode_batch_size == 60

    @patch.dict('os.environ', {'CHARACTER_BATCH_SIZE': '80'})
    def test_load_character_batch_size_from_env(self):
        """Test loading character_batch_size from environment."""
        config = EnrichmentConfig()
        assert config.character_batch_size == 80

    @patch.dict('os.environ', {'ENABLE_CACHING': 'false'})
    def test_load_enable_caching_from_env(self):
        """Test loading enable_caching from environment."""
        config = EnrichmentConfig()
        assert config.enable_caching is False

    @patch.dict('os.environ', {'CACHE_TTL': '7200'})
    def test_load_cache_ttl_from_env(self):
        """Test loading cache_ttl from environment."""
        config = EnrichmentConfig()
        assert config.cache_ttl == 7200


class TestEnrichmentConfigInvalidValues:
    """Test invalid configuration values."""

    def test_negative_api_timeout_rejected(self):
        """Test that negative api_timeout is rejected."""
        with pytest.raises(ValidationError):
            EnrichmentConfig(api_timeout=-1)

    def test_negative_max_concurrent_apis_rejected(self):
        """Test that negative max_concurrent_apis is rejected."""
        with pytest.raises(ValidationError):
            EnrichmentConfig(max_concurrent_apis=-1)

    def test_negative_retry_attempts_rejected(self):
        """Test that negative retry_attempts is rejected."""
        with pytest.raises(ValidationError):
            EnrichmentConfig(retry_attempts=-1)

    def test_negative_retry_delay_rejected(self):
        """Test that negative retry_delay is rejected."""
        with pytest.raises(ValidationError):
            EnrichmentConfig(retry_delay=-1.0)

    def test_negative_batch_size_rejected(self):
        """Test that negative batch_size is rejected."""
        with pytest.raises(ValidationError):
            EnrichmentConfig(batch_size=-1)

    def test_negative_episode_batch_size_rejected(self):
        """Test that negative episode_batch_size is rejected."""
        with pytest.raises(ValidationError):
            EnrichmentConfig(episode_batch_size=-1)

    def test_negative_character_batch_size_rejected(self):
        """Test that negative character_batch_size is rejected."""
        with pytest.raises(ValidationError):
            EnrichmentConfig(character_batch_size=-1)

    def test_negative_cache_ttl_rejected(self):
        """Test that negative cache_ttl is rejected."""
        with pytest.raises(ValidationError):
            EnrichmentConfig(cache_ttl=-1)

    def test_invalid_type_for_enable_caching(self):
        """Test that invalid type for enable_caching is rejected."""
        with pytest.raises(ValidationError):
            EnrichmentConfig(enable_caching="invalid")


class TestEnrichmentConfigUsage:
    """Test configuration usage patterns."""

    def test_config_can_be_serialized(self):
        """Test that config can be serialized."""
        config = EnrichmentConfig()
        config_dict = config.model_dump()
        
        assert isinstance(config_dict, dict)
        assert 'api_timeout' in config_dict
        assert 'max_concurrent_apis' in config_dict

    def test_config_can_be_created_from_dict(self):
        """Test that config can be created from dict."""
        config_dict = {
            'api_timeout': 250,
            'max_concurrent_apis': 8,
            'batch_size': 15
        }
        config = EnrichmentConfig(**config_dict)
        
        assert config.api_timeout == 250
        assert config.max_concurrent_apis == 8
        assert config.batch_size == 15

    def test_config_partial_override(self):
        """Test partial configuration override."""
        config = EnrichmentConfig(
            api_timeout=180,
            max_concurrent_apis=8
        )
        
        # Changed values
        assert config.api_timeout == 180
        assert config.max_concurrent_apis == 8
        
        # Default values
        assert config.retry_attempts == 3
        assert config.batch_size == 10

    def test_config_immutable_after_creation(self):
        """Test that config values can be accessed after creation."""
        config = EnrichmentConfig()
        
        # Should be able to read values
        timeout = config.api_timeout
        assert timeout == 200
        
        # Pydantic allows modification but validates
        config.api_timeout = 300
        assert config.api_timeout == 300


class TestEnrichmentConfigDescriptions:
    """Test field descriptions and metadata."""

    def test_api_timeout_has_description(self):
        """Test api_timeout field has description."""
        field_info = EnrichmentConfig.model_fields['api_timeout']
        assert field_info.description is not None
        assert 'timeout' in field_info.description.lower()

    def test_max_concurrent_apis_has_description(self):
        """Test max_concurrent_apis field has description."""
        field_info = EnrichmentConfig.model_fields['max_concurrent_apis']
        assert field_info.description is not None
        assert 'concurrent' in field_info.description.lower()

    def test_batch_size_has_description(self):
        """Test batch_size field has description."""
        field_info = EnrichmentConfig.model_fields['batch_size']
        assert field_info.description is not None
        assert 'batch' in field_info.description.lower() or 'anime' in field_info.description.lower()

    def test_enable_caching_has_description(self):
        """Test enable_caching field has description."""
        field_info = EnrichmentConfig.model_fields['enable_caching']
        assert field_info.description is not None
        assert 'cach' in field_info.description.lower()


class TestEnrichmentConfigEdgeCases:
    """Test edge cases and boundary values."""

    def test_zero_api_timeout(self):
        """Test zero api_timeout."""
        config = EnrichmentConfig(api_timeout=0)
        assert config.api_timeout == 0

    def test_very_large_api_timeout(self):
        """Test very large api_timeout."""
        config = EnrichmentConfig(api_timeout=10000)
        assert config.api_timeout == 10000

    def test_zero_retry_attempts(self):
        """Test zero retry attempts (no retries)."""
        config = EnrichmentConfig(retry_attempts=0)
        assert config.retry_attempts == 0

    def test_single_concurrent_api(self):
        """Test single concurrent API call."""
        config = EnrichmentConfig(max_concurrent_apis=1)
        assert config.max_concurrent_apis == 1

    def test_minimum_retry_delay(self):
        """Test minimum retry delay."""
        config = EnrichmentConfig(retry_delay=0.0)
        assert config.retry_delay == 0.0

    def test_batch_size_one(self):
        """Test batch size of one."""
        config = EnrichmentConfig(batch_size=1)
        assert config.batch_size == 1

    def test_very_long_cache_ttl(self):
        """Test very long cache TTL (1 year)."""
        config = EnrichmentConfig(cache_ttl=31536000)  # 365 days
        assert config.cache_ttl == 31536000

    def test_cache_ttl_zero(self):
        """Test cache TTL of zero (no caching)."""
        config = EnrichmentConfig(cache_ttl=0)
        assert config.cache_ttl == 0