"""
Comprehensive tests for src/enrichment/api_helpers/animeschedule_fetcher.py

Tests cover:
- fetch_animeschedule_data function with various scenarios
- API request handling and error cases
- JSON parsing and response validation
- File saving functionality
- Command-line usage
"""

import json
from unittest.mock import AsyncMock, MagicMock, patch, mock_open

import aiohttp
import pytest

from src.enrichment.api_helpers.animeschedule_fetcher import (
    fetch_animeschedule_data,
)


class TestFetchAnimescheduleData:
    """Test fetch_animeschedule_data function."""

    @pytest.mark.asyncio
    async def test_fetch_success_without_save(self):
        """Test successful fetch without saving to file."""
        search_term = "dandadan"
        expected_data = {
            "anime": [
                {
                    "title": "DanDaDan",
                    "id": 12345,
                    "episodes": 12
                }
            ]
        }
        
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.json = AsyncMock(return_value=expected_data)
        
        mock_session = MagicMock()
        mock_session.get = MagicMock(return_value=AsyncMock(
            __aenter__=AsyncMock(return_value=mock_response),
            __aexit__=AsyncMock()
        ))
        mock_session.close = AsyncMock()
        
        with patch('src.enrichment.api_helpers.animeschedule_fetcher._cache_manager') as mock_cache:
            mock_cache.get_aiohttp_session.return_value = mock_session
            
            result = await fetch_animeschedule_data(search_term, save_file=False)
        
        assert result == expected_data["anime"][0]
        mock_session.close.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_fetch_success_with_save(self):
        """Test successful fetch with file saving."""
        search_term = "steins gate"
        expected_data = {
            "anime": [
                {
                    "title": "Steins;Gate",
                    "id": 67890,
                    "episodes": 24
                }
            ]
        }
        
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.json = AsyncMock(return_value=expected_data)
        
        mock_session = MagicMock()
        mock_session.get = MagicMock(return_value=AsyncMock(
            __aenter__=AsyncMock(return_value=mock_response),
            __aexit__=AsyncMock()
        ))
        mock_session.close = AsyncMock()
        
        m_open = mock_open()
        
        with patch('src.enrichment.api_helpers.animeschedule_fetcher._cache_manager') as mock_cache:
            mock_cache.get_aiohttp_session.return_value = mock_session
            
            with patch('builtins.open', m_open):
                result = await fetch_animeschedule_data(search_term, save_file=True)
        
        assert result == expected_data["anime"][0]
        m_open.assert_called_once_with("temp/as.json", "w", encoding="utf-8")
        mock_session.close.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_fetch_no_results_found(self):
        """Test when no results are returned from API."""
        search_term = "nonexistent anime"
        empty_response = {"anime": []}
        
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.json = AsyncMock(return_value=empty_response)
        
        mock_session = MagicMock()
        mock_session.get = MagicMock(return_value=AsyncMock(
            __aenter__=AsyncMock(return_value=mock_response),
            __aexit__=AsyncMock()
        ))
        mock_session.close = AsyncMock()
        
        with patch('src.enrichment.api_helpers.animeschedule_fetcher._cache_manager') as mock_cache:
            mock_cache.get_aiohttp_session.return_value = mock_session
            
            result = await fetch_animeschedule_data(search_term, save_file=False)
        
        assert result is None
        mock_session.close.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_fetch_no_anime_key(self):
        """Test when response doesn't contain 'anime' key."""
        search_term = "test anime"
        invalid_response = {"results": []}
        
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.json = AsyncMock(return_value=invalid_response)
        
        mock_session = MagicMock()
        mock_session.get = MagicMock(return_value=AsyncMock(
            __aenter__=AsyncMock(return_value=mock_response),
            __aexit__=AsyncMock()
        ))
        mock_session.close = AsyncMock()
        
        with patch('src.enrichment.api_helpers.animeschedule_fetcher._cache_manager') as mock_cache:
            mock_cache.get_aiohttp_session.return_value = mock_session
            
            result = await fetch_animeschedule_data(search_term, save_file=False)
        
        assert result is None
        mock_session.close.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_fetch_api_client_error(self):
        """Test handling of aiohttp ClientError."""
        search_term = "test anime"
        
        mock_session = MagicMock()
        mock_session.get = MagicMock(side_effect=aiohttp.ClientError("Connection failed"))
        mock_session.close = AsyncMock()
        
        with patch('src.enrichment.api_helpers.animeschedule_fetcher._cache_manager') as mock_cache:
            mock_cache.get_aiohttp_session.return_value = mock_session
            
            result = await fetch_animeschedule_data(search_term, save_file=False)
        
        assert result is None
        mock_session.close.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_fetch_json_decode_error(self):
        """Test handling of JSON decode errors."""
        search_term = "test anime"
        
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.json = AsyncMock(side_effect=json.JSONDecodeError("Invalid JSON", "", 0))
        
        mock_session = MagicMock()
        mock_session.get = MagicMock(return_value=AsyncMock(
            __aenter__=AsyncMock(return_value=mock_response),
            __aexit__=AsyncMock()
        ))
        mock_session.close = AsyncMock()
        
        with patch('src.enrichment.api_helpers.animeschedule_fetcher._cache_manager') as mock_cache:
            mock_cache.get_aiohttp_session.return_value = mock_session
            
            result = await fetch_animeschedule_data(search_term, save_file=False)
        
        assert result is None
        mock_session.close.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_fetch_http_error_status(self):
        """Test handling of HTTP error status codes."""
        search_term = "test anime"
        
        mock_response = AsyncMock()
        mock_response.status = 404
        mock_response.raise_for_status = MagicMock(side_effect=aiohttp.ClientResponseError(
            request_info=MagicMock(),
            history=(),
            status=404
        ))
        
        mock_session = MagicMock()
        mock_session.get = MagicMock(return_value=AsyncMock(
            __aenter__=AsyncMock(return_value=mock_response),
            __aexit__=AsyncMock()
        ))
        mock_session.close = AsyncMock()
        
        with patch('src.enrichment.api_helpers.animeschedule_fetcher._cache_manager') as mock_cache:
            mock_cache.get_aiohttp_session.return_value = mock_session
            
            result = await fetch_animeschedule_data(search_term, save_file=False)
        
        assert result is None
        mock_session.close.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_fetch_constructs_correct_url(self):
        """Test that correct URL is constructed for API call."""
        search_term = "cowboy bebop"
        expected_url = f"https://animeschedule.net/api/v3/anime?q={search_term}"
        
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.json = AsyncMock(return_value={"anime": [{"title": "Cowboy Bebop"}]})
        
        mock_session = MagicMock()
        mock_get = MagicMock(return_value=AsyncMock(
            __aenter__=AsyncMock(return_value=mock_response),
            __aexit__=AsyncMock()
        ))
        mock_session.get = mock_get
        mock_session.close = AsyncMock()
        
        with patch('src.enrichment.api_helpers.animeschedule_fetcher._cache_manager') as mock_cache:
            mock_cache.get_aiohttp_session.return_value = mock_session
            
            await fetch_animeschedule_data(search_term, save_file=False)
        
        # Verify URL was called correctly
        mock_get.assert_called_once_with(expected_url)

    @pytest.mark.asyncio
    async def test_fetch_returns_first_result(self):
        """Test that first result is returned when multiple results exist."""
        search_term = "naruto"
        multiple_results = {
            "anime": [
                {"title": "Naruto", "id": 1},
                {"title": "Naruto Shippuden", "id": 2},
                {"title": "Boruto", "id": 3}
            ]
        }
        
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.json = AsyncMock(return_value=multiple_results)
        
        mock_session = MagicMock()
        mock_session.get = MagicMock(return_value=AsyncMock(
            __aenter__=AsyncMock(return_value=mock_response),
            __aexit__=AsyncMock()
        ))
        mock_session.close = AsyncMock()
        
        with patch('src.enrichment.api_helpers.animeschedule_fetcher._cache_manager') as mock_cache:
            mock_cache.get_aiohttp_session.return_value = mock_session
            
            result = await fetch_animeschedule_data(search_term, save_file=False)
        
        assert result == multiple_results["anime"][0]
        assert result["id"] == 1

    @pytest.mark.asyncio
    async def test_fetch_session_always_closed(self):
        """Test that session is always closed even on errors."""
        search_term = "test anime"
        
        mock_session = MagicMock()
        mock_session.get = MagicMock(side_effect=Exception("Unexpected error"))
        mock_session.close = AsyncMock()
        
        with patch('src.enrichment.api_helpers.animeschedule_fetcher._cache_manager') as mock_cache:
            mock_cache.get_aiohttp_session.return_value = mock_session
            
            result = await fetch_animeschedule_data(search_term, save_file=False)
        
        # Session should be closed even after exception
        mock_session.close.assert_awaited_once()
        assert result is None

    @pytest.mark.asyncio
    async def test_fetch_uses_cache_manager(self):
        """Test that cache manager is used for session creation."""
        search_term = "test anime"
        
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.json = AsyncMock(return_value={"anime": [{"title": "Test"}]})
        
        mock_session = MagicMock()
        mock_session.get = MagicMock(return_value=AsyncMock(
            __aenter__=AsyncMock(return_value=mock_response),
            __aexit__=AsyncMock()
        ))
        mock_session.close = AsyncMock()
        
        with patch('src.enrichment.api_helpers.animeschedule_fetcher._cache_manager') as mock_cache:
            mock_cache.get_aiohttp_session.return_value = mock_session
            
            await fetch_animeschedule_data(search_term, save_file=False)
        
        mock_cache.get_aiohttp_session.assert_called_once_with("animeschedule")


class TestFetchAnimescheduleDataEdgeCases:
    """Test edge cases for fetch_animeschedule_data."""

    @pytest.mark.asyncio
    async def test_fetch_empty_search_term(self):
        """Test with empty search term."""
        search_term = ""
        expected_url = "https://animeschedule.net/api/v3/anime?q="
        
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.json = AsyncMock(return_value={"anime": []})
        
        mock_session = MagicMock()
        mock_get = MagicMock(return_value=AsyncMock(
            __aenter__=AsyncMock(return_value=mock_response),
            __aexit__=AsyncMock()
        ))
        mock_session.get = mock_get
        mock_session.close = AsyncMock()
        
        with patch('src.enrichment.api_helpers.animeschedule_fetcher._cache_manager') as mock_cache:
            mock_cache.get_aiohttp_session.return_value = mock_session
            
            result = await fetch_animeschedule_data(search_term, save_file=False)
        
        assert result is None
        mock_get.assert_called_once_with(expected_url)

    @pytest.mark.asyncio
    async def test_fetch_special_characters_in_search(self):
        """Test search term with special characters."""
        search_term = "Re:Zero"
        
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.json = AsyncMock(return_value={"anime": [{"title": "Re:Zero"}]})
        
        mock_session = MagicMock()
        mock_session.get = MagicMock(return_value=AsyncMock(
            __aenter__=AsyncMock(return_value=mock_response),
            __aexit__=AsyncMock()
        ))
        mock_session.close = AsyncMock()
        
        with patch('src.enrichment.api_helpers.animeschedule_fetcher._cache_manager') as mock_cache:
            mock_cache.get_aiohttp_session.return_value = mock_session
            
            result = await fetch_animeschedule_data(search_term, save_file=False)
        
        assert result is not None
        assert result["title"] == "Re:Zero"

    @pytest.mark.asyncio
    async def test_fetch_unicode_search_term(self):
        """Test search term with unicode characters."""
        search_term = "進撃の巨人"  # Attack on Titan in Japanese
        
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.json = AsyncMock(return_value={"anime": [{"title": search_term}]})
        
        mock_session = MagicMock()
        mock_session.get = MagicMock(return_value=AsyncMock(
            __aenter__=AsyncMock(return_value=mock_response),
            __aexit__=AsyncMock()
        ))
        mock_session.close = AsyncMock()
        
        with patch('src.enrichment.api_helpers.animeschedule_fetcher._cache_manager') as mock_cache:
            mock_cache.get_aiohttp_session.return_value = mock_session
            
            result = await fetch_animeschedule_data(search_term, save_file=False)
        
        assert result is not None

    @pytest.mark.asyncio
    async def test_fetch_null_response(self):
        """Test when API returns null."""
        search_term = "test"
        
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.json = AsyncMock(return_value=None)
        
        mock_session = MagicMock()
        mock_session.get = MagicMock(return_value=AsyncMock(
            __aenter__=AsyncMock(return_value=mock_response),
            __aexit__=AsyncMock()
        ))
        mock_session.close = AsyncMock()
        
        with patch('src.enrichment.api_helpers.animeschedule_fetcher._cache_manager') as mock_cache:
            mock_cache.get_aiohttp_session.return_value = mock_session
            
            result = await fetch_animeschedule_data(search_term, save_file=False)
        
        assert result is None