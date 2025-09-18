"""
Parallel API fetcher for anime enrichment.
Fetches data from all APIs concurrently using existing helpers.
Reduces API fetching from 30-60s sequential to 5-10s parallel.
"""

import asyncio
import aiohttp
import time
import json
import os
import sys
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import logging

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from src.enrichment.api_helpers.anilist_helper import AniListEnrichmentHelper
from src.enrichment.api_helpers.kitsu_helper import KitsuEnrichmentHelper
from src.enrichment.api_helpers.anidb_helper import AniDBEnrichmentHelper
from src.enrichment.api_helpers.animeplanet_helper import AnimePlanetEnrichmentHelper
from src.enrichment.api_helpers.jikan_helper import JikanDetailedFetcher
from src.enrichment.api_helpers.animeschedule_fetcher import fetch_animeschedule_data

from .config import EnrichmentConfig

logger = logging.getLogger(__name__)


class ParallelAPIFetcher:
    """
    Fetches data from all anime APIs in parallel.
    Implements graceful degradation - continues with partial data if APIs fail.
    """
    
    def __init__(self, config: Optional[EnrichmentConfig] = None):
        """
        Initialize with configuration and API helpers.
        
        Args:
            config: Enrichment configuration (uses defaults if not provided)
        """
        self.config = config or EnrichmentConfig()
        
        # Initialize async helpers
        self.anilist_helper = None  # Lazy init in async context
        self.kitsu_helper = None
        self.anidb_helper = None
        self.anime_planet_helper = None
        
        # Track API performance
        self.api_timings: Dict[str, float] = {}
        self.api_errors: Dict[str, str] = {}
        
    async def initialize_helpers(self) -> None:
        """Initialize async API helpers."""
        if not self.anilist_helper:
            self.anilist_helper = AniListEnrichmentHelper()
        if not self.kitsu_helper:
            self.kitsu_helper = KitsuEnrichmentHelper()
        if not self.anidb_helper:
            self.anidb_helper = AniDBEnrichmentHelper()
        if not self.anime_planet_helper:
            self.anime_planet_helper = AnimePlanetEnrichmentHelper()
    
    async def fetch_all_data(
        self, 
        ids: Dict[str, str], 
        offline_data: Dict,
        temp_dir: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Fetch data from all APIs in parallel.
        
        Args:
            ids: Dictionary of platform IDs
            offline_data: Original offline anime data
            temp_dir: Optional temp directory for saving responses
            
        Returns:
            Dictionary with API responses
            
        Performance: 5-10 seconds (vs 30-60 seconds sequential)
        """
        await self.initialize_helpers()
        
        start_time = time.time()
        tasks: List[Tuple[str, Any]] = []
        
        # Create parallel tasks for each API
        if ids.get('mal_id'):
            tasks.append(('jikan', self._fetch_jikan_complete(ids['mal_id'], offline_data)))
        
        if ids.get('anilist_id'):
            tasks.append(('anilist', self._fetch_anilist(ids['anilist_id'])))
        
        if ids.get('kitsu_id'):
            tasks.append(('kitsu', self._fetch_kitsu(ids['kitsu_id'])))
        
        if ids.get('anidb_id'):
            tasks.append(('anidb', self._fetch_anidb(ids['anidb_id'])))
        
        if ids.get('anime_planet_slug'):
            tasks.append(('anime_planet', self._fetch_anime_planet(offline_data)))
        
        # Always try AnimSchedule with title search
        tasks.append(('animeschedule', self._fetch_animeschedule(offline_data)))
        
        # Execute all tasks in parallel with timeout
        results = await self._gather_with_timeout(tasks, timeout=self.config.api_timeout)
        
        # Save to temp files if directory provided
        if temp_dir:
            await self._save_temp_files(results, temp_dir)
        
        elapsed = time.time() - start_time
        logger.info(f"Fetched all API data in {elapsed:.2f} seconds")
        
        # Log performance metrics (context-rich logging)
        self._log_performance_metrics(elapsed)
        
        return results
    
    async def _fetch_jikan_complete(self, mal_id: str, offline_data: Dict) -> Optional[Dict[str, Any]]:
        """
        Fetch ALL Jikan data using the JikanDetailedFetcher helper.
        This properly handles rate limiting and batch processing for large series.
        """
        try:
            start = time.time()
            loop = asyncio.get_event_loop()
            
            # Create temp directory for this anime
            anime_title = offline_data.get('title', 'unknown')
            first_word = anime_title.split()[0] if anime_title else "unknown"
            clean_word = ''.join(c for c in first_word if c.isalnum() or c in '-_')
            temp_dir = os.path.join(self.config.temp_dir, clean_word)
            os.makedirs(temp_dir, exist_ok=True)
            
            # First, fetch anime full data
            anime_url = f"https://api.jikan.moe/v4/anime/{mal_id}/full"
            anime_data = await loop.run_in_executor(None, self._fetch_jikan_sync, anime_url)
            
            if not anime_data or not anime_data.get('data'):
                logger.warning(f"Failed to fetch Jikan anime data for MAL ID {mal_id}")
                return None
            
            anime_info = anime_data['data']
            # For ongoing series, episodes might be None - use offline data as fallback
            episode_count = anime_info.get('episodes')
            if episode_count is None:
                episode_count = offline_data.get('episodes', 0)
            
            # For episodes: Use JikanDetailedFetcher for large series
            if episode_count and episode_count > 100:
                logger.info(f"Using JikanDetailedFetcher for {episode_count} episodes...")
                # Save episode count to input file
                episodes_input = os.path.join(temp_dir, 'episodes.json')
                with open(episodes_input, 'w') as f:
                    json.dump({'episodes': episode_count}, f)
                
                # Use the helper's fetch_detailed_data method
                episodes_output = os.path.join(temp_dir, 'episodes_detailed.json')
                fetcher = JikanDetailedFetcher(mal_id, 'episodes')
                await loop.run_in_executor(None, fetcher.fetch_detailed_data, episodes_input, episodes_output)
                
                # Load the detailed episodes
                if os.path.exists(episodes_output):
                    with open(episodes_output, 'r') as f:
                        episodes_data = json.load(f)
                else:
                    episodes_data = []
            else:
                # For smaller series, fetch with pagination
                episodes_data = await self._fetch_all_jikan_episodes(mal_id, episode_count, loop)
            
            # For characters: Use JikanDetailedFetcher for detailed character data
            characters_url = f"https://api.jikan.moe/v4/anime/{mal_id}/characters"
            characters_basic = await loop.run_in_executor(None, self._fetch_jikan_sync, characters_url)
            
            if characters_basic and characters_basic.get('data') and len(characters_basic['data']) > 50:
                logger.info(f"Using JikanDetailedFetcher for {len(characters_basic['data'])} characters...")
                # Save characters to input file
                characters_input = os.path.join(temp_dir, 'characters.json')
                with open(characters_input, 'w') as f:
                    json.dump(characters_basic, f)
                
                # Use the helper's fetch_detailed_data method
                characters_output = os.path.join(temp_dir, 'characters_detailed.json')
                fetcher = JikanDetailedFetcher(mal_id, 'characters')
                await loop.run_in_executor(None, fetcher.fetch_detailed_data, characters_input, characters_output)
                
                # Load the detailed characters
                if os.path.exists(characters_output):
                    with open(characters_output, 'r') as f:
                        characters_data = json.load(f)
                else:
                    characters_data = characters_basic.get('data', [])
            else:
                characters_data = characters_basic.get('data', []) if characters_basic else []
            
            result = {
                'anime': anime_info,
                'episodes': episodes_data if isinstance(episodes_data, list) else [],
                'characters': characters_data
            }
            
            self.api_timings['jikan'] = time.time() - start
            logger.info(f"Jikan fetched: {len(result['episodes'])} episodes, {len(result['characters'])} characters in {self.api_timings['jikan']:.2f}s")
            return result
            
        except Exception as e:
            logger.error(f"Jikan fetch failed for MAL ID {mal_id}: {e}")
            self.api_errors['jikan'] = str(e)
            return None
    
    async def _fetch_all_jikan_episodes(self, mal_id: str, episode_count: int, loop) -> List[Dict]:
        """Fetch ALL episodes with pagination."""
        if episode_count == 0:
            return []
        
        all_episodes = []
        page = 1
        
        while True:
            url = f"https://api.jikan.moe/v4/anime/{mal_id}/episodes?page={page}"
            data = await loop.run_in_executor(None, self._fetch_jikan_sync, url)
            
            if not data or not data.get('data'):
                break
            
            all_episodes.extend(data['data'])
            
            # Check if there are more pages
            pagination = data.get('pagination', {})
            if not pagination.get('has_next_page', False):
                break
            
            page += 1
            
            # Log progress for long-running series
            if len(all_episodes) % 100 == 0:
                logger.debug(f"Fetched {len(all_episodes)}/{episode_count} episodes...")
        
        return all_episodes
    
    async def _fetch_all_jikan_characters(self, mal_id: str, loop) -> List[Dict]:
        """Fetch ALL characters with pagination."""
        all_characters = []
        has_next = True
        page = 1
        
        # First, get initial page to see how many there are
        url = f"https://api.jikan.moe/v4/anime/{mal_id}/characters"
        data = await loop.run_in_executor(None, self._fetch_jikan_sync, url)
        
        if data and data.get('data'):
            all_characters.extend(data['data'])
            
            # Jikan v4 doesn't paginate characters endpoint directly
            # It returns all characters in one response
            # But we should verify we got them all
            logger.debug(f"Fetched {len(all_characters)} characters from Jikan")
        
        return all_characters
    
    def _fetch_jikan_sync(self, url: str) -> Optional[Dict]:
        """Synchronous Jikan API fetch with rate limiting."""
        import requests
        import time
        
        # Respect Jikan rate limits (3 req/sec)
        time.sleep(0.35)  # ~3 requests per second
        
        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                return response.json()
            else:
                logger.warning(f"Jikan API returned status {response.status_code} for {url}")
                return None
        except Exception as e:
            logger.error(f"Jikan API request failed: {e}")
            return None
    
    async def _fetch_anilist(self, anilist_id: str) -> Optional[Dict]:
        """Fetch ALL AniList data using the helper as designed - with proper pagination."""
        try:
            start = time.time()
            
            logger.info(f"Fetching AniList data for ID {anilist_id} (will fetch ALL characters, staff, episodes)...")
            
            # The helper handles everything - pagination, rate limiting, etc.
            # Just call it and let it work
            if self.anilist_helper is None:
                raise RuntimeError("AniList helper not initialized")
            result = await self.anilist_helper.fetch_all_data_by_anilist_id(int(anilist_id))
            
            elapsed = time.time() - start
            self.api_timings['anilist'] = elapsed
            
            if result:
                # Log what we got
                chars = len(result.get('characters', {}).get('edges', []))
                staff = len(result.get('staff', {}).get('edges', []))
                eps = len(result.get('airingSchedule', {}).get('edges', []))
                logger.info(f"AniList fetched: {chars} characters, {staff} staff, {eps} episodes in {elapsed:.2f}s")
            else:
                logger.warning(f"AniList returned no data for ID {anilist_id}")
            
            return result
        except Exception as e:
            logger.error(f"AniList fetch failed for ID {anilist_id}: {e}")
            self.api_errors['anilist'] = str(e)
            return None
    
    async def _fetch_kitsu(self, kitsu_id: str) -> Optional[Dict]:
        """Fetch Kitsu data using async helper."""
        try:
            start = time.time()
            
            # Check if it's numeric or slug
            try:
                # Try as numeric ID first
                numeric_id = int(kitsu_id)
                if self.kitsu_helper is None:
                    raise RuntimeError("Kitsu helper not initialized")
                result = await self.kitsu_helper.fetch_all_data(numeric_id)
            except ValueError:
                # If not numeric, it's a slug - need to resolve to ID first
                logger.info(f"Resolving Kitsu slug '{kitsu_id}' to numeric ID...")
                
                # Use Kitsu API to search by slug
                import aiohttp
                async with aiohttp.ClientSession() as session:
                    url = f"https://kitsu.io/api/edge/anime?filter[slug]={kitsu_id}"
                    async with session.get(url, timeout=aiohttp.ClientTimeout(total=5)) as response:
                        if response.status == 200:
                            data = await response.json()
                            if data.get('data'):
                                numeric_id = int(data['data'][0]['id'])
                                logger.info(f"Resolved slug '{kitsu_id}' to ID {numeric_id}")
                                if self.kitsu_helper is None:
                                    raise RuntimeError("Kitsu helper not initialized")
                                result = await self.kitsu_helper.fetch_all_data(numeric_id)
                            else:
                                logger.warning(f"No Kitsu anime found for slug: {kitsu_id}")
                                result = None
                        else:
                            logger.warning(f"Failed to resolve Kitsu slug: {response.status}")
                            result = None
            
            self.api_timings['kitsu'] = time.time() - start
            return result
        except Exception as e:
            logger.error(f"Kitsu fetch failed for ID {kitsu_id}: {e}")
            self.api_errors['kitsu'] = str(e)
            return None
    
    async def _fetch_anidb(self, anidb_id: str) -> Optional[Dict]:
        """Fetch AniDB data using async helper."""
        try:
            start = time.time()
            if self.anidb_helper is None:
                raise RuntimeError("AniDB helper not initialized")
            result = await self.anidb_helper.fetch_all_data(int(anidb_id))
            self.api_timings['anidb'] = time.time() - start
            return result
        except Exception as e:
            logger.error(f"AniDB fetch failed for ID {anidb_id}: {e}")
            self.api_errors['anidb'] = str(e)
            return None
    
    async def _fetch_anime_planet(self, offline_data: Dict) -> Optional[Dict]:
        """Fetch Anime-Planet data using scraper."""
        try:
            start = time.time()
            if self.anime_planet_helper is None:
                raise RuntimeError("Anime Planet helper not initialized")
            result = await self.anime_planet_helper.fetch_all_data(offline_data)
            self.api_timings['anime_planet'] = time.time() - start
            return result
        except Exception as e:
            logger.error(f"Anime-Planet fetch failed: {e}")
            self.api_errors['anime_planet'] = str(e)
            return None
    
    async def _fetch_animeschedule(self, offline_data: Dict) -> Optional[Dict]:
        """
        Fetch AnimSchedule data using sync helper.
        Note: AnimSchedule helper is sync, so we run in executor.
        """
        try:
            start = time.time()
            
            # Get search term from offline data
            search_term = offline_data.get('title', '')
            if not search_term:
                return None
            
            # Run sync function in executor
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None, 
                fetch_animeschedule_data, 
                search_term
            )
            
            self.api_timings['animeschedule'] = time.time() - start
            return result
            
        except Exception as e:
            logger.error(f"AnimSchedule fetch failed: {e}")
            self.api_errors['animeschedule'] = str(e)
            return None
    
    async def _gather_with_timeout(
        self, 
        tasks: List[Tuple[str, Any]], 
        timeout: int
    ) -> Dict[str, Any]:
        """
        Execute tasks in parallel. In no-timeout mode, wait for ALL data.
        Implements graceful degradation - doesn't fail if one API is down.
        
        Args:
            tasks: List of (name, coroutine) tuples
            timeout: Timeout in seconds (ignored if no_timeout_mode is True)
            
        Returns:
            Dictionary of results
        """
        results = {}
        
        # Create tasks with names
        named_tasks = []
        for name, coro in tasks:
            task = asyncio.create_task(coro)
            named_tasks.append((name, task))
        
        # Check if we're in no-timeout mode
        if self.config.no_timeout_mode:
            logger.info("Running in NO TIMEOUT mode - will fetch ALL data from all APIs")
            # Just gather all results without timeout
            for name, task in named_tasks:
                try:
                    result = await task
                    results[name] = result
                    
                    if result:
                        logger.debug(f"API {name} completed successfully")
                    else:
                        logger.warning(f"API {name} returned empty result")
                        
                except Exception as e:
                    logger.error(f"API {name} failed with error: {e}")
                    results[name] = None
                    self.api_errors[name] = str(e)
        else:
            # Normal mode with timeouts
            for name, task in named_tasks:
                try:
                    # Each API gets its own timeout
                    result = await asyncio.wait_for(task, timeout=timeout)
                    results[name] = result
                    
                    if result:
                        logger.debug(f"API {name} completed successfully")
                    else:
                        logger.warning(f"API {name} returned empty result")
                        
                except asyncio.TimeoutError:
                    logger.warning(f"API {name} timed out after {timeout}s")
                    results[name] = None
                    task.cancel()
                    
                except Exception as e:
                    logger.error(f"API {name} failed with error: {e}")
                    results[name] = None
                    self.api_errors[name] = str(e)
        
        return results
    
    async def _save_temp_files(self, results: Dict[str, Any], temp_dir: str) -> None:
        """Save API responses to temp files for debugging/caching."""
        os.makedirs(temp_dir, exist_ok=True)
        
        for api_name, data in results.items():
            if data:
                file_path = os.path.join(temp_dir, f"{api_name}.json")
                try:
                    with open(file_path, 'w', encoding='utf-8') as f:
                        json.dump(data, f, ensure_ascii=False, indent=2)
                    logger.debug(f"Saved {api_name} response to {file_path}")
                except Exception as e:
                    logger.warning(f"Failed to save {api_name} response: {e}")
    
    def _log_performance_metrics(self, total_time: float) -> None:
        """Log detailed performance metrics for optimization."""
        logger.info("API Performance Metrics:")
        logger.info(f"  Total Time: {total_time:.2f}s")
        
        for api, timing in self.api_timings.items():
            logger.info(f"  {api}: {timing:.2f}s")
        
        if self.api_errors:
            logger.warning("API Errors:")
            for api, error in self.api_errors.items():
                logger.warning(f"  {api}: {error}")
        
        # Calculate success rate
        total_apis = len(self.api_timings) + len(self.api_errors)
        success_rate = (len(self.api_timings) / total_apis * 100) if total_apis > 0 else 0
        logger.info(f"  Success Rate: {success_rate:.1f}%")
    
    async def cleanup(self) -> None:
        """Clean up resources."""
        if self.anilist_helper:
            await self.anilist_helper.close()
        # Add cleanup for other helpers as needed