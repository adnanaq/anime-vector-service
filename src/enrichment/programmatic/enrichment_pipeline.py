"""
Main programmatic enrichment pipeline.
Orchestrates ID extraction, parallel API fetching, and episode processing.
Target performance: 10-30 seconds per anime (vs 5-15 minutes with AI).
"""

import asyncio
import json
import os
import time
from typing import Dict, List, Optional, Any, cast
from pathlib import Path
import logging

from .id_extractor import PlatformIDExtractor
from .api_fetcher import ParallelAPIFetcher
from .episode_processor import EpisodeProcessor
from .assembly import assemble_anime_entry, validate_and_fix_entry
from .config import EnrichmentConfig

logger = logging.getLogger(__name__)


class ProgrammaticEnrichmentPipeline:
    """
    Main orchestrator for programmatic enrichment.
    Implements Steps 1-3 of enrichment process programmatically.
    """
    
    def __init__(self, config: Optional[EnrichmentConfig] = None):
        """
        Initialize pipeline with configuration.
        
        Args:
            config: Enrichment configuration (uses defaults if not provided)
        """
        self.config = config or EnrichmentConfig()
        
        # Initialize components
        self.id_extractor = PlatformIDExtractor()
        self.api_fetcher = ParallelAPIFetcher(config)
        self.episode_processor = EpisodeProcessor()
        
        # Performance tracking
        self.timing_breakdown: Dict[str, float] = {}
        
        # Log configuration if verbose
        if self.config.verbose_logging:
            self.config.log_configuration()
    
    async def enrich_anime(self, offline_data: Dict) -> Dict[str, Any]:
        """
        Enrich a single anime with data from all APIs.
        
        Args:
            offline_data: Offline anime data from database
            
        Returns:
            Enriched anime data ready for AI enhancement
            
        Performance: 10-30 seconds (vs 5-15 minutes with AI)
        """
        start_time = time.time()
        anime_title = offline_data.get('title', 'Unknown')
        
        logger.info(f"Starting programmatic enrichment for: {anime_title}")
        
        try:
            # Step 1: Extract platform IDs (instant)
            step1_start = time.time()
            ids = self.id_extractor.extract_all_ids(offline_data)
            valid_ids = self.id_extractor.validate_ids(ids)
            self.timing_breakdown['id_extraction'] = time.time() - step1_start
            
            logger.info(f"Step 1 complete: Extracted {len(valid_ids)} platform IDs in {self.timing_breakdown['id_extraction']:.3f}s")
            
            # Create temp directory for this anime
            temp_dir = self._create_temp_dir(anime_title)
            
            # Step 2: Parallel API fetching (5-10 seconds)
            step2_start = time.time()
            api_data = await self.api_fetcher.fetch_all_data(
                valid_ids, 
                offline_data,
                temp_dir
            )
            self.timing_breakdown['api_fetching'] = time.time() - step2_start
            
            # Count successful API responses
            successful_apis = sum(1 for v in api_data.values() if v)
            logger.info(f"Step 2 complete: Fetched data from {successful_apis} APIs in {self.timing_breakdown['api_fetching']:.2f}s")
            
            # Step 3: Process episodes (instant)
            step3_start = time.time()
            processed_episodes = self._process_episodes(api_data)
            self.timing_breakdown['episode_processing'] = time.time() - step3_start
            
            logger.info(f"Step 3 complete: Processed {len(processed_episodes)} episodes in {self.timing_breakdown['episode_processing']:.3f}s")
            
            # Compile results
            result = {
                'offline_data': offline_data,
                'extracted_ids': valid_ids,
                'api_data': api_data,
                'processed_episodes': processed_episodes,
                'episode_statistics': self.episode_processor.extract_episode_statistics(processed_episodes),
                'enrichment_metadata': {
                    'method': 'programmatic',
                    'total_time': time.time() - start_time,
                    'timing_breakdown': self.timing_breakdown.copy(),
                    'successful_apis': successful_apis,
                    'temp_directory': temp_dir
                }
            }
            
            total_time = time.time() - start_time
            logger.info(f"✓ Enrichment complete for {anime_title} in {total_time:.2f}s")
            
            return result
            
        except Exception as e:
            logger.error(f"Enrichment failed for {anime_title}: {e}")
            if self.config.skip_failed_apis:
                # Return partial data on failure (graceful degradation)
                return {
                    'offline_data': offline_data,
                    'error': str(e),
                    'partial_data': True
                }
            raise
    
    async def enrich_batch(self, anime_list: List[Dict]) -> List[Dict]:
        """
        Enrich multiple anime in parallel.
        
        Args:
            anime_list: List of offline anime data
            
        Returns:
            List of enriched anime data
            
        Performance: Processes batch_size anime concurrently
        """
        logger.info(f"Starting batch enrichment for {len(anime_list)} anime")
        
        # Create semaphore to limit concurrent processing
        semaphore = asyncio.Semaphore(self.config.batch_size)
        
        async def enrich_with_limit(anime):
            async with semaphore:
                return await self.enrich_anime(anime)
        
        # Process all anime concurrently (limited by semaphore)
        tasks = [enrich_with_limit(anime) for anime in anime_list]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle results and exceptions
        successful: List[Dict[str, Any]] = []
        failed = []

        for anime, result in zip(anime_list, results):
            if isinstance(result, Exception):
                logger.error(f"Failed to enrich {anime.get('title')}: {result}")
                failed.append(anime)
            else:
                # result is guaranteed to be Dict[str, Any] here due to the isinstance check above
                successful.append(cast(Dict[str, Any], result))
        
        logger.info(f"Batch complete: {len(successful)} successful, {len(failed)} failed")
        
        return successful
    
    def _create_temp_dir(self, anime_title: str) -> str:
        """Create temp directory for anime processing."""
        # Get first word from title for directory name
        first_word = anime_title.split()[0] if anime_title else "unknown"
        # Remove special characters
        clean_word = ''.join(c for c in first_word if c.isalnum() or c in '-_')
        
        temp_dir = os.path.join(self.config.temp_dir, clean_word)
        os.makedirs(temp_dir, exist_ok=True)
        
        return temp_dir
    
    def _process_episodes(self, api_data: Dict) -> List[Dict]:
        """Process and merge episode data from all APIs."""
        episode_sources = []
        
        # Extract episodes from each API response
        if jikan_data := api_data.get('jikan'):
            if episodes := jikan_data.get('episodes'):
                episode_sources.append(episodes)
        
        if anilist_data := api_data.get('anilist'):
            if episodes := anilist_data.get('airingSchedule', {}).get('edges'):
                episode_sources.append(episodes)
        
        # Process and merge all episode sources
        if episode_sources:
            merged = self.episode_processor.merge_episode_sources(*episode_sources)
            return self.episode_processor.validate_episode_data(merged)
        
        return []
    
    async def load_and_enrich_from_file(self, file_path: str) -> Dict:
        """
        Load anime from file and enrich it.
        
        Args:
            file_path: Path to JSON file with offline anime data
            
        Returns:
            Enriched anime data
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            offline_data = json.load(f)
        
        return await self.enrich_anime(offline_data)
    
    async def enrich_anime_with_assembly(
        self, 
        offline_data: Dict, 
        stage_outputs_dir: Optional[Path] = None
    ) -> Dict[str, Any]:
        """
        Complete enrichment pipeline including Step 5 assembly.
        
        Args:
            offline_data: Raw anime data from offline database
            stage_outputs_dir: Directory containing AI stage outputs (stage1-6 JSON files)
            
        Returns:
            Complete assembled and validated AnimeEntry
        """
        anime_title = offline_data.get('title', 'Unknown')
        logger.info(f"Starting complete enrichment pipeline for {anime_title}")
        
        # Step 1-3: Programmatic enrichment
        programmatic_result = await self.enrich_anime(offline_data)
        
        if 'error' in programmatic_result:
            logger.error(f"Programmatic enrichment failed: {programmatic_result['error']}")
            return programmatic_result
        
        # Step 5: Assembly (if stage outputs available)
        if stage_outputs_dir and stage_outputs_dir.exists():
            logger.info(f"Running Step 5 assembly...")
            
            try:
                # Extract anime sources
                anime_sources = offline_data.get('sources', [])
                
                # Run assembly
                assembly_result = assemble_anime_entry(
                    stage_dir=stage_outputs_dir,
                    programmatic_data=programmatic_result,
                    anime_sources=anime_sources
                )
                
                if assembly_result.success and assembly_result.anime_entry:
                    logger.info(f"Assembly successful with {len(assembly_result.warnings)} warnings")
                    
                    # Apply final validation and auto-fix
                    final_entry, is_valid, validation_messages = validate_and_fix_entry(
                        assembly_result.anime_entry
                    )
                    
                    # Add assembly metadata to result
                    programmatic_result['assembled_entry'] = final_entry
                    programmatic_result['assembly_success'] = assembly_result.success
                    programmatic_result['validation_passed'] = is_valid
                    programmatic_result['assembly_errors'] = assembly_result.errors
                    programmatic_result['assembly_warnings'] = assembly_result.warnings
                    programmatic_result['validation_messages'] = validation_messages
                    
                    logger.info(f"Complete pipeline finished - Validation: {is_valid}")
                    
                else:
                    logger.error(f"Assembly failed: {assembly_result.errors}")
                    programmatic_result['assembly_errors'] = assembly_result.errors
                    programmatic_result['assembly_success'] = False
                    
            except Exception as e:
                logger.error(f"Step 5 assembly failed: {e}")
                programmatic_result['assembly_error'] = str(e)
                programmatic_result['assembly_success'] = False
        else:
            logger.info(f"No stage outputs found, skipping Step 5 assembly")
            programmatic_result['assembly_skipped'] = True
        
        return programmatic_result
    
    async def cleanup(self):
        """Clean up resources."""
        await self.api_fetcher.cleanup()
    
    def get_performance_report(self) -> str:
        """Generate performance report."""
        report = ["Performance Report:"]
        report.append(f"  Total APIs configured: {self.config.max_concurrent_apis}")
        report.append(f"  Batch size: {self.config.batch_size}")
        
        if self.timing_breakdown:
            report.append("\nTiming Breakdown:")
            for step, time_taken in self.timing_breakdown.items():
                report.append(f"  {step}: {time_taken:.3f}s")
        
        if self.api_fetcher.api_timings:
            report.append("\nAPI Response Times:")
            for api, time_taken in self.api_fetcher.api_timings.items():
                report.append(f"  {api}: {time_taken:.2f}s")
        
        return "\n".join(report)


async def main():
    """Test the pipeline with a sample anime."""
    
    # Sample offline data
    sample_anime = {
        "sources": [
            "https://myanimelist.net/anime/21/One_Piece",
            "https://anilist.co/anime/21",
            "https://kitsu.io/anime/one-piece"
        ],
        "title": "One Piece",
        "episodes": 1000,
        "type": "TV",
        "status": "Currently Airing"
    }
    
    # Initialize pipeline
    pipeline = ProgrammaticEnrichmentPipeline()
    
    try:
        # Run enrichment
        result = await pipeline.enrich_anime(sample_anime)
        
        # Print results
        print("\n" + "="*60)
        print("ENRICHMENT RESULTS")
        print("="*60)
        
        print(f"\nExtracted IDs: {result['extracted_ids']}")
        print(f"\nSuccessful APIs: {result['enrichment_metadata']['successful_apis']}")
        print(f"Total Time: {result['enrichment_metadata']['total_time']:.2f}s")
        
        print("\n" + pipeline.get_performance_report())
        
        # Save result
        output_file = "programmatic_enrichment_test.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2, default=str)
        print(f"\nResults saved to {output_file}")
        
    finally:
        await pipeline.cleanup()


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())