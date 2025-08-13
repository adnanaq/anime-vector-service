"""
Programmatic enrichment modules for deterministic operations.
Following async-first architecture and configuration-driven patterns from lessons learned.
"""

from .id_extractor import PlatformIDExtractor
from .api_fetcher import ParallelAPIFetcher
from .episode_processor import EpisodeProcessor
from .enrichment_pipeline import ProgrammaticEnrichmentPipeline
from .config import EnrichmentConfig

__all__ = [
    'PlatformIDExtractor',
    'ParallelAPIFetcher', 
    'EpisodeProcessor',
    'ProgrammaticEnrichmentPipeline',
    'EnrichmentConfig'
]