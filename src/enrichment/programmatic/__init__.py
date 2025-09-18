"""
Programmatic enrichment modules for deterministic operations.
Following async-first architecture and configuration-driven patterns from lessons learned.
"""

from .api_fetcher import ParallelAPIFetcher
from .config import EnrichmentConfig
from .enrichment_pipeline import ProgrammaticEnrichmentPipeline
from .episode_processor import EpisodeProcessor
from .id_extractor import PlatformIDExtractor

__all__ = [
    "PlatformIDExtractor",
    "ParallelAPIFetcher",
    "EpisodeProcessor",
    "ProgrammaticEnrichmentPipeline",
    "EnrichmentConfig",
]
