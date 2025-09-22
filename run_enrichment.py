#!/usr/bin/env python3
"""
Minimal script to trigger the enrichment pipeline with an anime entry.
The pipeline handles all processing internally.
"""

import asyncio

from src.enrichment.programmatic.enrichment_pipeline import (
    ProgrammaticEnrichmentPipeline,
)


async def main():
    # Initialize pipeline
    pipeline = ProgrammaticEnrichmentPipeline()

    # Anime entry (can be modified as needed)
    anime_entry = {
        "sources": [
            "https://anidb.net/anime/69",
            "https://anilist.co/anime/21",
            "https://anime-planet.com/anime/one-piece",
            "https://animecountdown.com/38636",
            "https://anisearch.com/anime/2227",
            "https://kitsu.app/anime/12",
            "https://livechart.me/anime/321",
            "https://myanimelist.net/anime/21",
            "https://notify.moe/anime/jdZp5KmiR",
            "https://simkl.com/anime/38636",
        ],
        "title": "One Piece",
        "episodes": 1139,
        "type": "TV",
        "status": "Currently Airing",
    }

    # Run enrichment
    result = await pipeline.enrich_anime(anime_entry)

    # Show results
    for api_name, data in result["api_data"].items():
        status = "✓" if data else "✗"
        print(f"{api_name}: {status}")

    print(f"Time: {result['enrichment_metadata']['total_time']:.2f}s")

    # Cleanup
    await pipeline.cleanup()


if __name__ == "__main__":
    asyncio.run(main())

