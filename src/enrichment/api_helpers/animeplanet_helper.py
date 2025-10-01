#!/usr/bin/env python3
"""
Anime-Planet Helper for AI Enrichment Integration

Helper function to fetch Anime-Planet data using the scraper for AI enrichment pipeline.
"""

import asyncio
import json
import logging
import os
import re
import sys
from typing import Any, Dict, Optional

# Add project root to path to allow absolute imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from ..scrapers.anime_planet_scraper import AnimePlanetScraper

logger = logging.getLogger(__name__)


class AnimePlanetEnrichmentHelper:
    """Helper for Anime-Planet data fetching in AI enrichment pipeline."""

    def __init__(self) -> None:
        """Initialize Anime-Planet enrichment helper."""
        self.scraper = AnimePlanetScraper()

    async def extract_slug_from_url(self, url: str) -> Optional[str]:
        """Extract Anime-Planet slug from URL."""
        try:
            # Pattern: https://www.anime-planet.com/anime/SLUG
            match = re.search(r"anime-planet\.com/anime/([^/?]+)", url)
            if match:
                return match.group(1)
            return None
        except Exception as e:
            logger.error(f"Error extracting slug from URL {url}: {e}")
            return None

    async def find_animeplanet_url(
        self, offline_anime_data: Dict[str, Any]
    ) -> Optional[str]:
        """Find Anime-Planet URL from offline anime data sources."""
        try:
            sources = offline_anime_data.get("sources", [])
            for source in sources:
                if "anime-planet.com" in source:
                    return source
            return None
        except Exception as e:
            logger.error(f"Error finding Anime-Planet URL: {e}")
            return None

    async def search_anime_by_title(
        self, title: str, limit: int = 5
    ) -> Optional[Dict[str, Any]]:
        """Search for anime by title and return the best match."""
        try:
            search_results = await self.scraper.search_anime(title, limit=limit)
            if search_results:
                # Return the first result as the best match
                return search_results[0]
            return None
        except Exception as e:
            logger.error(f"Error searching anime by title '{title}': {e}")
            return None

    async def fetch_anime_data(self, slug: str) -> Optional[Dict[str, Any]]:
        """Fetch anime data by slug, including characters."""
        try:
            # Fetch main anime data
            anime_data = await self.scraper.get_anime_by_slug(slug)
            if not anime_data:
                return None

            # Fetch character data
            try:
                characters_data = await self.scraper.get_anime_characters(slug)
                if characters_data:
                    anime_data["characters"] = characters_data.get("characters", [])
                    anime_data["character_count"] = characters_data.get(
                        "total_count", 0
                    )
                    logger.info(
                        f"Fetched {anime_data['character_count']} characters for '{slug}'"
                    )
            except Exception as e:
                logger.warning(f"Failed to fetch characters for '{slug}': {e}")
                # Continue without characters data

            return anime_data
        except Exception as e:
            logger.error(f"Error fetching anime data for slug '{slug}': {e}")
            return None

    async def fetch_all_data(
        self, offline_anime_data: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        Fetch all Anime-Planet data for an anime.

        Args:
            offline_anime_data: The offline anime data containing sources

        Returns:
            Dict containing Anime-Planet data or None if not found
        """
        try:
            # First try to find direct URL in sources
            animeplanet_url = await self.find_animeplanet_url(offline_anime_data)

            if animeplanet_url:
                # Extract slug from URL
                slug = await self.extract_slug_from_url(animeplanet_url)
                if slug:
                    logger.info(f"Found Anime-Planet slug: {slug}")
                    return await self.fetch_anime_data(slug)

            # If no direct URL, try searching by title
            title = offline_anime_data.get("title")
            if title:
                logger.info(f"Searching Anime-Planet for title: {title}")
                search_result = await self.search_anime_by_title(title)
                if search_result and search_result.get("slug"):
                    return await self.fetch_anime_data(search_result["slug"])

            logger.warning("No Anime-Planet data found")
            return None

        except Exception as e:
            logger.error(f"Error in fetch_all_data: {e}")
            return None

    async def close(self) -> None:
        """Close the scraper."""
        await self.scraper.close()


async def main() -> None:
    if len(sys.argv) != 3:
        print("Usage: python animeplanet_helper.py <slug> <output_file>")
        sys.exit(1)

    slug = sys.argv[1]
    output_file = sys.argv[2]

    helper = AnimePlanetEnrichmentHelper()
    data = await helper.fetch_anime_data(slug)
    await helper.close()

    if data:
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        print(f"Data for {slug} saved to {output_file}")
    else:
        print(f"Could not fetch data for {slug}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())
