"""
This script crawls episode information from a given anisearch.com anime URL.

It accepts a URL as a command-line argument. It then uses the crawl4ai
library to extract episode data based on a predefined CSS schema.
The extracted data is processed to clean up the episode number.

The final processed data, a list of episodes with their details, is saved
to 'anisearch_episodes.json' in the project root.

Usage:
    python episode_crawler.py <anisearch_url>
"""

import argparse
import asyncio
import json
import re
from typing import Optional

from crawl4ai import (
    AsyncWebCrawler,
    CrawlerRunConfig,
    CrawlResult,
    JsonCssExtractionStrategy,
)
from crawl4ai.types import RunManyReturn

from src.cache_manager.result_cache import cached_result

from .utils import sanitize_output_path


@cached_result(ttl=86400, key_prefix="anisearch_episodes")  # 24 hours cache
async def fetch_anisearch_episodes(
    url: str, return_data: bool = True, output_path: Optional[str] = None
) -> Optional[list]:
    """
    Crawl and extract episode metadata from an anisearch.com anime episodes page.
    
    Extracted fields include episodeNumber (normalized to an integer when present), runtime, releaseDate, and title. Results are cached for 24 hours (Redis) under the function's cache key. Optionally writes the cleaned results to a JSON file.
    
    Parameters:
        url (str): The anisearch.com URL of the anime episodes page to crawl.
        return_data (bool): If True, return the parsed list of episode dictionaries; if False, do not return the data.
        output_path (Optional[str]): Optional file path to write the JSON output; if provided, the data is saved there.
    
    Returns:
        list[dict] | None: A list of episode dictionaries when `return_data` is True and extraction succeeds, otherwise `None`.
    """
    css_schema = {
        "baseSelector": "tr[data-episode='true']",
        "fields": [
            {
                "name": "episodeNumber",
                "selector": "th[itemprop='episodeNumber']",
                "type": "text",
            },
            {
                "name": "runtime",
                "selector": "td[data-title='Runtime'] div[lang='en']",
                "type": "text",
            },
            {
                "name": "releaseDate",
                "selector": "td[data-title='Date of Original Release'] div[lang='en']",
                "type": "text",
            },
            {
                "name": "title",
                "selector": "td[data-title='Title'] span[itemprop='name'][lang='en']",
                "type": "text",
            },
        ],
    }

    extraction_strategy = JsonCssExtractionStrategy(css_schema)
    config = CrawlerRunConfig(extraction_strategy=extraction_strategy)

    async with AsyncWebCrawler() as crawler:
        results: RunManyReturn = await crawler.arun(url=url, config=config)

        if not results:
            print("No results found.")
            return None

        for result in results:
            if not isinstance(result, CrawlResult):
                raise TypeError(
                    f"Unexpected result type: {type(result)}, expected CrawlResult."
                )

            if result.success and result.extracted_content:
                data = json.loads(result.extracted_content)
                # Clean up the data
                for item in data:
                    if "episodeNumber" in item and item["episodeNumber"]:
                        match = re.search(r"\d+", item["episodeNumber"])
                        if match:
                            item["episodeNumber"] = int(match.group(0))

                # Conditionally write to file
                if output_path:
                    safe_path = sanitize_output_path(output_path)
                    with open(safe_path, "w", encoding="utf-8") as f:
                        json.dump(data, f, ensure_ascii=False, indent=2)
                    print(f"Data written to {safe_path}")

                # Return data for programmatic usage
                if return_data:
                    return data

                return None
            else:
                print(f"Extraction failed: {result.error_message}")
                return None
        return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Crawl episode data from an anisearch.com URL."
    )
    parser.add_argument(
        "url", type=str, help="The anisearch.com URL for the anime episodes page."
    )
    parser.add_argument(
        "--output",
        type=str,
        default="anisearch_episodes.json",
        help="Output file path (default: anisearch_episodes.json in current directory)",
    )
    args = parser.parse_args()
    asyncio.run(
        fetch_anisearch_episodes(
            args.url,
            return_data=False,  # CLI doesn't need return value
            output_path=args.output,
        )
    )