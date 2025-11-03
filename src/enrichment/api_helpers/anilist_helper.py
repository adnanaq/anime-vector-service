#!/usr/bin/env python3
"""
AniList Helper for AI Enrichment Integration

Test script to fetch and analyze AniList data for anime entries using GraphQL API.
"""

import argparse
import asyncio
import json
import logging
from typing import Any, Dict, List, Optional

import aiohttp

from src.cache_manager.instance import http_cache_manager as _cache_manager

logger = logging.getLogger(__name__)


class AniListEnrichmentHelper:
    """Helper for AniList data fetching in AI enrichment pipeline."""

    def __init__(self) -> None:
        """
        Create an AniListEnrichmentHelper and initialize internal state.
        
        Initializes:
        - base_url: AniList GraphQL endpoint.
        - session: HTTP session (created on demand; may become a cached or plain aiohttp session).
        - rate_limit_remaining: default remaining requests before throttling.
        - rate_limit_reset: optional timestamp when the rate limit resets.
        - _session_event_loop: event loop the session is bound to for per-loop session management.
        """
        self.base_url = "https://graphql.anilist.co"
        self.session: Optional[aiohttp.ClientSession] = None
        self.rate_limit_remaining = 90
        self.rate_limit_reset: Optional[int] = None
        self._session_event_loop: Optional[Any] = None

    async def _make_request(
        self, query: str, variables: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Send a GraphQL request to AniList and return the parsed response augmented with cache metadata.
        
        This method ensures an HTTP session bound to the current event loop (attempting a Redis-backed cached session and falling back to an uncached session), respects AniList rate limits by sleeping when limits are low, and retries after `Retry-After` when receiving HTTP 429. It adds an observable `_from_cache` flag to the returned result indicating whether the response was served from cache.
        
        Parameters:
            query (str): GraphQL query string to execute.
            variables (Optional[Dict[str, Any]]): GraphQL variables to include with the query.
        
        Returns:
            Dict[str, Any]: The GraphQL `data` object from the response with an additional `_from_cache` boolean. If the request fails or GraphQL `errors` are present, returns a dictionary containing `_from_cache` (typically `False` on failures).
        """
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        payload = {"query": query, "variables": variables or {}}

        # Check if we need to create/recreate session for current event loop
        current_loop = asyncio.get_running_loop()
        if self.session is None or self._session_event_loop != current_loop:
            # Close old session if it exists
            if self.session is not None:
                try:
                    await self.session.close()
                except Exception:
                    pass  # Ignore errors closing old session

            # Create cached session for THIS event loop
            # Each event loop gets its own Redis client to avoid "different event loop" errors
            # This enables GraphQL caching while preventing event loop conflicts
            try:
                from src.cache_manager.aiohttp_adapter import CachedAiohttpSession
                from src.cache_manager.async_redis_storage import AsyncRedisStorage
                from redis.asyncio import Redis

                # Create NEW Redis client bound to current event loop
                redis_client = Redis.from_url(
                    "redis://localhost:6379/0",
                    decode_responses=False,
                    socket_connect_timeout=5.0,
                )

                # Create storage with event-loop-specific Redis client
                storage = AsyncRedisStorage(
                    client=redis_client,
                    default_ttl=86400.0,  # 24 hours
                    refresh_ttl_on_access=True,
                    key_prefix="hishel_cache",
                )

                # Create cached session with body-based caching for GraphQL
                # X-Hishel-Body-Key ensures different queries/variables get different cache entries
                headers = {"X-Hishel-Body-Key": "true"}
                self.session = CachedAiohttpSession(
                    storage=storage,
                    timeout=aiohttp.ClientTimeout(total=None),
                    headers=headers,
                )
                logger.debug("AniList cached session created for current event loop")
            except Exception as e:
                logger.warning(f"Failed to create cached session: {e}, using uncached")
                # Fallback to uncached session
                self.session = aiohttp.ClientSession(
                    timeout=aiohttp.ClientTimeout(total=None)
                )

            self._session_event_loop = current_loop

        try:
            if self.rate_limit_remaining < 5:
                logging.info(
                    f"Rate limit low ({self.rate_limit_remaining}), waiting 60 seconds..."
                )
                await asyncio.sleep(60)
                self.rate_limit_remaining = 90

            async with self.session.post(
                self.base_url, json=payload, headers=headers
            ) as response:
                # Capture cache status before response is consumed
                from_cache = getattr(response, 'from_cache', False)

                if "X-RateLimit-Remaining" in response.headers:
                    self.rate_limit_remaining = int(
                        response.headers["X-RateLimit-Remaining"]
                    )

                if response.status == 429:
                    retry_after = int(response.headers.get("Retry-After", 60))
                    logger.warning(
                        f"Rate limit exceeded. Waiting {retry_after} seconds..."
                    )
                    await asyncio.sleep(retry_after)
                    return await self._make_request(query, variables)

                response.raise_for_status()
                data: Any = await response.json()
                if "errors" in data:
                    logger.error(f"AniList GraphQL errors: {data['errors']}")
                    return {"_from_cache": from_cache}
                result: Dict[str, Any] = data.get("data", {})
                # Add cache metadata to result
                result["_from_cache"] = from_cache
                return result
        except Exception as e:
            logger.error(f"AniList API request failed: {e}")
            return {"_from_cache": False}

    def _get_media_query_fields(self) -> str:
        """
        GraphQL selection set for Media (anime) fields used when querying AniList.
        
        Returns:
            str: A multiline GraphQL field selection string that requests identifiers, titles, descriptions, images, format and airing metadata, scores and popularity, genres and tags, relations, studios, external links, streaming episodes, next airing episode, rankings, statistics, and the last update timestamp.
        """
        return """
        id
        idMal
        title {
          romaji
          english
          native
          userPreferred
        }
        description(asHtml: false)
        source
        format
        episodes
        duration
        status
        season
        seasonYear
        countryOfOrigin
        isAdult
        hashtag
        coverImage {
          extraLarge
          large
          medium
          color
        }
        bannerImage
        trailer {
          id
          site
          thumbnail
        }
        averageScore
        meanScore
        popularity
        favourites
        trending
        genres
        synonyms
        tags {
          id
          name
          description
          category
          rank
          isGeneralSpoiler
          isMediaSpoiler
          isAdult
        }
        relations {
          edges {
            node {
              id
              title {
                romaji
                english
              }
              format
              status
            }
            relationType
          }
        }
        studios {
          edges {
            node {
              id
              name
              isAnimationStudio
            }
            isMain
          }
        }
        externalLinks {
          id
          url
          site
          type
          language
          color
          icon
        }
        streamingEpisodes {
          title
          thumbnail
          url
          site
        }
        nextAiringEpisode {
          episode
          airingAt
          timeUntilAiring
        }
        rankings {
          id
          rank
          type
          format
          year
          season
          allTime
          context
        }
        stats {
          scoreDistribution {
            score
            amount
          }
          statusDistribution {
            status
            amount
          }
        }
        updatedAt
        """

    # def _build_query_by_mal_id(self) -> str:
    #     return f"query ($idMal: Int) {{ Media(idMal: $idMal, type: ANIME) {{ {self._get_media_query_fields()} }} }}"

    def _build_query_by_anilist_id(self) -> str:
        return f"query ($id: Int) {{ Media(id: $id, type: ANIME) {{ {self._get_media_query_fields()} }} }}"

    # async def fetch_anime_by_mal_id(self, mal_id: int) -> Optional[Dict[str, Any]]:
    #     query = self._build_query_by_mal_id()
    #     variables = {"idMal": mal_id}
    #     response = await self._make_request(query, variables)
    #     return response.get("Media")

    async def fetch_anime_by_anilist_id(
        self, anilist_id: int
    ) -> Optional[Dict[str, Any]]:
        query = self._build_query_by_anilist_id()
        variables = {"id": anilist_id}
        response = await self._make_request(query, variables)
        return response.get("Media")

    async def _fetch_paginated_data(
        self, anilist_id: int, query_template: str, data_key: str
    ) -> List[Dict[str, Any]]:
        """
        Fetches and accumulates all paginated edge items for a given AniList media ID using the provided GraphQL query.
        
        Parameters:
            anilist_id (int): AniList ID of the media to query.
            query_template (str): GraphQL query string that accepts `id` and `page` variables and returns a paginated connection under `Media`.
            data_key (str): Key inside the returned `Media` object that holds the paginated connection (for example, `"characters"`, `"staff"`, or `"airingSchedule"`).
        
        Returns:
            List[Dict[str, Any]]: A list of edge objects collected from every page for the specified `data_key`. Returns an empty list if no data is found.
        """
        all_items = []
        page = 1
        has_next_page = True
        while has_next_page:
            variables = {"id": anilist_id, "page": page}
            response = await self._make_request(query_template, variables)
            if (
                not response
                or not response.get("Media")
                or not response["Media"].get(data_key)
            ):
                break
            data = response["Media"][data_key]
            all_items.extend(data.get("edges", []))
            has_next_page = data.get("pageInfo", {}).get("hasNextPage", False)
            page += 1

            # Only rate limit for network requests, not cache hits
            # Cache hits are instant, no need to throttle
            if not response.get('_from_cache', False):
                await asyncio.sleep(0.5)

        return all_items

    async def fetch_all_characters(self, anilist_id: int) -> List[Dict[str, Any]]:
        """
        Fetches all character edges for the anime with the given AniList ID, aggregating paginated results.
        
        Returns:
            List[Dict[str, Any]]: A list of character edge objects from AniList; each entry typically contains 'node' (character details), 'role', and 'voiceActors'.
        """
        query = """
        query ($id: Int!, $page: Int!) {
          Media(id: $id, type: ANIME) {
            characters(page: $page, perPage: 25, sort: ROLE) {
              pageInfo { hasNextPage }
              edges {
                node {
                  id
                  name {
                    full
                    native
                    alternative
                    alternativeSpoiler
                  }
                  image { large medium }
                  favourites
                  gender
                }
                role
                voiceActors(language: JAPANESE) {
                  id
                  name { full native }
                }
              }
            }
          }
        }
        """
        return await self._fetch_paginated_data(anilist_id, query, "characters")

    async def fetch_all_staff(self, anilist_id: int) -> List[Dict[str, Any]]:
        query = """
        query ($id: Int!, $page: Int!) {
          Media(id: $id, type: ANIME) {
            staff(page: $page, perPage: 25, sort: RELEVANCE) {
              pageInfo { hasNextPage }
              edges { node { id name { full native } } role }
            }
          }
        }
        """
        return await self._fetch_paginated_data(anilist_id, query, "staff")

    async def fetch_all_episodes(self, anilist_id: int) -> List[Dict[str, Any]]:
        query = """
        query ($id: Int!, $page: Int!) {
          Media(id: $id, type: ANIME) {
            airingSchedule(page: $page, perPage: 50) {
              pageInfo { hasNextPage }
              edges { node { id episode airingAt } }
            }
          }
        }
        """
        return await self._fetch_paginated_data(anilist_id, query, "airingSchedule")

    async def _fetch_and_populate_details(
        self, anime_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        anilist_id = anime_data.get("id")
        if not anilist_id:
            return anime_data

        details = {
            "characters": await self.fetch_all_characters(anilist_id),
            "staff": await self.fetch_all_staff(anilist_id),
            "airingSchedule": await self.fetch_all_episodes(anilist_id),
        }

        for key, data in details.items():
            if data:
                anime_data[key] = {"edges": data}
                logger.info(f"Total {key} fetched: {len(data)}")

        return anime_data

    # async def fetch_all_data_by_mal_id(self, mal_id: int) -> Optional[Dict[str, Any]]:
    #     anime_data = await self.fetch_anime_by_mal_id(mal_id)
    #     if not anime_data:
    #         logger.warning(f"No AniList data found for MAL ID: {mal_id}")
    #         return None
    #     return await self._fetch_and_populate_details(anime_data)

    async def fetch_all_data_by_anilist_id(
        self, anilist_id: int
    ) -> Optional[Dict[str, Any]]:
        anime_data = await self.fetch_anime_by_anilist_id(anilist_id)
        if not anime_data:
            logger.warning(f"No AniList data found for AniList ID: {anilist_id}")
            return None
        return await self._fetch_and_populate_details(anime_data)

    async def close(self) -> None:
        if self.session:
            await self.session.close()


async def main() -> None:
    parser = argparse.ArgumentParser(description="Test AniList data fetching")
    group = parser.add_mutually_exclusive_group(required=True)
    # group.add_argument("--mal-id", type=int, help="MyAnimeList ID to fetch")
    group.add_argument("--anilist-id", type=int, help="AniList ID to fetch")
    parser.add_argument(
        "--output",
        type=str,
        default="test_anilist_output.json",
        help="Output file path",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    helper = AniListEnrichmentHelper()
    anime_data = None
    try:
        if args.anilist_id:
            anime_data = await helper.fetch_all_data_by_anilist_id(args.anilist_id)
        # elif args.mal_id:
        #     anime_data = await helper.fetch_all_data_by_mal_id(args.mal_id)

        if anime_data:
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(anime_data, f, indent=2, ensure_ascii=False)
            logger.info(f"Data saved to {args.output}")
        else:
            logger.error("No data found for the given ID.")
    finally:
        await helper.close()


if __name__ == "__main__":
    asyncio.run(main())