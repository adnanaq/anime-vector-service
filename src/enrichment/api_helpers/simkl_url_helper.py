#!/usr/bin/env python3
"""
Lightweight Simkl URL helper for Stage 3 relationship processing using cloudscraper.

This uses the same cloudscraper approach that works for AniSearch to bypass anti-bot protection.

Usage:
    from src.batch_enrichment.simkl_url_helper import fetch_simkl_url_data
    
    result = fetch_simkl_url_data("https://simkl.com/anime/2308514")
    
    # Result format:
    # {
    #     "title": "Anime Title",
    #     "relationship": "Inferred relationship type",
    #     "confidence": "high|medium|low",
    #     "url": "original_url"
    # }
"""

import time
import re
from typing import Optional, Dict, Any
from bs4 import BeautifulSoup
import logging

try:
    import cloudscraper
    HAS_CLOUDSCRAPER = True
except ImportError:
    HAS_CLOUDSCRAPER = False

logger = logging.getLogger(__name__)

# MODB's dead entry detection pattern
DEAD_ENTRY_INDICATOR = '<meta property="og:title" content="Simkl - Watch and Track Movies, Anime, TV Shows" />'


def fetch_simkl_url_data(url: str) -> Optional[Dict[str, Any]]:
    """
    Fetch title and relationship data from a Simkl URL using cloudscraper.
    
    Uses the same cloudscraper approach that successfully works for AniSearch.
    
    Args:
        url: Full Simkl URL (e.g., "https://simkl.com/anime/2308514")
        
    Returns:
        Dict with title, relationship, confidence, and original URL
        None if fetch fails
    """
    if not HAS_CLOUDSCRAPER:
        logger.error("cloudscraper not available - install with: pip install cloudscraper")
        return _create_fallback_result(url, "cloudscraper_not_available")
    
    try:
        # Extract anime ID from URL for context
        anime_id_match = re.search(r'/anime/(\d+)', url)
        anime_id = anime_id_match.group(1) if anime_id_match else "unknown"
        
        # Create cloudscraper instance (same as BaseScraper)
        scraper = cloudscraper.create_scraper(
            browser={"browser": "chrome", "platform": "linux", "desktop": True}
        )
        
        # Rate limiting (conservative for Stage 3 batch processing)
        time.sleep(1.5)
        
        logger.debug(f"Fetching Simkl URL with cloudscraper: {url}")
        
        response = scraper.get(url, timeout=10)
        
        if response.status_code == 200:
            # Check for MODB's dead entry pattern first
            if DEAD_ENTRY_INDICATOR in response.text:
                logger.debug(f"Simkl URL {url} detected as dead entry")
                return {
                    "title": f"Simkl Anime {anime_id}",
                    "relationship": None,  # Cannot determine relationship for dead entry
                    "confidence": "none",
                    "is_dead_entry": True,
                    "url": url,
                    "source": "simkl"
                }
            
            return _parse_simkl_response(response.text, url, anime_id)
        else:
            logger.warning(f"Simkl returned status {response.status_code} for URL {url}")
            return _create_fallback_result(url, anime_id, response.status_code)
            
    except Exception as e:
        logger.error(f"Error fetching Simkl URL {url}: {e}")
        anime_id_match = re.search(r'/anime/(\d+)', url)
        anime_id = anime_id_match.group(1) if anime_id_match else "unknown"
        return _create_fallback_result(url, anime_id)


def _parse_simkl_response(html_content: str, url: str, anime_id: str) -> Dict[str, Any]:
    """Parse Simkl HTML response to extract title and relationship info."""
    try:
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Extract title using multiple strategies
        title = _extract_title_from_page(soup)
        if not title:
            title = f"Simkl Anime {anime_id}"
            confidence = "low"
        else:
            confidence = "high"
        
        # Extract additional context for AI analysis instead of hardcoded inference
        context = _extract_context_data(soup)
        
        return {
            "title": title,
            "relationship": None,  # Let AI infer from context
            "confidence": confidence,
            "context": context,  # Provide rich context for AI inference
            "url": url,
            "source": "simkl"
        }
        
    except Exception as e:
        logger.error(f"Error parsing Simkl response for URL {url}: {e}")
        return _create_fallback_result(url, anime_id)


def _extract_title_from_page(soup: BeautifulSoup) -> Optional[str]:
    """Extract anime title from Simkl page using multiple strategies."""
    
    # Strategy 1: Try OpenGraph title (most reliable)
    og_title = soup.find('meta', property='og:title')
    if og_title and og_title.get('content'):
        title = og_title['content'].strip()
        # Clean Simkl-specific parts
        title = re.sub(r' \| Simkl.*$', '', title)
        if title and not title.lower().startswith('simkl'):
            return title
    
    # Strategy 2: Try h1 heading
    h1_tag = soup.find('h1')
    if h1_tag:
        title = h1_tag.get_text().strip()
        if title and not title.lower().startswith('simkl'):
            return title
    
    # Strategy 3: Try page title
    title_tag = soup.find('title')
    if title_tag:
        title_text = title_tag.get_text().strip()
        # Clean Simkl-specific parts
        title = re.sub(r' \| Simkl.*$', '', title_text)
        if title and title != title_text and not title.lower().startswith('simkl'):
            return title
    
    # Strategy 4: Try JSON-LD structured data
    json_ld_script = soup.find('script', type='application/ld+json')
    if json_ld_script:
        try:
            import json
            data = json.loads(json_ld_script.string)
            if isinstance(data, dict) and 'name' in data:
                name = data['name'].strip()
                if name and not name.lower().startswith('simkl'):
                    return name
        except:
            pass
    
    return None


def _extract_context_data(soup: BeautifulSoup) -> Dict[str, Any]:
    """
    Extract contextual data for AI-powered relationship inference.
    
    Instead of hardcoded regex patterns, extract rich context that AI can analyze.
    """
    context = {}
    
    # Extract description/synopsis for context
    og_desc = soup.find('meta', property='og:description')
    if og_desc and og_desc.get('content'):
        context['description'] = og_desc['content'].strip()
    
    # Extract type/format information
    type_elements = soup.find_all(['span', 'div'], class_=re.compile(r'type|format'))
    for element in type_elements:
        type_text = element.get_text().strip()
        if type_text and len(type_text) < 20:
            context['type'] = type_text
            break
    
    # Extract year information
    date_elements = soup.find_all(['time', 'span'], class_=re.compile(r'date|year'))
    for element in date_elements:
        date_text = element.get_text().strip()
        if re.match(r'\d{4}', date_text):
            context['year'] = date_text
            break
    
    # Extract genre information
    genre_elements = soup.select('.genres a, .genre-tag, .tags a, a[href*="/genre/"]')
    genres = []
    for element in genre_elements:
        genre_text = element.get_text().strip()
        if genre_text and len(genre_text) < 30:
            genres.append(genre_text)
    if genres:
        context['genres'] = genres[:5]  # Limit to 5 most relevant
    
    # Extract any related/similar anime mentions
    related_elements = soup.find_all(['div', 'section'], class_=re.compile(r'related|similar|franchise'))
    if related_elements:
        context['has_related_section'] = True
    
    return context


def _create_fallback_result(url: str, anime_id: str, status_code: Optional[int] = None) -> Dict[str, Any]:
    """Create fallback result when extraction fails."""
    return {
        "title": f"Simkl Anime {anime_id}",
        "relationship": None,  # Cannot determine relationship when blocked
        "confidence": "none",  # No confidence when we can't access data
        "url": url,
        "source": "simkl",
        "status_code": status_code
    }


# Batch processing function for multiple URLs
def fetch_multiple_simkl_urls(urls: list[str]) -> Dict[str, Dict[str, Any]]:
    """
    Fetch data for multiple Simkl URLs with proper rate limiting.
    
    Args:
        urls: List of Simkl URLs
        
    Returns:
        Dict mapping URL to result data
    """
    results = {}
    
    for i, url in enumerate(urls):
        logger.info(f"Processing Simkl URL {i+1}/{len(urls)}: {url}")
        
        result = fetch_simkl_url_data(url)
        if result:
            results[url] = result
        
        # Rate limiting between requests
        if i < len(urls) - 1:  # Don't sleep after the last request
            time.sleep(2.0)  # Conservative rate limiting
    
    logger.info(f"Completed processing {len(results)}/{len(urls)} Simkl URLs")
    return results


if __name__ == "__main__":
    # Test the helper
    import sys
    
    if len(sys.argv) != 2:
        print("Usage: python simkl_url_helper.py <simkl_url>")
        sys.exit(1)
    
    url = sys.argv[1]
    result = fetch_simkl_url_data(url)
    
    if result:
        print(f"Title: {result['title']}")
        print(f"Relationship: {result['relationship']}")
        print(f"Confidence: {result['confidence']}")
        print(f"Dead Entry: {result.get('is_dead_entry', False)}")
        print(f"URL: {result['url']}")
    else:
        print(f"Failed to fetch data for URL: {url}")