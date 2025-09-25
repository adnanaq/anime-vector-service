#!/usr/bin/env python3
"""
Debug raw similarity scores without fusion.
"""

import sys
import json
import requests
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.config import get_settings
from src.vector.processors.text_processor import TextProcessor

def load_anime_database():
    """Load anime database."""
    with open('./data/qdrant_storage/enriched_anime_database.json', 'r') as f:
        return json.load(f)

def test_raw_similarity_scores():
    """Test raw Qdrant similarity scores without fusion."""
    print("🔍 Testing Raw Similarity Scores (No Fusion)")

    settings = get_settings()
    text_processor = TextProcessor(settings=settings)

    # Load database
    anime_database = load_anime_database()
    anime_data = anime_database.get('data', [])

    test_anime = anime_data[0]
    anime_title = test_anime.get('title', 'Unknown')

    print(f"📽️ Testing against: '{anime_title}'")

    # Test queries with different similarity levels
    test_queries = [
        anime_title,  # Exact match
        anime_title[:5],  # Partial match
        "completely unrelated random text",  # No match
    ]

    for i, query in enumerate(test_queries, 1):
        print(f"\n   Query {i}: '{query}'")

        # Generate embedding
        embedding = text_processor.encode_text(query)

        if embedding:
            # Direct Qdrant API call (no fusion)
            search_payload = {
                "vector": {
                    "name": "title_vector",
                    "vector": embedding
                },
                "limit": 3,
                "with_payload": True
            }

            response = requests.post(
                f"{settings.qdrant_url}/collections/{settings.qdrant_collection_name}/points/search",
                headers={
                    "api-key": settings.qdrant_api_key,
                    "Content-Type": "application/json"
                },
                json=search_payload
            )

            if response.status_code == 200:
                results = response.json()["result"]
                print(f"   📊 Raw Scores (no fusion):")
                for j, result in enumerate(results[:3]):
                    title = result["payload"]["title"]
                    score = result["score"]
                    print(f"      {j+1}. {title[:30]} (raw score: {score:.6f})")
            else:
                print(f"   ❌ Search failed: {response.status_code}")

if __name__ == "__main__":
    test_raw_similarity_scores()