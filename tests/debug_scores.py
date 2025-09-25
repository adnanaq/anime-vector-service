#!/usr/bin/env python3
"""
Debug script to check if scores are hardcoded or if there's a real issue.
"""

import sys
import json
import asyncio
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.config import get_settings
from src.vector.client.qdrant_client import QdrantClient
from src.vector.processors.text_processor import TextProcessor

def load_anime_database():
    """Load anime database."""
    with open('./data/qdrant_storage/enriched_anime_database.json', 'r') as f:
        return json.load(f)

async def debug_search_scores():
    """Debug actual search scores to see if they vary."""
    print("🔍 Debugging Search Scores")

    settings = get_settings()
    qdrant_client = QdrantClient(settings=settings)
    text_processor = TextProcessor(settings=settings)

    # Load database
    anime_database = load_anime_database()
    anime_data = anime_database.get('data', [])

    if len(anime_data) < 3:
        print("❌ Not enough data for testing")
        return

    # Test different queries against the same anime to see score variation
    test_anime = anime_data[0]  # First anime
    anime_title = test_anime.get('title', 'Unknown')

    print(f"📽️ Testing against: '{anime_title}'")

    # Test queries with different similarity levels
    test_queries = [
        anime_title,  # Exact match - should have high score
        anime_title[:5],  # Partial match - should have lower score
        "completely unrelated random text",  # No match - should have very low score
        anime_title.replace('a', 'z'),  # Slight variation - medium score
    ]

    print("\n🎯 Testing Score Variations:")
    for i, query in enumerate(test_queries, 1):
        print(f"\n   Query {i}: '{query}'")

        # Generate embedding
        embedding = text_processor.encode_text(query)

        if embedding:
            # Search title_vector
            vector_queries = [{"vector_name": "title_vector", "vector_data": embedding}]

            results = await qdrant_client.search_multi_vector(
                vector_queries=vector_queries,
                limit=3,
                fusion_method="rrf"
            )

            print(f"   📊 Results:")
            for j, result in enumerate(results[:3]):
                title = result.get('title', 'Unknown')
                score = result.get('score', 0.0)
                print(f"      {j+1}. {title[:30]} (score: {score:.6f})")

        print()

async def debug_database_content():
    """Check what's actually in the database."""
    print("📊 Checking Database Content")

    settings = get_settings()
    qdrant_client = QdrantClient(settings=settings)

    # Get database stats
    stats = await qdrant_client.get_stats()
    print(f"   Total documents: {stats.get('total_documents', 0)}")
    print(f"   Vector size: {stats.get('vector_size', 0)}")
    print(f"   Collection status: {stats.get('status', 'Unknown')}")

if __name__ == "__main__":
    asyncio.run(debug_database_content())
    asyncio.run(debug_search_scores())