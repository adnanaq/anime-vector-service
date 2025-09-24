#!/usr/bin/env python3
"""Comprehensive Title Vector Validation using src/validation infrastructure."""

import asyncio
import sys
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Any

sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.config.settings import Settings
from src.vector.client.qdrant_client import QdrantClient
from src.validation.vector_field_mapping import get_vector_fields
from qdrant_client.models import NamedVector


class ComprehensiveTitleValidator:
    """Comprehensive validation for title_vector using all field combinations."""

    def __init__(self):
        self.settings = Settings()
        self.client = QdrantClient(settings=self.settings)
        self.title_fields = get_vector_fields("title_vector")
        print(f"🎯 COMPREHENSIVE TITLE VECTOR VALIDATION")
        print(f"Title vector fields: {self.title_fields}")
        print("=" * 70)

    def generate_field_combinations(self) -> List[List[str]]:
        """Generate all possible field combinations for title_vector."""
        combinations_list = []

        # Generate combinations of length 1 to N
        for r in range(1, len(self.title_fields) + 1):
            for combo in combinations(self.title_fields, r):
                combinations_list.append(list(combo))

        return combinations_list

    async def get_sample_anime_data(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get random sample anime data for testing."""
        # First, get total count to enable random sampling
        collection_info = self.client.client.get_collection(self.settings.qdrant_collection_name)
        total_points = collection_info.points_count

        print(f"📊 Database contains {total_points} anime entries")

        if total_points == 0:
            return []

        # Get a larger sample to filter from
        sample_size = min(limit * 3, total_points, 100)
        scroll_response = self.client.client.scroll(
            collection_name=self.settings.qdrant_collection_name,
            limit=sample_size,
            with_payload=True,
        )

        # Filter valid anime and shuffle
        anime_data = []
        for point in scroll_response[0]:
            payload = point.payload if hasattr(point, "payload") else {}
            if payload.get("title"):
                anime_data.append(payload)

        # Randomly sample the requested number
        import random
        random.shuffle(anime_data)
        selected_anime = anime_data[:limit]

        print(f"🎲 Randomly selected {len(selected_anime)} anime from {len(anime_data)} valid entries")

        # Log the selected anime titles at the top
        print(f"\n🎯 SELECTED ANIME FOR TESTING:")
        for i, anime in enumerate(selected_anime, 1):
            title = anime.get("title", "Unknown Title")
            anime_id = anime.get("id", "Unknown ID")
            print(f"   {i}. {title} (ID: {anime_id})")
        print()

        return selected_anime

    def create_combination_queries(self, anime: Dict[str, Any]) -> Dict[str, str]:
        """Create test queries for all field combinations from real anime data."""
        field_combinations = self.generate_field_combinations()
        test_queries = {}

        for combo in field_combinations:
            query_parts = []

            for field in combo:
                field_value = anime.get(field)
                if field_value:
                    if field == "synopsis" and len(str(field_value)) > 100:
                        # Use first 100 chars for synopsis
                        query_parts.append(str(field_value)[:100])
                    elif field == "background" and len(str(field_value)) > 100:
                        # Use first 100 chars for background
                        query_parts.append(str(field_value)[:100])
                    else:
                        query_parts.append(str(field_value))

            if query_parts:
                combo_key = "+".join(combo)
                test_queries[combo_key] = " ".join(query_parts)

        return test_queries

    async def test_title_vector_combination(self, query: str, target_anime_id: str) -> Dict[str, Any]:
        """Test a specific query against title_vector and find target anime."""
        try:
            # Create embedding for query
            text_embedding = self.client.embedding_manager.text_processor.encode_text(query)
            if text_embedding is None:
                return {"error": "Failed to create embedding", "found_rank": None}

            # Search title_vector directly using new query_points method
            results = self.client.client.query_points(
                collection_name=self.client.collection_name,
                query=text_embedding,
                using="title_vector",
                limit=20,
                with_payload=True
            ).points

            # Find target anime in results
            for rank, result in enumerate(results, 1):
                result_id = result.payload.get("id", "")
                if str(result_id) == str(target_anime_id):
                    return {
                        "found_rank": rank,
                        "score": result.score,
                        "success": rank <= 10
                    }

            return {"found_rank": None, "success": False}

        except Exception as e:
            return {"error": str(e), "found_rank": None}

    async def validate_anime_comprehensively(self, anime: Dict[str, Any]) -> Dict[str, Any]:
        """Validate all field combinations for a single anime."""
        anime_id = anime.get("id")
        anime_title = anime.get("title", "Unknown")

        print(f"\n📝 TESTING: {anime_title} (ID: {anime_id})")

        # Generate all combination queries
        combination_queries = self.create_combination_queries(anime)

        if not combination_queries:
            return {
                "anime_id": anime_id,
                "anime_title": anime_title,
                "error": "No valid queries generated"
            }

        print(f"   Generated {len(combination_queries)} field combinations")

        # Test each combination
        results = {}
        successful_combinations = 0

        for combo_name, query in combination_queries.items():
            print(f"   Testing: {combo_name}")

            result = await self.test_title_vector_combination(query, anime_id)
            results[combo_name] = result

            if result.get("success", False):
                successful_combinations += 1
                rank = result.get("found_rank", "N/A")
                print(f"     ✅ Found at rank {rank}")
            else:
                rank = result.get("found_rank", "Not found")
                print(f"     ❌ {rank}")

        success_rate = (successful_combinations / len(combination_queries)) * 100
        print(f"   📊 Success Rate: {successful_combinations}/{len(combination_queries)} = {success_rate:.1f}%")

        return {
            "anime_id": anime_id,
            "anime_title": anime_title,
            "total_combinations": len(combination_queries),
            "successful_combinations": successful_combinations,
            "success_rate": success_rate,
            "detailed_results": results
        }

    async def run_comprehensive_validation(self, sample_size: int = 5) -> Dict[str, Any]:
        """Run comprehensive validation on randomly sampled anime."""
        print(f"🔍 Getting {sample_size} random anime for validation...")

        sample_anime = await self.get_sample_anime_data(sample_size)

        if not sample_anime:
            return {"error": "No anime data found"}

        print(f"📊 Testing {len(sample_anime)} randomly selected anime with comprehensive field combinations")

        all_results = []
        total_success_rate = 0

        for anime in sample_anime:
            result = await self.validate_anime_comprehensively(anime)
            all_results.append(result)
            total_success_rate += result.get("success_rate", 0)

        overall_success_rate = total_success_rate / len(sample_anime)

        print(f"\n🎯 COMPREHENSIVE TITLE VECTOR VALIDATION RESULTS")
        print("=" * 70)
        print(f"📈 OVERALL SUCCESS RATE: {overall_success_rate:.1f}%")
        print(f"📊 Field combinations tested per anime: {len(self.generate_field_combinations())}")
        print(f"🎯 Total tests performed: {len(sample_anime) * len(self.generate_field_combinations())}")

        return {
            "overall_success_rate": overall_success_rate,
            "total_anime_tested": len(sample_anime),
            "field_combinations_per_anime": len(self.generate_field_combinations()),
            "detailed_results": all_results
        }


async def main():
    """Run comprehensive title vector validation."""
    try:
        validator = ComprehensiveTitleValidator()
        results = await validator.run_comprehensive_validation(sample_size=3)

        print(f"\n🎯 FINAL ASSESSMENT:")
        print(f"Title vector achieves {results['overall_success_rate']:.1f}% success across all field combinations")
        print(f"This validates semantic understanding of individual fields vs combined content.")

    except Exception as e:
        print(f"❌ Validation failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())