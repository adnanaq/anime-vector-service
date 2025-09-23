#!/usr/bin/env python3
"""Fixed Semantic Genre Vector Validation - Tests using proper filtering approach."""

import asyncio
import sys
from pathlib import Path
from typing import Dict, List, Any, Set

sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.config.settings import Settings
from src.vector.client.qdrant_client import QdrantClient
from src.validation.vector_field_mapping import get_vector_fields
from qdrant_client.models import Filter, FieldCondition, MatchAny, MatchValue


class FixedSemanticGenreValidator:
    """Fixed semantic validation using proper filtering approach."""

    def __init__(self):
        self.settings = Settings()
        self.client = QdrantClient(settings=self.settings)
        self.genre_fields = get_vector_fields("genre_vector")
        print(f"🔧 FIXED SEMANTIC GENRE VECTOR VALIDATION")
        print(f"Testing using filter-based ground truth comparison")
        print("=" * 70)

    async def get_ground_truth_for_concept(self, concept: str, field_type: str) -> List[Dict[str, Any]]:
        """Get ground truth anime that actually contain the concept using filtering."""
        try:
            # Create filter condition based on field type
            if field_type in ["genres", "tags"]:
                # For list fields, use MatchAny
                filter_condition = Filter(
                    must=[
                        FieldCondition(
                            key=field_type,
                            match=MatchAny(any=[concept, concept.capitalize(), concept.upper(), concept.lower()])
                        )
                    ]
                )
            elif field_type in ["demographics", "content_warnings"]:
                # For list fields that might be strings
                filter_condition = Filter(
                    must=[
                        FieldCondition(
                            key=field_type,
                            match=MatchAny(any=[concept, concept.capitalize(), concept.upper(), concept.lower()])
                        )
                    ]
                )
            else:
                # For themes or other complex fields, we'll use scroll and check manually
                filter_condition = None

            if filter_condition:
                # Use filter-based search to get ground truth
                scroll_response = self.client.client.scroll(
                    collection_name=self.settings.qdrant_collection_name,
                    scroll_filter=filter_condition,
                    with_payload=True,
                    limit=100
                )

                ground_truth = []
                for point in scroll_response[0]:
                    if hasattr(point, "payload") and point.payload:
                        ground_truth.append({
                            "id": point.payload.get("id"),
                            "title": point.payload.get("title"),
                            "payload": point.payload
                        })

                return ground_truth
            else:
                # Manual filtering for complex fields like themes
                return await self._manual_filter_concept(concept, field_type)

        except Exception as e:
            print(f"    ⚠️ Filter search failed for {concept}, using manual approach: {e}")
            return await self._manual_filter_concept(concept, field_type)

    async def _manual_filter_concept(self, concept: str, field_type: str) -> List[Dict[str, Any]]:
        """Manually filter concept when automatic filtering fails."""
        # Get all anime and manually check
        scroll_response = self.client.client.scroll(
            collection_name=self.settings.qdrant_collection_name,
            with_payload=True,
            limit=100
        )

        ground_truth = []
        concept_lower = concept.lower()

        for point in scroll_response[0]:
            if not (hasattr(point, "payload") and point.payload):
                continue

            field_value = point.payload.get(field_type, [])
            has_concept = False

            if isinstance(field_value, list):
                for item in field_value:
                    if isinstance(item, dict) and item.get('name'):
                        # Handle theme objects
                        if concept_lower in item['name'].lower():
                            has_concept = True
                            break
                    elif isinstance(item, str):
                        if concept_lower in item.lower():
                            has_concept = True
                            break
            elif isinstance(field_value, str):
                if concept_lower in field_value.lower():
                    has_concept = True

            if has_concept:
                ground_truth.append({
                    "id": point.payload.get("id"),
                    "title": point.payload.get("title"),
                    "payload": point.payload
                })

        return ground_truth

    async def test_concept_with_ground_truth(self, concept: str, field_type: str) -> Dict[str, Any]:
        """Test a concept by comparing vector search results with ground truth filtering."""
        try:
            # Step 1: Get ground truth using filtering
            ground_truth = await self.get_ground_truth_for_concept(concept, field_type)

            if not ground_truth:
                return {
                    "concept": concept,
                    "field_type": field_type,
                    "ground_truth_count": 0,
                    "precision": 0.0,
                    "recall": 0.0,
                    "message": "No anime found with this concept"
                }

            ground_truth_ids = {item["id"] for item in ground_truth}

            print(f"    Ground truth: {len(ground_truth)} anime have '{concept}' in {field_type}")
            if len(ground_truth) <= 3:
                titles = [item["title"] for item in ground_truth]
                print(f"      Examples: {', '.join(titles)}")

            # Step 2: Search using genre vector (limit to ground truth count + small buffer)
            search_limit = max(len(ground_truth), 5)  # At least 5, but prefer ground truth count

            text_embedding = self.client.embedding_manager.text_processor.encode_text(concept)
            if text_embedding is None:
                return {"error": "Failed to create embedding"}

            vector_results = self.client.client.query_points(
                collection_name=self.client.collection_name,
                query=text_embedding,
                using="genre_vector",
                limit=search_limit,
                with_payload=True
            ).points

            # Step 3: Compare results
            vector_ids = set()
            for result in vector_results:
                if result.payload and result.payload.get("id"):
                    vector_ids.add(result.payload["id"])

            # Calculate precision and recall
            correct_matches = ground_truth_ids & vector_ids
            precision = len(correct_matches) / len(vector_results) if vector_results else 0.0
            recall = len(correct_matches) / len(ground_truth) if ground_truth else 0.0

            print(f"    Vector search: {len(vector_results)} results")
            print(f"    Correct matches: {len(correct_matches)}")
            print(f"    Precision: {precision:.1%} | Recall: {recall:.1%}")

            # Show what vector search returned
            if len(vector_results) <= 5:
                print(f"    Vector results:")
                for i, result in enumerate(vector_results, 1):
                    if result.payload:
                        title = result.payload.get("title", "Unknown")
                        is_correct = "✅" if result.payload.get("id") in ground_truth_ids else "❌"
                        print(f"      {i}. {is_correct} {title}")

            return {
                "concept": concept,
                "field_type": field_type,
                "ground_truth_count": len(ground_truth),
                "vector_results_count": len(vector_results),
                "correct_matches": len(correct_matches),
                "precision": precision * 100,  # Convert to percentage
                "recall": recall * 100,
                "f1_score": (2 * precision * recall) / (precision + recall) * 100 if (precision + recall) > 0 else 0.0
            }

        except Exception as e:
            return {"error": str(e)}

    async def extract_testable_concepts(self) -> Dict[str, List[str]]:
        """Extract concepts that actually exist in the database for testing."""
        # Get all anime data
        scroll_response = self.client.client.scroll(
            collection_name=self.settings.qdrant_collection_name,
            with_payload=True,
            limit=100
        )

        concept_counts = {
            "genres": {},
            "tags": {},
            "themes": {},
            "demographics": {},
            "content_warnings": {}
        }

        # Count occurrences of each concept
        for point in scroll_response[0]:
            if not (hasattr(point, "payload") and point.payload):
                continue

            for field in concept_counts.keys():
                field_value = point.payload.get(field, [])

                if isinstance(field_value, list):
                    for item in field_value:
                        if isinstance(item, dict) and item.get('name'):
                            concept = item['name'].lower().strip()
                            concept_counts[field][concept] = concept_counts[field].get(concept, 0) + 1
                        elif isinstance(item, str) and item.strip():
                            concept = item.lower().strip()
                            concept_counts[field][concept] = concept_counts[field].get(concept, 0) + 1

        # Select concepts with reasonable representation (at least 2 anime)
        testable_concepts = {}
        for field, concepts in concept_counts.items():
            # Sort by frequency and take top concepts with at least 2 occurrences
            sorted_concepts = sorted(concepts.items(), key=lambda x: x[1], reverse=True)
            testable_concepts[field] = [
                concept for concept, count in sorted_concepts
                if count >= 2  # At least 2 anime have this concept
            ][:5]  # Take top 5 per field

        return testable_concepts

    async def run_fixed_validation(self) -> Dict[str, Any]:
        """Run the fixed semantic validation."""
        print(f"🔍 Extracting testable concepts from database...")

        testable_concepts = await self.extract_testable_concepts()

        print(f"\n🎯 TESTABLE CONCEPTS (with ≥2 anime):")
        total_tests = 0
        for field, concepts in testable_concepts.items():
            if concepts:
                print(f"  {field}: {len(concepts)} concepts")
                print(f"    Examples: {', '.join(concepts[:3])}")
                total_tests += len(concepts)
            else:
                print(f"  {field}: No testable concepts")

        if total_tests == 0:
            return {"error": "No testable concepts found"}

        print(f"\n🧪 TESTING SEMANTIC ACCURACY WITH GROUND TRUTH:")

        all_results = []
        field_scores = {}

        for field, concepts in testable_concepts.items():
            if not concepts:
                continue

            print(f"\n  {field.upper()} CONCEPTS:")
            field_results = []

            for concept in concepts:
                print(f"    Testing: '{concept}'")

                result = await self.test_concept_with_ground_truth(concept, field)

                if "error" in result:
                    print(f"      ❌ Error: {result['error']}")
                    continue

                field_results.append(result)
                all_results.append(result)

                # Show summary
                precision = result['precision']
                recall = result['recall']
                f1 = result['f1_score']

                if precision >= 80 and recall >= 80:
                    print(f"      ✅ Excellent: P={precision:.1f}% R={recall:.1f}% F1={f1:.1f}%")
                elif precision >= 60 or recall >= 60:
                    print(f"      🟡 Good: P={precision:.1f}% R={recall:.1f}% F1={f1:.1f}%")
                else:
                    print(f"      ❌ Poor: P={precision:.1f}% R={recall:.1f}% F1={f1:.1f}%")

            # Calculate field averages
            if field_results:
                avg_precision = sum(r['precision'] for r in field_results) / len(field_results)
                avg_recall = sum(r['recall'] for r in field_results) / len(field_results)
                avg_f1 = sum(r['f1_score'] for r in field_results) / len(field_results)

                field_scores[field] = {
                    "precision": avg_precision,
                    "recall": avg_recall,
                    "f1_score": avg_f1,
                    "tests": len(field_results)
                }

                print(f"    📊 {field} Average: P={avg_precision:.1f}% R={avg_recall:.1f}% F1={avg_f1:.1f}%")

        # Calculate overall results
        if not all_results:
            return {"error": "No successful tests"}

        overall_precision = sum(r['precision'] for r in all_results) / len(all_results)
        overall_recall = sum(r['recall'] for r in all_results) / len(all_results)
        overall_f1 = sum(r['f1_score'] for r in all_results) / len(all_results)

        print(f"\n🎯 FIXED SEMANTIC GENRE VECTOR VALIDATION RESULTS")
        print("=" * 70)
        print(f"📈 OVERALL PRECISION: {overall_precision:.1f}%")
        print(f"📈 OVERALL RECALL: {overall_recall:.1f}%")
        print(f"📈 OVERALL F1-SCORE: {overall_f1:.1f}%")
        print(f"📊 Total concepts tested: {len(all_results)}")

        print(f"\n📊 FIELD-SPECIFIC PERFORMANCE:")
        for field, scores in field_scores.items():
            precision = scores['precision']
            recall = scores['recall']
            f1 = scores['f1_score']
            tests = scores['tests']

            if f1 >= 80:
                status = "✅ Excellent"
            elif f1 >= 60:
                status = "🟡 Good"
            else:
                status = "❌ Needs improvement"

            print(f"  {field}: P={precision:.1f}% R={recall:.1f}% F1={f1:.1f}% ({tests} tests) {status}")

        return {
            "overall_precision": overall_precision,
            "overall_recall": overall_recall,
            "overall_f1_score": overall_f1,
            "field_scores": field_scores,
            "total_tests": len(all_results),
            "detailed_results": all_results
        }


async def main():
    """Run the fixed semantic validation."""
    try:
        validator = FixedSemanticGenreValidator()
        results = await validator.run_fixed_validation()

        if "error" in results:
            print(f"❌ Validation failed: {results['error']}")
            return

        print(f"\n🎯 FINAL ASSESSMENT:")
        f1_score = results['overall_f1_score']
        precision = results['overall_precision']
        recall = results['overall_recall']

        if f1_score >= 80:
            print(f"🎉 EXCELLENT: Genre vector achieves {f1_score:.1f}% F1-score")
            print(f"Strong semantic understanding with {precision:.1f}% precision and {recall:.1f}% recall")
        elif f1_score >= 60:
            print(f"✅ GOOD: Genre vector achieves {f1_score:.1f}% F1-score")
            print(f"Solid semantic understanding with {precision:.1f}% precision and {recall:.1f}% recall")
        else:
            print(f"⚠️ MIXED RESULTS: Genre vector achieves {f1_score:.1f}% F1-score")
            print(f"Precision: {precision:.1f}%, Recall: {recall:.1f}% - varies by concept type")

    except Exception as e:
        print(f"❌ Validation failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())