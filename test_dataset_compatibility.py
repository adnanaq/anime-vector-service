#!/usr/bin/env python3
"""Test script to verify enriched JSON compatibility with AnimeDataset.

This script tests whether the enriched anime database JSON structure
is compatible with the existing AnimeDataset class.
"""

import json
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.config.settings import Settings
from src.vector.enhancement.anime_dataset import AnimeDataset
from src.vector.enhancement.anime_fine_tuning import FineTuningConfig
from src.vector.processors.text_processor import TextProcessor
from src.vector.processors.vision_processor import VisionProcessor


def test_dataset_compatibility():
    """Test compatibility between enriched JSON and AnimeDataset."""
    print("🧪 Testing enriched JSON compatibility with AnimeDataset")
    print("=" * 60)

    # Load enriched data
    data_path = Path("data/qdrant_storage/enriched_anime_database.json")
    print(f"📁 Loading data from: {data_path}")

    try:
        with open(data_path, 'r', encoding='utf-8') as f:
            enriched_data = json.load(f)

        # Extract anime list from nested structure
        anime_list = enriched_data['data']  # The actual anime array
        print(f"✅ Successfully loaded {len(anime_list)} anime entries")

        # Initialize processors
        print("⚙️ Initializing processors...")
        settings = Settings()
        text_processor = TextProcessor(settings)
        vision_processor = VisionProcessor(settings)
        config = FineTuningConfig()

        print("✅ Processors initialized successfully")

        # Test dataset creation
        print("📊 Testing AnimeDataset creation...")
        try:
            dataset = AnimeDataset(
                anime_data=anime_list,  # Pass the anime list directly
                text_processor=text_processor,
                vision_processor=vision_processor,
                config=config,
                augment_data=False  # Disable augmentation for testing
            )

            print(f"✅ Dataset created successfully!")
            print(f"📈 Dataset size: {len(dataset)} samples")
            print(f"🎭 Character vocabulary: {len(dataset.character_vocab)} entries")
            print(f"🎨 Art style vocabulary: {len(dataset.art_style_vocab)} entries")
            print(f"🎯 Genre vocabulary: {len(dataset.genre_vocab)} entries")

            # Test sample access
            print("\n🔍 Testing sample access...")
            if len(dataset) > 0:
                sample = dataset[0]
                print(f"✅ Sample 0 structure:")
                print(f"  📝 Text length: {len(sample['text'])}")
                print(f"  🏷️ Labels available: {list(sample.keys())}")
                print(f"  🎯 Genre labels: {sample.get('genre_labels', [])}")

                # Show sample text preview
                text_preview = sample['text'][:200] + "..." if len(sample['text']) > 200 else sample['text']
                print(f"  📄 Text preview: {text_preview}")

            return True

        except Exception as e:
            print(f"❌ Dataset creation failed: {e}")
            import traceback
            traceback.print_exc()
            return False

    except Exception as e:
        print(f"❌ Data loading failed: {e}")
        return False


def check_genre_data_structure():
    """Check the specific structure of genre-related data."""
    print("\n🎯 ANALYZING GENRE DATA STRUCTURE")
    print("=" * 60)

    data_path = Path("data/qdrant_storage/enriched_anime_database.json")

    with open(data_path, 'r', encoding='utf-8') as f:
        enriched_data = json.load(f)

    anime_list = enriched_data['data']

    # Analyze first few anime for genre structure
    for i, anime in enumerate(anime_list[:3]):
        print(f"\n📺 Anime {i+1}: {anime.get('title', 'Unknown')}")

        # Check genre fields
        genres = anime.get('genres', [])
        tags = anime.get('tags', [])
        themes = anime.get('themes', [])

        print(f"  🎭 Genres ({len(genres)}): {genres}")
        print(f"  🏷️ Tags ({len(tags)}): {tags}")
        print(f"  💭 Themes ({len(themes)} objects):")

        for theme in themes[:2]:  # Show first 2 themes
            if isinstance(theme, dict):
                print(f"    - {theme.get('name', 'Unknown')}: {theme.get('description', 'No description')[:100]}...")

        # Check for content demographics, content_warnings if they exist
        demographics = anime.get('demographics', [])
        content_warnings = anime.get('content_warnings', [])

        if demographics:
            print(f"  👥 Demographics: {demographics}")
        if content_warnings:
            print(f"  ⚠️ Content Warnings: {content_warnings}")


if __name__ == "__main__":
    success = test_dataset_compatibility()

    if success:
        print("\n🎉 COMPATIBILITY TEST PASSED!")
        print("The enriched JSON is compatible with AnimeDataset")

        check_genre_data_structure()

        print("\n✅ Ready for genre enhancement training!")
    else:
        print("\n❌ COMPATIBILITY TEST FAILED!")
        print("Need to fix compatibility issues before training")
        sys.exit(1)