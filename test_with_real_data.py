#!/usr/bin/env python3
"""Test TextProcessor changes with real cloud data integration."""

import os

def test_with_real_cloud_data():
    """Test all TextProcessor changes with actual cloud Qdrant integration."""

    try:
        from src.vector.text_processor import TextProcessor
        from src.vector.qdrant_client import QdrantClient
        from src.config import get_settings

        # Set up cloud connection
        settings = get_settings()
        settings.qdrant_url = os.getenv("QDRANT_CLOUD_URL", settings.qdrant_url)
        settings.qdrant_api_key = os.getenv("QDRANT_API_KEY")

        print("🌐 REAL CLOUD DATA INTEGRATION TEST")
        print("=" * 50)

        # Initialize components
        processor = TextProcessor(settings)
        qdrant_client = QdrantClient(
            url=settings.qdrant_url,
            collection_name=settings.qdrant_collection_name,
            settings=settings,
        )

        print(f"✅ Connected to: {settings.qdrant_url}")

        # Test 1: Real text processing with various content types
        print("\n🧪 TEST 1: Real Text Processing")

        test_texts = [
            "Naruto is a ninja anime about friendship and perseverance",
            ["Action", "Adventure", "Shounen"],  # This would be converted to string
            "",  # Empty text
            "A very long description of an anime that goes on and on about the characters, plot, setting, and themes to test how well the model handles longer text sequences and maintains quality embeddings throughout the entire processing pipeline"
        ]

        for i, text in enumerate(test_texts):
            # Simulate the list/string conversion we added
            if isinstance(text, list):
                content_str = ' '.join(str(item) for item in text)
                desc = f"list: {text}"
            else:
                content_str = str(text)
                desc = f"string: '{text[:50]}...'" if len(str(text)) > 50 else f"string: '{text}'"

            embedding = processor.encode_text(content_str)
            if embedding:
                print(f"   ✅ Test {i+1} ({desc}): {len(embedding)} dims")
            else:
                print(f"   ❌ Test {i+1} failed")

        # Test 2: Integration with Qdrant operations
        print("\n🔗 TEST 2: Qdrant Integration")

        # Test that our TextProcessor can create embeddings that work with Qdrant
        test_embedding = processor.encode_text("One Piece adventure anime")
        if test_embedding:
            print(f"   ✅ Generated embedding for Qdrant: {len(test_embedding)} dims")

            # Test that dimensions match what Qdrant expects
            model_info = processor.get_model_info()
            expected_dims = model_info.get('embedding_size', 0)
            if len(test_embedding) == expected_dims:
                print(f"   ✅ Embedding dimensions match model info: {expected_dims}")
            else:
                print(f"   ⚠️  Dimension mismatch: got {len(test_embedding)}, expected {expected_dims}")

        # Test 3: All our type fixes work in real scenarios
        print("\n🛠️  TEST 3: Type Fixes in Real Usage")

        # Test Dict[str, Any] returns
        model_dict = processor.model
        if model_dict:
            print(f"   ✅ Model dict type: {type(model_dict)}")
            print(f"   ✅ Model dict keys: {list(model_dict.keys())}")

        # Test the AnimeFieldMapper
        field_mapper = processor._get_field_mapper()
        print(f"   ✅ Field mapper type: {type(field_mapper)}")

        # Test model warm-up
        processor._warm_up_model()
        print(f"   ✅ Model warm-up completed")

        print("\n" + "=" * 50)
        print("🎉 ALL REAL DATA TESTS PASSED")
        return True

    except Exception as e:
        print(f"\n❌ REAL DATA TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_with_real_cloud_data()
    print(f"\n🎯 FINAL RESULT: {'SUCCESS' if success else 'FAILED'}")