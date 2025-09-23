#!/usr/bin/env python3
"""Genre Enhancement Training CLI - Fix BGE-M3 semantic accuracy from 60% to 90%+ precision.

This script implements domain-specific fine-tuning using the existing infrastructure
in src/vector/enhancement/ to address the critical quality issue where genre vector
search shows poor semantic accuracy due to BGE-M3 false positives.

Key Issues Being Fixed:
- False positives from theme descriptions ("Drama is more serious than humorous" → comedy)
- Semantic drift (entertainment content clustering with comedy)
- 60% precision vs 90%+ industry standard

Solution: LoRA fine-tuning with enriched JSON as ground truth.
"""

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Dict, Any, Optional

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.config.settings import Settings
from src.vector.enhancement.anime_fine_tuning import AnimeFineTuner, FineTuningConfig
from src.vector.enhancement.anime_dataset import AnimeDataset


def setup_logging(debug: bool = False) -> None:
    """Setup logging configuration.

    Args:
        debug: Enable debug logging
    """
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("genre_enhancement_training.log")
        ]
    )


def validate_data_file(data_path: Path) -> bool:
    """Validate the enriched anime database file format.

    Args:
        data_path: Path to enriched anime database JSON

    Returns:
        True if valid format, False otherwise
    """
    try:
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Check expected structure: {data: [...], enrichmentInfo: {...}}
        if not isinstance(data, dict):
            print(f"❌ Data file must be a JSON object, got {type(data)}")
            return False

        if 'data' not in data:
            print("❌ Missing 'data' key in JSON structure")
            return False

        if not isinstance(data['data'], list):
            print(f"❌ 'data' must be a list, got {type(data['data'])}")
            return False

        anime_count = len(data['data'])
        print(f"✅ Found {anime_count} anime entries in database")

        # Sample first anime to check structure
        if anime_count > 0:
            sample = data['data'][0]
            required_fields = ['genres', 'tags', 'themes']
            for field in required_fields:
                if field in sample:
                    print(f"✅ Sample anime has '{field}' field: {type(sample[field])}")
                else:
                    print(f"⚠️ Sample anime missing '{field}' field")

        return True

    except FileNotFoundError:
        print(f"❌ Data file not found: {data_path}")
        return False
    except json.JSONDecodeError as e:
        print(f"❌ Invalid JSON format: {e}")
        return False
    except Exception as e:
        print(f"❌ Error validating data file: {e}")
        return False


def create_training_config(args: argparse.Namespace) -> FineTuningConfig:
    """Create training configuration from command line arguments.

    Args:
        args: Parsed command line arguments

    Returns:
        Training configuration
    """
    config = FineTuningConfig()

    # Update configuration based on arguments
    if hasattr(args, 'batch_size'):
        config.batch_size = args.batch_size
    if hasattr(args, 'learning_rate'):
        config.learning_rate = args.learning_rate
    if hasattr(args, 'num_epochs'):
        config.num_epochs = args.num_epochs
    if hasattr(args, 'lora_r'):
        config.lora_r = args.lora_r
    if hasattr(args, 'lora_alpha'):
        config.lora_alpha = args.lora_alpha
    if hasattr(args, 'lora_dropout'):
        config.lora_dropout = args.lora_dropout
    if hasattr(args, 'output_dir'):
        config.model_output_dir = args.output_dir

    # Focus entirely on genre enhancement (disable other tasks)
    config.genre_weight = 1.0  # Focus entirely on genre enhancement
    config.character_weight = 0.0  # Disable other tasks for focused training
    config.art_style_weight = 0.0

    return config


async def run_training(
    data_path: Path,
    config: FineTuningConfig,
    settings: Settings
) -> Dict[str, Any]:
    """Run the genre enhancement training.

    Args:
        data_path: Path to training data
        config: Training configuration
        settings: Application settings

    Returns:
        Training results and statistics
    """
    logger = logging.getLogger(__name__)
    logger.info("🚀 Starting Genre Enhancement Training")
    logger.info(f"📁 Data source: {data_path}")
    logger.info(f"🎯 Target: Improve genre precision from 60% to 90%+")

    # Initialize fine-tuner
    fine_tuner = AnimeFineTuner(settings)
    fine_tuner.config = config

    # Prepare dataset
    logger.info("📊 Preparing training dataset...")
    dataset = fine_tuner.prepare_dataset(str(data_path))
    if dataset is None:
        raise RuntimeError("Failed to prepare dataset")

    logger.info(f"✅ Dataset prepared with {len(dataset)} samples")

    # Run training
    logger.info("🔄 Starting training loop...")
    training_stats = fine_tuner.train_multi_task(dataset)

    # Get training summary
    summary = fine_tuner.get_training_summary()

    logger.info("✅ Training completed successfully!")
    logger.info(f"📈 Best loss: {training_stats['best_loss']:.4f} at epoch {training_stats['best_epoch']}")

    return summary


def main():
    """Main entry point for genre enhancement training."""
    parser = argparse.ArgumentParser(
        description="Train genre enhancement to fix BGE-M3 semantic accuracy",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic training with default settings
  python train_genre_enhancement.py

  # Custom training configuration
  python train_genre_enhancement.py --epochs 5 --batch-size 32 --learning-rate 2e-4

  # Focus on specific data file
  python train_genre_enhancement.py --data-path data/custom_anime_data.json

  # High-performance training
  python train_genre_enhancement.py --lora-r 16 --lora-alpha 32 --epochs 10
        """
    )

    # Data configuration
    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path("data/qdrant_storage/enriched_anime_database.json"),
        help="Path to enriched anime database JSON (default: enriched_anime_database.json)"
    )

    # Training hyperparameters
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Training batch size (default: 16)"
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-4,
        help="Learning rate (default: 1e-4)"
    )

    parser.add_argument(
        "--num-epochs",
        type=int,
        default=15,
        help="Number of training epochs (default: 15 for 90%+ accuracy)"
    )

    # LoRA configuration
    parser.add_argument(
        "--lora-r",
        type=int,
        default=8,
        help="LoRA rank parameter (default: 8)"
    )

    parser.add_argument(
        "--lora-alpha",
        type=int,
        default=32,
        help="LoRA alpha parameter (default: 32)"
    )

    parser.add_argument(
        "--lora-dropout",
        type=float,
        default=0.1,
        help="LoRA dropout rate (default: 0.1)"
    )

    # Output configuration
    parser.add_argument(
        "--output-dir",
        type=str,
        default="models/genre_enhanced",
        help="Output directory for trained models (default: models/genre_enhanced)"
    )

    # System configuration
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging"
    )

    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Only validate data format, don't train"
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.debug)
    logger = logging.getLogger(__name__)

    print("🎯 GENRE ENHANCEMENT TRAINING")
    print("=" * 50)
    print("Fix BGE-M3 semantic accuracy from 60% to 90%+ precision")
    print("Addressing false positives and semantic drift issues")
    print()

    # Validate data file
    print("📊 VALIDATING DATA SOURCE:")
    if not validate_data_file(args.data_path):
        print("❌ Data validation failed. Please check your data file.")
        sys.exit(1)

    if args.validate_only:
        print("✅ Data validation successful. Exiting (--validate-only mode).")
        sys.exit(0)

    print()
    print("⚙️ TRAINING CONFIGURATION:")
    print(f"  Data path: {args.data_path}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.learning_rate}")
    print(f"  Epochs: {args.num_epochs}")
    print(f"  LoRA r: {args.lora_r}, alpha: {args.lora_alpha}, dropout: {args.lora_dropout}")
    print(f"  Output: {args.output_dir}")
    print()

    try:
        # Load settings
        settings = Settings()

        # Create training configuration
        config = create_training_config(args)

        # Run training
        print("🚀 STARTING TRAINING:")
        results = asyncio.run(run_training(args.data_path, config, settings))

        print()
        print("✅ TRAINING COMPLETED SUCCESSFULLY!")
        print("=" * 50)
        print(f"📈 Best loss: {results['training_stats']['best_loss']:.4f}")
        print(f"🏆 Best epoch: {results['training_stats']['best_epoch']}")
        print(f"💾 Models saved to: {results['best_model_path']}")
        print()
        print("🎯 Next steps:")
        print("  1. Run validation to test improved semantic accuracy")
        print("  2. Integrate enhanced embeddings with QdrantClient")
        print("  3. Deploy enhanced model for production use")

    except KeyboardInterrupt:
        print("\n⚠️ Training interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Training failed: {e}")
        print(f"\n❌ Training failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()