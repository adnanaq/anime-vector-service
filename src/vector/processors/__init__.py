"""Core Embedding Processors

Text, vision, and multi-vector embedding generation with field mapping.
"""

from .text_processor import TextProcessor
from .vision_processor import VisionProcessor
from .embedding_manager import MultiVectorEmbeddingManager
from .anime_field_mapper import AnimeFieldMapper

__all__ = [
    "TextProcessor",
    "VisionProcessor",
    "MultiVectorEmbeddingManager",
    "AnimeFieldMapper",
]