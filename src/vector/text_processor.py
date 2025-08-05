"""Text embedding processor supporting multiple models for anime text search.

Supports FastEmbed, HuggingFace, and BGE models with dynamic model selection
for optimal performance.
"""

import logging
from typing import Dict, List, Optional, Union
from pathlib import Path

import numpy as np

from ..config import Settings

logger = logging.getLogger(__name__)


class TextProcessor:
    """Text embedding processor supporting multiple models."""

    def __init__(self, settings: Optional[Settings] = None):
        """Initialize modern text processor with configuration.
        
        Args:
            settings: Configuration settings instance
        """
        if settings is None:
            from ..config import get_settings
            settings = get_settings()
            
        self.settings = settings
        self.provider = settings.text_embedding_provider
        self.model_name = settings.text_embedding_model
        self.cache_dir = settings.model_cache_dir
        
        # Model instance
        self.model = None
        
        # Model metadata
        self.model_info = {}
        
        # Initialize models
        self._init_models()
    
    def _init_models(self):
        """Initialize text embedding model."""
        try:
            # Initialize model
            self.model = self._create_model(self.provider, self.model_name)
                
            # Warm up model if enabled
            if self.settings.model_warm_up:
                self._warm_up_model()
                
            logger.info(f"Initialized modern text processor with {self.provider} model: {self.model_name}")
            
        except Exception as e:
            logger.error(f"Failed to initialize modern text processor: {e}")
            raise
    
    def _create_model(self, provider: str, model_name: str) -> Dict:
        """Create a model instance based on provider and model name.
        
        Args:
            provider: Model provider (fastembed, huggingface, sentence-transformers)
            model_name: Model name/path
            
        Returns:
            Dictionary containing model instance and metadata
        """
        if provider == "fastembed":
            return self._create_fastembed_model(model_name)
        elif provider == "huggingface":
            return self._create_huggingface_model(model_name)
        elif provider == "sentence-transformers":
            return self._create_sentence_transformers_model(model_name)
        else:
            raise ValueError(f"Unsupported provider: {provider}")
    
    def _create_fastembed_model(self, model_name: str) -> Dict:
        """Create FastEmbed model instance.
        
        Args:
            model_name: FastEmbed model name
            
        Returns:
            Dictionary with FastEmbed model and metadata
        """
        try:
            from fastembed import TextEmbedding
            
            # Initialize FastEmbed model
            init_kwargs = {"model_name": model_name}
            if self.cache_dir:
                init_kwargs["cache_dir"] = self.cache_dir
                
            model = TextEmbedding(**init_kwargs)
            
            # Get model info
            embedding_size = self._get_fastembed_embedding_size(model_name)
            
            return {
                "model": model,
                "provider": "fastembed",
                "model_name": model_name,
                "embedding_size": embedding_size,
                "max_length": 512,  # FastEmbed default
                "batch_size": 256,
                "supports_multilingual": "multilingual" in model_name.lower() or "m3" in model_name.lower()
            }
            
        except ImportError as e:
            logger.error("FastEmbed not installed. Install with: pip install fastembed")
            raise ImportError("FastEmbed dependencies missing") from e
    
    def _create_huggingface_model(self, model_name: str) -> Dict:
        """Create HuggingFace model instance.
        
        Args:
            model_name: HuggingFace model name
            
        Returns:
            Dictionary with HuggingFace model and metadata
        """
        try:
            from transformers import AutoModel, AutoTokenizer
            import torch
            
            # Load model and tokenizer
            model = AutoModel.from_pretrained(model_name, cache_dir=self.cache_dir)
            tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=self.cache_dir)
            
            # Set device
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model.to(device)
            model.eval()
            
            # Get embedding size
            embedding_size = model.config.hidden_size
            
            # Get max length
            max_length = min(tokenizer.model_max_length, 512)
            
            return {
                "model": model,
                "tokenizer": tokenizer,
                "device": device,
                "provider": "huggingface",
                "model_name": model_name,
                "embedding_size": embedding_size,
                "max_length": max_length,
                "batch_size": 32,
                "supports_multilingual": self._is_multilingual_model(model_name)
            }
            
        except ImportError as e:
            logger.error("HuggingFace dependencies not installed. Install with: pip install transformers torch")
            raise ImportError("HuggingFace dependencies missing") from e
    
    def _create_sentence_transformers_model(self, model_name: str) -> Dict:
        """Create Sentence Transformers model instance.
        
        Args:
            model_name: Sentence Transformers model name
            
        Returns:
            Dictionary with Sentence Transformers model and metadata
        """
        try:
            from sentence_transformers import SentenceTransformer
            
            # Load model
            model = SentenceTransformer(model_name, cache_folder=self.cache_dir)
            
            # Get model info
            embedding_size = model.get_sentence_embedding_dimension()
            max_length = model.max_seq_length
            
            return {
                "model": model,
                "provider": "sentence-transformers",
                "model_name": model_name,
                "embedding_size": embedding_size,
                "max_length": max_length,
                "batch_size": 32,
                "supports_multilingual": self._is_multilingual_model(model_name)
            }
            
        except ImportError as e:
            logger.error("Sentence Transformers not installed. Install with: pip install sentence-transformers")
            raise ImportError("Sentence Transformers dependencies missing") from e
    
    def _get_fastembed_embedding_size(self, model_name: str) -> int:
        """Get embedding size for FastEmbed model.
        
        Args:
            model_name: FastEmbed model name
            
        Returns:
            Embedding dimension size
        """
        # Common FastEmbed model dimensions
        model_dimensions = {
            "BAAI/bge-small-en-v1.5": 384,
            "BAAI/bge-base-en-v1.5": 768,
            "BAAI/bge-large-en-v1.5": 1024,
            "BAAI/bge-m3": 1024,
            "sentence-transformers/all-MiniLM-L6-v2": 384,
            "sentence-transformers/all-mpnet-base-v2": 768,
            "intfloat/e5-small-v2": 384,
            "intfloat/e5-base-v2": 768,
            "intfloat/e5-large-v2": 1024,
        }
        
        return model_dimensions.get(model_name, 384)  # Default to 384
    
    def _is_multilingual_model(self, model_name: str) -> bool:
        """Check if model supports multilingual text.
        
        Args:
            model_name: Model name
            
        Returns:
            True if model supports multiple languages
        """
        multilingual_indicators = [
            "multilingual", "m3", "xlm", "xlm-roberta", "mbert", 
            "distilbert-base-multilingual", "jina-embeddings-v2-base-multilingual"
        ]
        
        model_lower = model_name.lower()
        return any(indicator in model_lower for indicator in multilingual_indicators)
    
    def _detect_model_provider(self, model_name: str) -> str:
        """Detect model provider from model name.
        
        Args:
            model_name: Model name or path
            
        Returns:
            Provider name (fastembed, huggingface, sentence-transformers)
        """
        model_lower = model_name.lower()
        
        # FastEmbed common models
        fastembed_models = [
            "baai/bge", "sentence-transformers/all-minilm", 
            "intfloat/e5", "sentence-transformers/all-mpnet"
        ]
        
        if any(model in model_lower for model in fastembed_models):
            return "fastembed"
        elif "sentence-transformers" in model_lower:
            return "sentence-transformers"
        else:
            return "huggingface"
    
    def _warm_up_model(self):
        """Warm up model with dummy data."""
        try:
            dummy_text = "This is a test sentence for model warm-up."
            self._encode_text_with_model(dummy_text, self.model)
            logger.info("Text model warmed up successfully")
            
        except Exception as e:
            logger.warning(f"Text model warm-up failed: {e}")
    
    def encode_text(self, text: str) -> Optional[List[float]]:
        """Encode text to embedding vector.
        
        Args:
            text: Input text string
            
        Returns:
            Embedding vector or None if encoding fails
        """
        try:
            # Handle empty text
            if not text.strip():
                return self._create_zero_vector()
            
            # Encode with model
            embedding = self._encode_text_with_model(text, self.model)
            return embedding
            
        except Exception as e:
            logger.error(f"Text encoding failed: {e}")
            return None
    
    def _encode_text_with_model(self, text: str, model_dict: Dict) -> Optional[List[float]]:
        """Encode text with specific model.
        
        Args:
            text: Input text
            model_dict: Model dictionary with instance and metadata
            
        Returns:
            Embedding vector or None if encoding fails
        """
        try:
            provider = model_dict["provider"]
            
            if provider == "fastembed":
                return self._encode_with_fastembed(text, model_dict)
            elif provider == "huggingface":
                return self._encode_with_huggingface(text, model_dict)
            elif provider == "sentence-transformers":
                return self._encode_with_sentence_transformers(text, model_dict)
            else:
                logger.error(f"Unsupported provider: {provider}")
                return None
                
        except Exception as e:
            logger.error(f"Model encoding failed: {e}")
            return None
    
    def _encode_with_fastembed(self, text: str, model_dict: Dict) -> Optional[List[float]]:
        """Encode text with FastEmbed model.
        
        Args:
            text: Input text
            model_dict: FastEmbed model dictionary
            
        Returns:
            Embedding vector or None if encoding fails
        """
        try:
            model = model_dict["model"]
            
            # Generate embedding
            embeddings = list(model.embed([text]))
            if embeddings:
                return embeddings[0].tolist()
            else:
                return None
                
        except Exception as e:
            logger.error(f"FastEmbed encoding failed: {e}")
            return None
    
    def _encode_with_huggingface(self, text: str, model_dict: Dict) -> Optional[List[float]]:
        """Encode text with HuggingFace model.
        
        Args:
            text: Input text
            model_dict: HuggingFace model dictionary
            
        Returns:
            Embedding vector or None if encoding fails
        """
        try:
            import torch
            
            model = model_dict["model"]
            tokenizer = model_dict["tokenizer"]
            device = model_dict["device"]
            max_length = model_dict["max_length"]
            
            # Tokenize text
            inputs = tokenizer(
                text, 
                return_tensors="pt", 
                truncation=True, 
                padding=True, 
                max_length=max_length
            ).to(device)
            
            # Generate embedding
            with torch.no_grad():
                outputs = model(**inputs)
                # Use CLS token or mean pooling
                if hasattr(outputs, 'last_hidden_state'):
                    embeddings = outputs.last_hidden_state.mean(dim=1)
                else:
                    embeddings = outputs.pooler_output
                    
                # Normalize
                embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
                embedding = embeddings.cpu().numpy().flatten().tolist()
                
            return embedding
            
        except Exception as e:
            logger.error(f"HuggingFace encoding failed: {e}")
            return None
    
    def _encode_with_sentence_transformers(self, text: str, model_dict: Dict) -> Optional[List[float]]:
        """Encode text with Sentence Transformers model.
        
        Args:
            text: Input text
            model_dict: Sentence Transformers model dictionary
            
        Returns:
            Embedding vector or None if encoding fails
        """
        try:
            model = model_dict["model"]
            
            # Generate embedding
            embedding = model.encode(text)
            return embedding.tolist()
            
        except Exception as e:
            logger.error(f"Sentence Transformers encoding failed: {e}")
            return None
    
    def _create_zero_vector(self) -> List[float]:
        """Create zero vector for empty text.
        
        Returns:
            Zero vector with appropriate dimensions
        """
        if self.model:
            size = self.model["embedding_size"]
            return [0.0] * size
        else:
            return [0.0] * 384  # Default size
    
    def encode_texts_batch(self, texts: List[str], batch_size: Optional[int] = None) -> List[Optional[List[float]]]:
        """Encode multiple texts in batches.
        
        Args:
            texts: List of text strings
            batch_size: Batch size (uses model-specific default if None)
            
        Returns:
            List of embedding vectors
        """
        try:
            if batch_size is None:
                batch_size = self.model.get("batch_size", 32)
                
            embeddings = []
            total_texts = len(texts)
            
            logger.info(f"Processing {total_texts} texts in batches of {batch_size}")
            
            for i in range(0, total_texts, batch_size):
                batch = texts[i : i + batch_size]
                batch_embeddings = []
                
                for text in batch:
                    embedding = self.encode_text(text)
                    batch_embeddings.append(embedding)
                    
                embeddings.extend(batch_embeddings)
                
                # Log progress
                processed = min(i + batch_size, total_texts)
                logger.info(f"Processed {processed}/{total_texts} texts")
                
            return embeddings
            
        except Exception as e:
            logger.error(f"Batch text encoding failed: {e}")
            return [None] * len(texts)
    
    def get_model_info(self) -> Dict:
        """Get information about the current model.
        
        Returns:
            Dictionary with model information
        """
        if self.model:
            return {
                "provider": self.model["provider"],
                "model_name": self.model["model_name"],
                "embedding_size": self.model["embedding_size"],
                "max_length": self.model["max_length"],
                "batch_size": self.model["batch_size"],
                "supports_multilingual": self.model["supports_multilingual"]
            }
        else:
            return {"error": "No model loaded"}
    
    def switch_model(self, provider: str, model_name: str) -> bool:
        """Switch to a different model.
        
        Args:
            provider: New model provider
            model_name: New model name
            
        Returns:
            True if switch successful, False otherwise
        """
        try:
            # Create new model
            new_model = self._create_model(provider, model_name)
            
            # Switch to new model
            self.model = new_model
            self.provider = provider
            self.model_name = model_name
            
            logger.info(f"Switched to {provider} model: {model_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to switch model: {e}")
            return False
    
    def get_bge_model_name(self) -> str:
        """Get the appropriate BGE model name based on settings.
        
        Returns:
            BGE model name
        """
        version = self.settings.bge_model_version
        size = self.settings.bge_model_size
        multilingual = self.settings.bge_enable_multilingual
        
        if multilingual or version == "m3":
            return "BAAI/bge-m3"
        elif version == "reranker":
            return f"BAAI/bge-reranker-{size}"
        else:
            return f"BAAI/bge-{size}-en-{version}"
    
    def validate_text(self, text: str) -> bool:
        """Validate if text can be processed.
        
        Args:
            text: Input text string
            
        Returns:
            True if text is valid, False otherwise
        """
        try:
            if not isinstance(text, str):
                return False
                
            # Check length
            if self.model:
                max_length = self.model.get("max_length", 512)
                if len(text) > max_length * 4:  # Rough token estimate
                    return False
                    
            return True
            
        except Exception:
            return False