"""Vision processor supporting multiple embedding models for anime image search.

Supports CLIP, SigLIP, and JinaCLIP v2 models with dynamic model selection
for optimal performance.
"""

import base64
import io
import logging
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path

import numpy as np
from PIL import Image

from ..config import Settings

logger = logging.getLogger(__name__)


class VisionProcessor:
    """Vision processor supporting multiple embedding models."""

    def __init__(self, settings: Optional[Settings] = None):
        """Initialize modern vision processor with configuration.
        
        Args:
            settings: Configuration settings instance
        """
        if settings is None:
            from ..config import get_settings
            settings = get_settings()
            
        self.settings = settings
        self.provider = settings.image_embedding_provider
        self.model_name = settings.image_embedding_model
        self.cache_dir = settings.model_cache_dir
        
        # Model instance
        self.model = None
        
        # Device configuration
        self.device = None
        
        # Model metadata
        self.model_info = {}
        
        # Initialize models
        self._init_models()
    
    def _init_models(self):
        """Initialize vision embedding model."""
        try:
            # Initialize model
            self.model = self._create_model(self.provider, self.model_name)
                
            # Warm up model if enabled
            if self.settings.model_warm_up:
                self._warm_up_model()
                
            logger.info(f"Initialized modern vision processor with {self.provider} model: {self.model_name}")
            
        except Exception as e:
            logger.error(f"Failed to initialize modern vision processor: {e}")
            raise
    
    def _create_model(self, provider: str, model_name: str) -> Dict:
        """Create a model instance based on provider and model name.
        
        Args:
            provider: Model provider (clip, siglip, jinaclip)
            model_name: Model name/path
            
        Returns:
            Dictionary containing model instance and metadata
        """
        if provider == "clip":
            return self._create_clip_model(model_name)
        elif provider == "siglip":
            return self._create_siglip_model(model_name)
        elif provider == "jinaclip":
            return self._create_jinaclip_model(model_name)
        else:
            raise ValueError(f"Unsupported provider: {provider}")
    
    def _create_clip_model(self, model_name: str) -> Dict:
        """Create CLIP model instance.
        
        Args:
            model_name: CLIP model name
            
        Returns:
            Dictionary with CLIP model and metadata
        """
        try:
            import clip
            import torch
            
            # Set device
            device = "cuda" if torch.cuda.is_available() else "cpu"
            
            # Load CLIP model
            model, preprocess = clip.load(model_name, device=device, download_root=self.cache_dir)
            model.eval()
            
            return {
                "model": model,
                "preprocess": preprocess,
                "device": device,
                "provider": "clip",
                "model_name": model_name,
                "embedding_size": 512,
                "input_resolution": 224,
                "supports_text": True,
                "supports_image": True,
                "batch_size": 32
            }
            
        except ImportError as e:
            logger.error("CLIP dependencies not installed. Install with: pip install torch torchvision clip-by-openai")
            raise ImportError("CLIP dependencies missing") from e
    
    def _create_siglip_model(self, model_name: str) -> Dict:
        """Create SigLIP model instance.
        
        Args:
            model_name: SigLIP model name
            
        Returns:
            Dictionary with SigLIP model and metadata
        """
        try:
            from transformers import SiglipModel, SiglipProcessor
            import torch
            
            # Set device
            device = "cuda" if torch.cuda.is_available() else "cpu"
            
            # Load SigLIP model
            model = SiglipModel.from_pretrained(model_name, cache_dir=self.cache_dir)
            processor = SiglipProcessor.from_pretrained(model_name, cache_dir=self.cache_dir)
            
            model.to(device)
            model.eval()
            
            # Get input resolution from settings
            input_resolution = self.settings.siglip_input_resolution
            
            return {
                "model": model,
                "processor": processor,
                "device": device,
                "provider": "siglip",
                "model_name": model_name,
                "embedding_size": model.config.projection_dim,
                "input_resolution": input_resolution,
                "supports_text": True,
                "supports_image": True,
                "batch_size": 16  # SigLIP works well with smaller batches
            }
            
        except ImportError as e:
            logger.error("SigLIP dependencies not installed. Install with: pip install transformers torch")
            raise ImportError("SigLIP dependencies missing") from e
    
    def _create_jinaclip_model(self, model_name: str) -> Dict:
        """Create JinaCLIP model instance.
        
        Args:
            model_name: JinaCLIP model name
            
        Returns:
            Dictionary with JinaCLIP model and metadata
        """
        try:
            from transformers import AutoModel, AutoProcessor
            import torch
            
            # Set device
            device = "cuda" if torch.cuda.is_available() else "cpu"
            
            # Load JinaCLIP model
            model = AutoModel.from_pretrained(model_name, trust_remote_code=True, cache_dir=self.cache_dir)
            processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True, cache_dir=self.cache_dir)
            
            model.to(device)
            model.eval()
            
            # Get input resolution from settings
            input_resolution = self.settings.jinaclip_input_resolution
            
            return {
                "model": model,
                "processor": processor,
                "device": device,
                "provider": "jinaclip",
                "model_name": model_name,
                "embedding_size": 768,  # JinaCLIP v2 embedding size
                "input_resolution": input_resolution,
                "supports_text": True,
                "supports_image": True,
                "batch_size": 8  # JinaCLIP v2 works with smaller batches due to higher resolution
            }
            
        except ImportError as e:
            logger.error("JinaCLIP dependencies not installed. Install with: pip install transformers torch")
            raise ImportError("JinaCLIP dependencies missing") from e
    
    def _detect_model_provider(self, model_name: str) -> str:
        """Detect model provider from model name.
        
        Args:
            model_name: Model name or path
            
        Returns:
            Provider name (clip, siglip, jinaclip)
        """
        model_lower = model_name.lower()
        
        if "siglip" in model_lower or "google/siglip" in model_lower:
            return "siglip"
        elif "jina" in model_lower or "jinaai" in model_lower:
            return "jinaclip"
        elif "vit" in model_lower or "rn" in model_lower:
            return "clip"
        else:
            # Default to CLIP for unknown models
            return "clip"
    
    def _warm_up_model(self):
        """Warm up model with dummy data."""
        try:
            # Create dummy image
            dummy_image = Image.new('RGB', (224, 224), color='red')
            dummy_image_b64 = self._pil_to_base64(dummy_image)
            
            # Warm up model
            self._encode_image_with_model(dummy_image_b64, self.model)
            logger.info("Model warmed up successfully")
            
        except Exception as e:
            logger.warning(f"Model warm-up failed: {e}")
    
    def _pil_to_base64(self, image: Image.Image) -> str:
        """Convert PIL image to base64 string.
        
        Args:
            image: PIL Image
            
        Returns:
            Base64 encoded image string
        """
        buffer = io.BytesIO()
        image.save(buffer, format='JPEG')
        img_str = base64.b64encode(buffer.getvalue()).decode()
        return img_str
    
    def encode_image(self, image_data: str) -> Optional[List[float]]:
        """Encode image to embedding vector.
        
        Args:
            image_data: Base64 encoded image data
            
        Returns:
            Embedding vector or None if encoding fails
        """
        try:
            # Encode with model
            embedding = self._encode_image_with_model(image_data, self.model)
            return embedding
            
        except Exception as e:
            logger.error(f"Image encoding failed: {e}")
            return None
    
    def _encode_image_with_model(self, image_data: str, model_dict: Dict) -> Optional[List[float]]:
        """Encode image with specific model.
        
        Args:
            image_data: Base64 encoded image data
            model_dict: Model dictionary with instance and metadata
            
        Returns:
            Embedding vector or None if encoding fails
        """
        try:
            # Decode image
            image = self._decode_base64_image(image_data)
            if image is None:
                return None
            
            # Encode based on provider
            provider = model_dict["provider"]
            
            if provider == "clip":
                return self._encode_with_clip(image, model_dict)
            elif provider == "siglip":
                return self._encode_with_siglip(image, model_dict)
            elif provider == "jinaclip":
                return self._encode_with_jinaclip(image, model_dict)
            else:
                logger.error(f"Unsupported provider: {provider}")
                return None
                
        except Exception as e:
            logger.error(f"Model encoding failed: {e}")
            return None
    
    def _encode_with_clip(self, image: Image.Image, model_dict: Dict) -> Optional[List[float]]:
        """Encode image with CLIP model.
        
        Args:
            image: PIL Image
            model_dict: CLIP model dictionary
            
        Returns:
            Embedding vector or None if encoding fails
        """
        try:
            import torch
            
            model = model_dict["model"]
            preprocess = model_dict["preprocess"]
            device = model_dict["device"]
            
            # Preprocess image
            processed_image = preprocess(image).unsqueeze(0).to(device)
            
            # Generate embedding
            with torch.no_grad():
                image_features = model.encode_image(processed_image)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                embedding = image_features.cpu().numpy().flatten().tolist()
                
            return embedding
            
        except Exception as e:
            logger.error(f"CLIP encoding failed: {e}")
            return None
    
    def _encode_with_siglip(self, image: Image.Image, model_dict: Dict) -> Optional[List[float]]:
        """Encode image with SigLIP model.
        
        Args:
            image: PIL Image
            model_dict: SigLIP model dictionary
            
        Returns:
            Embedding vector or None if encoding fails
        """
        try:
            import torch
            
            model = model_dict["model"]
            processor = model_dict["processor"]
            device = model_dict["device"]
            
            # Process image
            inputs = processor(images=image, return_tensors="pt").to(device)
            
            # Generate embedding
            with torch.no_grad():
                image_features = model.get_image_features(**inputs)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                embedding = image_features.cpu().numpy().flatten().tolist()
                
            return embedding
            
        except Exception as e:
            logger.error(f"SigLIP encoding failed: {e}")
            return None
    
    def _encode_with_jinaclip(self, image: Image.Image, model_dict: Dict) -> Optional[List[float]]:
        """Encode image with JinaCLIP model.
        
        Args:
            image: PIL Image
            model_dict: JinaCLIP model dictionary
            
        Returns:
            Embedding vector or None if encoding fails
        """
        try:
            import torch
            
            model = model_dict["model"]
            processor = model_dict["processor"]
            device = model_dict["device"]
            
            # Process image
            inputs = processor(images=image, return_tensors="pt").to(device)
            
            # Generate embedding
            with torch.no_grad():
                image_features = model.get_image_features(**inputs)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                embedding = image_features.cpu().numpy().flatten().tolist()
                
            return embedding
            
        except Exception as e:
            logger.error(f"JinaCLIP encoding failed: {e}")
            return None
    
    def _decode_base64_image(self, image_data: str) -> Optional[Image.Image]:
        """Decode base64 image data to PIL Image.
        
        Args:
            image_data: Base64 encoded image string
            
        Returns:
            PIL Image object or None if decoding fails
        """
        try:
            # Handle data URL format
            if image_data.startswith("data:"):
                base64_part = image_data.split(",", 1)[1]
            else:
                base64_part = image_data
                
            # Decode base64
            image_bytes = base64.b64decode(base64_part)
            
            # Create PIL Image
            image = Image.open(io.BytesIO(image_bytes))
            
            # Convert to RGB if necessary
            if image.mode != "RGB":
                image = image.convert("RGB")
                
            return image
            
        except Exception as e:
            logger.error(f"Failed to decode base64 image: {e}")
            return None
    
    def encode_images_batch(self, image_data_list: List[str], batch_size: Optional[int] = None) -> List[Optional[List[float]]]:
        """Encode multiple images in batches.
        
        Args:
            image_data_list: List of base64 encoded images
            batch_size: Batch size (uses model-specific default if None)
            
        Returns:
            List of embedding vectors
        """
        try:
            if batch_size is None:
                batch_size = self.model.get("batch_size", 8)
                
            embeddings = []
            total_images = len(image_data_list)
            
            logger.info(f"Processing {total_images} images in batches of {batch_size}")
            
            for i in range(0, total_images, batch_size):
                batch = image_data_list[i : i + batch_size]
                batch_embeddings = []
                
                for image_data in batch:
                    embedding = self.encode_image(image_data)
                    batch_embeddings.append(embedding)
                    
                embeddings.extend(batch_embeddings)
                
                # Log progress
                processed = min(i + batch_size, total_images)
                logger.info(f"Processed {processed}/{total_images} images")
                
            return embeddings
            
        except Exception as e:
            logger.error(f"Batch image encoding failed: {e}")
            return [None] * len(image_data_list)
    
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
                "input_resolution": self.model["input_resolution"],
                "device": self.model["device"],
                "supports_text": self.model["supports_text"],
                "supports_image": self.model["supports_image"],
                "batch_size": self.model["batch_size"]
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
    
    def validate_image_data(self, image_data: str) -> bool:
        """Validate if image data can be processed.
        
        Args:
            image_data: Base64 encoded image data
            
        Returns:
            True if image data is valid, False otherwise
        """
        try:
            image = self._decode_base64_image(image_data)
            return image is not None
        except Exception:
            return False
    
    def get_supported_formats(self) -> List[str]:
        """Get list of supported image formats.
        
        Returns:
            List of supported image format extensions
        """
        return ["jpg", "jpeg", "png", "bmp", "tiff", "webp", "gif"]