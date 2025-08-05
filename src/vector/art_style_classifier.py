"""Art style classifier for anime-specific visual style classification.

This module implements art style classification capabilities through fine-tuning
vision models for anime art style recognition tasks.
"""

import logging
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType
import numpy as np

from ..config import Settings

logger = logging.getLogger(__name__)


class ArtStyleClassificationHead(nn.Module):
    """Classification head for art style classification."""
    
    def __init__(self, input_dim: int, num_styles: int, dropout: float = 0.1):
        """Initialize art style classification head.
        
        Args:
            input_dim: Input embedding dimension
            num_styles: Number of art style classes
            dropout: Dropout probability
        """
        super().__init__()
        
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(input_dim // 2, input_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(input_dim // 4, num_styles)
        )
    
    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Forward pass through classification head.
        
        Args:
            embeddings: Input embeddings
            
        Returns:
            Art style classification logits
        """
        return self.classifier(embeddings)


class ArtStyleFeatureExtractor(nn.Module):
    """Feature extractor for art style-specific features."""
    
    def __init__(self, input_dim: int, feature_dim: int = 256, dropout: float = 0.1):
        """Initialize art style feature extractor.
        
        Args:
            input_dim: Input embedding dimension
            feature_dim: Feature dimension
            dropout: Dropout probability
        """
        super().__init__()
        
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, feature_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(feature_dim * 2, feature_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        
        # Style-specific attention
        self.style_attention = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )
        
        self.layer_norm = nn.LayerNorm(feature_dim)
    
    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Forward pass through feature extractor.
        
        Args:
            embeddings: Input embeddings
            
        Returns:
            Style-specific features
        """
        # Extract features
        features = self.feature_extractor(embeddings)
        
        # Apply self-attention for style-specific features
        features_expanded = features.unsqueeze(1)  # Add sequence dimension
        attended_features, _ = self.style_attention(
            features_expanded, features_expanded, features_expanded
        )
        attended_features = attended_features.squeeze(1)  # Remove sequence dimension
        
        # Layer normalization
        features = self.layer_norm(features + attended_features)
        
        return features


class ArtStyleClassifierModel(nn.Module):
    """Art style classifier model with enhanced visual features."""
    
    def __init__(
        self,
        input_dim: int,
        num_styles: int,
        feature_dim: int = 256,
        dropout: float = 0.1
    ):
        """Initialize art style classifier model.
        
        Args:
            input_dim: Input embedding dimension
            num_styles: Number of art style classes
            feature_dim: Feature dimension
            dropout: Dropout probability
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.num_styles = num_styles
        self.feature_dim = feature_dim
        
        # Feature extractor
        self.feature_extractor = ArtStyleFeatureExtractor(
            input_dim, feature_dim, dropout
        )
        
        # Classification head
        self.classifier = ArtStyleClassificationHead(
            feature_dim, num_styles, dropout
        )
        
        # Auxiliary tasks for better feature learning
        self.studio_classifier = nn.Linear(feature_dim, 50)  # Predict studio
        self.era_classifier = nn.Linear(feature_dim, 5)  # Predict era (vintage, classic, digital, modern, contemporary)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, image_embeddings: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass through art style classifier.
        
        Args:
            image_embeddings: Image embeddings
            
        Returns:
            Art style classification outputs
        """
        # Extract style-specific features
        style_features = self.feature_extractor(image_embeddings)
        style_features = self.dropout(style_features)
        
        # Art style classification
        style_logits = self.classifier(style_features)
        
        # Auxiliary predictions
        studio_logits = self.studio_classifier(style_features)
        era_logits = self.era_classifier(style_features)
        
        return {
            'style_logits': style_logits,
            'studio_logits': studio_logits,
            'era_logits': era_logits,
            'style_features': style_features
        }


class ArtStyleClassifier:
    """Art style classifier for anime visual styles."""
    
    def __init__(self, settings: Settings, vision_processor: Any):
        """Initialize art style classifier.
        
        Args:
            settings: Configuration settings instance
            vision_processor: Vision processing utility
        """
        self.settings = settings
        self.vision_processor = vision_processor
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Model components
        self.classifier_model = None
        self.optimizer = None
        self.loss_fn = None
        
        # Training state
        self.num_styles = 0
        self.style_vocab = {}
        self.studio_vocab = {}
        self.era_vocab = {}
        self.is_trained = False
        
        logger.info(f"Art style classifier initialized on {self.device}")
    
    def setup_lora_model(self, lora_config: LoraConfig, fine_tuning_config: Any):
        """Setup LoRA model for parameter-efficient fine-tuning.
        
        Args:
            lora_config: LoRA configuration
            fine_tuning_config: Fine-tuning configuration
        """
        self.fine_tuning_config = fine_tuning_config
        
        try:
            # Get vision model info
            vision_info = self.vision_processor.get_model_info()
            input_dim = vision_info.get('embedding_size', 512)
            
            # Create art style classifier
            self.classifier_model = ArtStyleClassifierModel(
                input_dim=input_dim,
                num_styles=self.num_styles or 20,  # Default placeholder
                feature_dim=256,
                dropout=0.1
            )
            
            # Move to device
            self.classifier_model = self.classifier_model.to(self.device)
            
            # Setup optimizer
            self.optimizer = torch.optim.AdamW(
                self.classifier_model.parameters(),
                lr=1e-4,
                weight_decay=1e-5
            )
            
            # Setup loss function (multi-task loss)
            self.loss_fn = {
                'style': nn.CrossEntropyLoss(),
                'studio': nn.CrossEntropyLoss(),
                'era': nn.CrossEntropyLoss()
            }
            
            logger.info("LoRA model setup completed")
            
        except Exception as e:
            logger.error(f"Error setting up LoRA model: {e}")
            raise
    
    def prepare_for_training(self, dataset):
        """Prepare model for training with dataset vocabulary.
        
        Args:
            dataset: Training dataset
        """
        vocab_sizes = dataset.get_vocab_sizes()
        self.num_styles = vocab_sizes['art_style']
        self.style_vocab = dataset.art_style_vocab
        
        # Create auxiliary vocabularies
        self._create_auxiliary_vocabularies(dataset)
        
        # Recreate model with correct vocabulary size
        if self.classifier_model is not None:
            input_dim = self.classifier_model.input_dim
            
            # Create new model with correct vocabulary sizes
            self.classifier_model = ArtStyleClassifierModel(
                input_dim=input_dim,
                num_styles=self.num_styles,
                feature_dim=256,
                dropout=0.1
            )
            
            # Update auxiliary classifiers
            self.classifier_model.studio_classifier = nn.Linear(256, len(self.studio_vocab))
            self.classifier_model.era_classifier = nn.Linear(256, len(self.era_vocab))
            
            # Move to device
            self.classifier_model = self.classifier_model.to(self.device)
            
            # Setup optimizer
            self.optimizer = torch.optim.AdamW(
                self.classifier_model.parameters(),
                lr=self.fine_tuning_config.learning_rate,
                weight_decay=self.fine_tuning_config.weight_decay
            )
        
        logger.info(f"Model prepared for training with {self.num_styles} art styles")
    
    def _create_auxiliary_vocabularies(self, dataset):
        """Create auxiliary vocabularies for multi-task learning.
        
        Args:
            dataset: Training dataset
        """
        # Studio vocabulary
        studios = set()
        for sample in dataset.samples:
            if sample.studio:
                studios.add(sample.studio)
        
        self.studio_vocab = {studio: i for i, studio in enumerate(sorted(studios))}
        
        # Era vocabulary based on years
        eras = ['vintage', 'classic', 'digital', 'modern', 'contemporary']
        self.era_vocab = {era: i for i, era in enumerate(eras)}
        
        logger.info(f"Created auxiliary vocabularies: {len(self.studio_vocab)} studios, {len(self.era_vocab)} eras")
    
    def _get_era_from_year(self, year: Optional[int]) -> str:
        """Get era from year.
        
        Args:
            year: Year
            
        Returns:
            Era label
        """
        if year is None:
            return 'unknown'
        
        if year < 1980:
            return 'vintage'
        elif year < 1995:
            return 'classic'
        elif year < 2005:
            return 'digital'
        elif year < 2015:
            return 'modern'
        else:
            return 'contemporary'
    
    def train_step(self, batch: Dict[str, Any]) -> float:
        """Perform one training step.
        
        Args:
            batch: Training batch
            
        Returns:
            Training loss
        """
        if self.classifier_model is None:
            raise RuntimeError("Model not initialized. Call setup_lora_model first.")
        
        self.classifier_model.train()
        self.optimizer.zero_grad()
        
        # Get inputs
        image_embeddings = batch.get('image_embedding')
        if image_embeddings is None:
            # Skip batch if no image embeddings
            return 0.0
        
        image_embeddings = image_embeddings.to(self.device)
        art_style_labels = batch['art_style_label'].to(self.device)
        metadata = batch['metadata']
        
        # Create auxiliary labels
        studio_labels = torch.zeros(len(metadata), dtype=torch.long, device=self.device)
        era_labels = torch.zeros(len(metadata), dtype=torch.long, device=self.device)
        
        for i, meta in enumerate(metadata):
            # Studio label
            if meta.get('studio') and meta['studio'] in self.studio_vocab:
                studio_labels[i] = self.studio_vocab[meta['studio']]
            
            # Era label
            era = self._get_era_from_year(meta.get('year'))
            if era in self.era_vocab:
                era_labels[i] = self.era_vocab[era]
        
        # Forward pass
        outputs = self.classifier_model(image_embeddings)
        
        # Calculate losses
        style_loss = self.loss_fn['style'](outputs['style_logits'], art_style_labels.argmax(dim=1))
        studio_loss = self.loss_fn['studio'](outputs['studio_logits'], studio_labels)
        era_loss = self.loss_fn['era'](outputs['era_logits'], era_labels)
        
        # Combined loss
        total_loss = style_loss + 0.3 * studio_loss + 0.2 * era_loss
        
        # Backward pass
        total_loss.backward()
        self.optimizer.step()
        
        return total_loss.item()
    
    def evaluate(self, dataloader: DataLoader) -> Dict[str, float]:
        """Evaluate model on validation set.
        
        Args:
            dataloader: Validation data loader
            
        Returns:
            Evaluation metrics
        """
        if self.classifier_model is None:
            raise RuntimeError("Model not initialized")
        
        self.classifier_model.eval()
        
        total_loss = 0.0
        correct_predictions = 0
        total_predictions = 0
        
        with torch.no_grad():
            for batch in dataloader:
                # Get inputs
                image_embeddings = batch.get('image_embedding')
                if image_embeddings is None:
                    continue
                
                image_embeddings = image_embeddings.to(self.device)
                art_style_labels = batch['art_style_label'].to(self.device)
                
                # Forward pass
                outputs = self.classifier_model(image_embeddings)
                
                # Calculate loss
                style_loss = self.loss_fn['style'](outputs['style_logits'], art_style_labels.argmax(dim=1))
                total_loss += style_loss.item()
                
                # Calculate accuracy
                predictions = outputs['style_logits'].argmax(dim=1)
                correct_predictions += (predictions == art_style_labels.argmax(dim=1)).sum().item()
                total_predictions += art_style_labels.size(0)
        
        avg_loss = total_loss / len(dataloader)
        accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0.0
        
        return {
            'loss': avg_loss,
            'accuracy': accuracy,
            'total_predictions': total_predictions
        }
    
    def get_enhanced_embedding(self, image_embedding: np.ndarray) -> np.ndarray:
        """Get enhanced art style-aware embedding.
        
        Args:
            image_embedding: Image embedding
            
        Returns:
            Enhanced embedding
        """
        if self.classifier_model is None:
            logger.warning("Model not initialized, returning original embedding")
            return image_embedding
        
        self.classifier_model.eval()
        
        with torch.no_grad():
            # Convert to tensor
            image_tensor = torch.tensor(image_embedding, dtype=torch.float32).unsqueeze(0).to(self.device)
            
            # Get enhanced features
            outputs = self.classifier_model(image_tensor)
            
            # Return style features
            style_features = outputs['style_features']
            return style_features.cpu().numpy().flatten()
    
    def predict_style(self, image_embedding: np.ndarray) -> List[Tuple[str, float]]:
        """Predict art style from image embedding.
        
        Args:
            image_embedding: Image embedding
            
        Returns:
            List of (style, confidence) tuples
        """
        if self.classifier_model is None:
            logger.warning("Model not initialized, returning empty predictions")
            return []
        
        self.classifier_model.eval()
        
        with torch.no_grad():
            # Convert to tensor
            image_tensor = torch.tensor(image_embedding, dtype=torch.float32).unsqueeze(0).to(self.device)
            
            # Get predictions
            outputs = self.classifier_model(image_tensor)
            
            # Get style probabilities
            style_logits = outputs['style_logits']
            style_probs = F.softmax(style_logits, dim=1)
            
            # Get top predictions
            top_probs, top_indices = torch.topk(style_probs, k=min(5, len(self.style_vocab)))
            
            predictions = []
            for prob, idx in zip(top_probs[0], top_indices[0]):
                # Find style name
                style_name = None
                for style, style_idx in self.style_vocab.items():
                    if style_idx == idx.item():
                        style_name = style
                        break
                
                if style_name:
                    predictions.append((style_name, prob.item()))
            
            return predictions
    
    def save_model(self, save_path: Path):
        """Save fine-tuned model.
        
        Args:
            save_path: Path to save model
        """
        if self.classifier_model is None:
            raise RuntimeError("Model not initialized")
        
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save model state
        model_state = {
            'model_state_dict': self.classifier_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'num_styles': self.num_styles,
            'style_vocab': self.style_vocab,
            'studio_vocab': self.studio_vocab,
            'era_vocab': self.era_vocab,
            'is_trained': self.is_trained
        }
        
        torch.save(model_state, save_path / 'art_style_classifier.pth')
        logger.info(f"Art style classifier saved to {save_path}")
    
    def load_model(self, load_path: Path):
        """Load fine-tuned model.
        
        Args:
            load_path: Path to load model from
        """
        model_path = load_path / 'art_style_classifier.pth'
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        # Load model state
        model_state = torch.load(model_path, map_location=self.device)
        
        # Restore configuration
        self.num_styles = model_state['num_styles']
        self.style_vocab = model_state['style_vocab']
        self.studio_vocab = model_state['studio_vocab']
        self.era_vocab = model_state['era_vocab']
        self.is_trained = model_state['is_trained']
        
        # Recreate model with correct configuration
        self.classifier_model = ArtStyleClassifierModel(
            input_dim=512,  # Default dimension
            num_styles=self.num_styles,
            feature_dim=256,
            dropout=0.1
        )
        
        # Update auxiliary classifiers
        self.classifier_model.studio_classifier = nn.Linear(256, len(self.studio_vocab))
        self.classifier_model.era_classifier = nn.Linear(256, len(self.era_vocab))
        
        # Load state dict
        self.classifier_model.load_state_dict(model_state['model_state_dict'])
        self.classifier_model = self.classifier_model.to(self.device)
        
        # Setup optimizer
        self.optimizer = torch.optim.AdamW(
            self.classifier_model.parameters(),
            lr=1e-4,
            weight_decay=1e-5
        )
        self.optimizer.load_state_dict(model_state['optimizer_state_dict'])
        
        logger.info(f"Art style classifier loaded from {load_path}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information.
        
        Returns:
            Model information dictionary
        """
        return {
            'num_styles': self.num_styles,
            'style_vocab_size': len(self.style_vocab),
            'studio_vocab_size': len(self.studio_vocab),
            'era_vocab_size': len(self.era_vocab),
            'is_trained': self.is_trained,
            'device': str(self.device),
            'model_type': 'art_style_classifier'
        }