"""Contrastive learning loss functions for genre enhancement."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import numpy as np


class InfoNCELoss(nn.Module):
    """InfoNCE contrastive loss for genre embeddings."""

    def __init__(self, temperature: float = 0.07):
        """Initialize InfoNCE loss.

        Args:
            temperature: Temperature parameter for softmax
        """
        super().__init__()
        self.temperature = temperature

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute InfoNCE loss.

        Args:
            embeddings: Normalized embeddings [batch_size, embedding_dim]
            labels: Multi-hot genre labels [batch_size, num_genres]

        Returns:
            InfoNCE loss
        """
        batch_size = embeddings.size(0)

        # Compute similarity matrix
        similarity_matrix = torch.matmul(embeddings, embeddings.T) / self.temperature

        # Create positive mask (anime with same genres)
        labels_norm = F.normalize(labels.float(), p=2, dim=1)
        positive_mask = torch.matmul(labels_norm, labels_norm.T) > 0.5

        # Remove self-similarity
        positive_mask = positive_mask.fill_diagonal_(False)

        # Create negative mask
        negative_mask = ~positive_mask
        negative_mask = negative_mask.fill_diagonal_(False)

        # Compute loss
        numerator = torch.exp(similarity_matrix[positive_mask])
        denominator = torch.exp(similarity_matrix[negative_mask]).sum(dim=-1, keepdim=True)

        loss = -torch.log(numerator / (numerator + denominator + 1e-8))

        return loss.mean()


class TripletLoss(nn.Module):
    """Triplet loss with hard negative mining for genre separation."""

    def __init__(self, margin: float = 0.2):
        """Initialize triplet loss.

        Args:
            margin: Margin parameter for triplet loss
        """
        super().__init__()
        self.margin = margin

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute triplet loss with hard negative mining.

        Args:
            embeddings: Normalized embeddings [batch_size, embedding_dim]
            labels: Multi-hot genre labels [batch_size, num_genres]

        Returns:
            Triplet loss
        """
        batch_size = embeddings.size(0)

        # Compute pairwise distances
        distances = torch.cdist(embeddings, embeddings, p=2)

        # Create positive and negative masks
        labels_norm = F.normalize(labels.float(), p=2, dim=1)
        similarity = torch.matmul(labels_norm, labels_norm.T)

        positive_mask = similarity > 0.5
        negative_mask = similarity < 0.3

        # Remove self-similarity
        positive_mask = positive_mask.fill_diagonal_(False)
        negative_mask = negative_mask.fill_diagonal_(False)

        # Hard negative mining: find hardest negatives for each anchor
        losses = []

        for i in range(batch_size):
            # Get positive distances for anchor i
            positive_distances = distances[i][positive_mask[i]]
            if len(positive_distances) == 0:
                continue

            # Get negative distances for anchor i
            negative_distances = distances[i][negative_mask[i]]
            if len(negative_distances) == 0:
                continue

            # Hard positive: furthest positive
            hard_positive = positive_distances.max()

            # Hard negative: closest negative
            hard_negative = negative_distances.min()

            # Triplet loss
            loss = F.relu(hard_positive - hard_negative + self.margin)
            losses.append(loss)

        if not losses:
            return torch.tensor(0.0, device=embeddings.device, requires_grad=True)

        return torch.stack(losses).mean()


class ContrastiveLoss(nn.Module):
    """Contrastive loss for genre similarity learning."""

    def __init__(self, margin: float = 1.0):
        """Initialize contrastive loss.

        Args:
            margin: Margin parameter for negative pairs
        """
        super().__init__()
        self.margin = margin

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute contrastive loss.

        Args:
            embeddings: Embeddings [batch_size, embedding_dim]
            labels: Multi-hot genre labels [batch_size, num_genres]

        Returns:
            Contrastive loss
        """
        batch_size = embeddings.size(0)

        # Compute pairwise distances
        distances = torch.cdist(embeddings, embeddings, p=2)

        # Create positive and negative masks
        labels_norm = F.normalize(labels.float(), p=2, dim=1)
        similarity = torch.matmul(labels_norm, labels_norm.T)

        positive_mask = similarity > 0.5
        negative_mask = similarity < 0.3

        # Remove self-similarity
        positive_mask = positive_mask.fill_diagonal_(False)
        negative_mask = negative_mask.fill_diagonal_(False)

        # Positive loss: minimize distance for similar genres
        positive_loss = (distances * positive_mask.float()).sum() / (positive_mask.sum() + 1e-8)

        # Negative loss: maximize distance for dissimilar genres
        negative_distances = distances * negative_mask.float()
        negative_loss = F.relu(self.margin - negative_distances)
        negative_loss = negative_loss.sum() / (negative_mask.sum() + 1e-8)

        return positive_loss + negative_loss


class SupConLoss(nn.Module):
    """Supervised Contrastive Loss for genre understanding."""

    def __init__(self, temperature: float = 0.07, contrast_mode: str = 'all', base_temperature: float = 0.07):
        """Initialize SupConLoss.

        Args:
            temperature: Temperature parameter
            contrast_mode: Contrast mode ('all' or 'one')
            base_temperature: Base temperature
        """
        super().__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute supervised contrastive loss.

        Args:
            features: Feature embeddings [batch_size, embedding_dim]
            labels: Multi-hot genre labels [batch_size, num_genres]

        Returns:
            Supervised contrastive loss
        """
        device = features.device
        batch_size = features.shape[0]

        # Normalize features
        features = F.normalize(features, dim=1)

        # Compute genre similarity for label matching
        labels_norm = F.normalize(labels.float(), p=2, dim=1)
        label_similarity = torch.matmul(labels_norm, labels_norm.T)

        # Create mask for positive pairs (similar genres)
        mask = label_similarity > 0.5
        mask = mask.float()

        # Remove diagonal
        mask = mask.fill_diagonal_(0)

        # Compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(features, features.T), self.temperature
        )

        # For numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # Create mask to remove self-contrast
        mask_no_diag = torch.ones_like(mask) - torch.eye(batch_size).to(device)
        mask = mask * mask_no_diag

        # Compute log probabilities
        exp_logits = torch.exp(logits) * mask_no_diag
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-8)

        # Compute mean of log-likelihood over positive pairs
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-8)

        # Loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.mean()

        return loss


class GenreContrastiveLearning(nn.Module):
    """Combined contrastive learning module for genre enhancement."""

    def __init__(self,
                 temperature: float = 0.07,
                 margin: float = 0.2,
                 alpha_infonce: float = 0.4,
                 alpha_triplet: float = 0.3,
                 alpha_supcon: float = 0.3):
        """Initialize genre contrastive learning.

        Args:
            temperature: Temperature for InfoNCE and SupCon
            margin: Margin for triplet loss
            alpha_infonce: Weight for InfoNCE loss
            alpha_triplet: Weight for triplet loss
            alpha_supcon: Weight for supervised contrastive loss
        """
        super().__init__()

        self.infonce_loss = InfoNCELoss(temperature)
        self.triplet_loss = TripletLoss(margin)
        self.supcon_loss = SupConLoss(temperature)

        self.alpha_infonce = alpha_infonce
        self.alpha_triplet = alpha_triplet
        self.alpha_supcon = alpha_supcon

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        """Compute combined contrastive loss.

        Args:
            embeddings: Normalized embeddings [batch_size, embedding_dim]
            labels: Multi-hot genre labels [batch_size, num_genres]

        Returns:
            Combined loss and individual loss components
        """
        # Normalize embeddings
        embeddings_norm = F.normalize(embeddings, dim=1)

        # Compute individual losses
        infonce_loss = self.infonce_loss(embeddings_norm, labels)
        triplet_loss = self.triplet_loss(embeddings_norm, labels)
        supcon_loss = self.supcon_loss(embeddings_norm, labels)

        # Combined loss
        total_loss = (
            self.alpha_infonce * infonce_loss +
            self.alpha_triplet * triplet_loss +
            self.alpha_supcon * supcon_loss
        )

        loss_components = {
            'infonce_loss': infonce_loss.item(),
            'triplet_loss': triplet_loss.item(),
            'supcon_loss': supcon_loss.item(),
            'total_contrastive_loss': total_loss.item()
        }

        return total_loss, loss_components