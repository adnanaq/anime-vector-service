"""Data augmentation strategies for genre enhancement training."""

import random
import re
from typing import List, Dict, Tuple, Optional
import torch
import torch.nn.functional as F
import numpy as np


class GenreDataAugmentation:
    """Data augmentation for genre training to achieve 90%+ accuracy."""

    def __init__(self,
                 synonym_prob: float = 0.3,
                 dropout_prob: float = 0.1,
                 shuffle_prob: float = 0.2,
                 negative_sample_prob: float = 0.25):
        """Initialize data augmentation.

        Args:
            synonym_prob: Probability of synonym replacement
            dropout_prob: Probability of word dropout
            shuffle_prob: Probability of word shuffling
            negative_sample_prob: Probability of negative sampling
        """
        self.synonym_prob = synonym_prob
        self.dropout_prob = dropout_prob
        self.shuffle_prob = shuffle_prob
        self.negative_sample_prob = negative_sample_prob

        # Genre-specific synonyms for better semantic understanding
        self.genre_synonyms = {
            'action': ['fighting', 'combat', 'battle', 'adventure', 'explosive'],
            'comedy': ['funny', 'humorous', 'hilarious', 'comedic', 'amusing'],
            'drama': ['emotional', 'serious', 'dramatic', 'intense', 'moving'],
            'romance': ['love', 'romantic', 'relationship', 'dating', 'affection'],
            'thriller': ['suspense', 'mystery', 'tense', 'gripping', 'exciting'],
            'horror': ['scary', 'frightening', 'terrifying', 'spooky', 'dark'],
            'fantasy': ['magical', 'mystical', 'supernatural', 'enchanted', 'mythical'],
            'science fiction': ['sci-fi', 'futuristic', 'space', 'technology', 'cyberpunk'],
            'sports': ['athletic', 'competition', 'tournament', 'training', 'championship'],
            'slice of life': ['everyday', 'daily life', 'realistic', 'mundane', 'ordinary'],
            'psychological': ['mental', 'mind-bending', 'complex', 'introspective', 'cerebral'],
            'historical': ['period', 'past', 'traditional', 'ancient', 'classical'],
            'shounen': ['young male', 'teenage boy', 'youth', 'coming-of-age'],
            'shoujo': ['young female', 'teenage girl', 'romantic', 'emotional'],
            'seinen': ['adult male', 'mature', 'sophisticated', 'complex'],
            'josei': ['adult female', 'mature woman', 'realistic romance']
        }

        # Negative genre patterns (things that should NOT match certain genres)
        self.negative_patterns = {
            'comedy': [
                'serious drama', 'dark theme', 'tragic story', 'psychological horror',
                'intense violence', 'war documentary', 'political thriller'
            ],
            'romance': [
                'action-packed', 'pure action', 'military combat', 'horror story',
                'political drama', 'war film', 'disaster movie'
            ],
            'horror': [
                'light-hearted', 'comedic relief', 'feel-good', 'uplifting',
                'romantic comedy', 'slice of life', 'children\'s story'
            ],
            'slice of life': [
                'supernatural powers', 'magic system', 'space opera', 'time travel',
                'giant robots', 'epic battles', 'world-ending threat'
            ]
        }

    def augment_text(self, text: str, target_genres: List[str]) -> List[str]:
        """Augment text with multiple variations.

        Args:
            text: Original text
            target_genres: Target genres for the anime

        Returns:
            List of augmented text variations
        """
        augmented_texts = [text]  # Include original

        # Generate 3-5 augmented versions
        for _ in range(random.randint(3, 5)):
            augmented = text

            # Apply random augmentations
            if random.random() < self.synonym_prob:
                augmented = self._replace_synonyms(augmented, target_genres)

            if random.random() < self.dropout_prob:
                augmented = self._word_dropout(augmented)

            if random.random() < self.shuffle_prob:
                augmented = self._shuffle_segments(augmented)

            augmented_texts.append(augmented)

        return augmented_texts

    def generate_negative_samples(self,
                                positive_texts: List[str],
                                positive_genres: List[str]) -> List[Tuple[str, List[str]]]:
        """Generate negative samples to reduce false positives.

        Args:
            positive_texts: Positive training texts
            positive_genres: Positive genres

        Returns:
            List of (negative_text, incorrect_genres) pairs
        """
        negative_samples = []

        for positive_text in positive_texts:
            if random.random() < self.negative_sample_prob:
                # Generate negative samples with wrong genre associations
                for genre in positive_genres:
                    if genre in self.negative_patterns:
                        # Create text that should NOT match this genre
                        negative_text = self._create_genre_negative(positive_text, genre)
                        if negative_text:
                            # Create incorrect genre labels
                            incorrect_genres = [g for g in positive_genres if g != genre]
                            negative_samples.append((negative_text, incorrect_genres))

        return negative_samples

    def _replace_synonyms(self, text: str, target_genres: List[str]) -> str:
        """Replace words with genre-specific synonyms.

        Args:
            text: Input text
            target_genres: Target genres

        Returns:
            Text with synonym replacements
        """
        words = text.split()
        result_words = []

        for word in words:
            word_lower = word.lower().strip('.,!?;:')

            # Check if word relates to any target genre
            replaced = False
            for genre in target_genres:
                if genre in self.genre_synonyms:
                    # If word is in genre context, maybe replace with synonym
                    if (word_lower in self.genre_synonyms[genre] or
                        any(syn in word_lower for syn in self.genre_synonyms[genre])):
                        if random.random() < 0.4:  # 40% chance to replace
                            synonym = random.choice(self.genre_synonyms[genre])
                            result_words.append(synonym)
                            replaced = True
                            break

            if not replaced:
                result_words.append(word)

        return ' '.join(result_words)

    def _word_dropout(self, text: str) -> str:
        """Randomly drop some words to test robustness.

        Args:
            text: Input text

        Returns:
            Text with some words dropped
        """
        words = text.split()
        if len(words) <= 3:
            return text  # Don't drop from short texts

        # Drop 10-20% of words
        keep_prob = 0.8 + random.random() * 0.1
        kept_words = [word for word in words if random.random() < keep_prob]

        # Ensure at least 2 words remain
        if len(kept_words) < 2:
            kept_words = words[:2]

        return ' '.join(kept_words)

    def _shuffle_segments(self, text: str) -> str:
        """Shuffle sentence segments to test order robustness.

        Args:
            text: Input text

        Returns:
            Text with shuffled segments
        """
        # Split by sentence boundaries
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]

        if len(sentences) <= 1:
            # Split by commas instead
            segments = [s.strip() for s in text.split(',') if s.strip()]
            if len(segments) > 1:
                random.shuffle(segments)
                return ', '.join(segments)
            return text

        # Shuffle sentences
        random.shuffle(sentences)
        return '. '.join(sentences) + '.'

    def _create_genre_negative(self, positive_text: str, target_genre: str) -> Optional[str]:
        """Create negative text that should NOT match the target genre.

        Args:
            positive_text: Original positive text
            target_genre: Genre that should NOT match

        Returns:
            Negative text or None
        """
        if target_genre not in self.negative_patterns:
            return None

        # Use negative patterns for this genre
        negative_phrases = self.negative_patterns[target_genre]

        # Replace positive genre indicators with negative ones
        words = positive_text.split()

        # Add negative phrase
        negative_phrase = random.choice(negative_phrases)

        # Insert negative phrase into text
        insert_pos = random.randint(0, len(words))
        words.insert(insert_pos, negative_phrase)

        return ' '.join(words)

    def create_hard_negatives(self,
                            embeddings: torch.Tensor,
                            labels: torch.Tensor,
                            num_negatives: int = 2) -> Tuple[torch.Tensor, torch.Tensor]:
        """Create hard negative examples for contrastive learning.

        Args:
            embeddings: Input embeddings [batch_size, embedding_dim]
            labels: Genre labels [batch_size, num_genres]
            num_negatives: Number of hard negatives per sample

        Returns:
            Augmented embeddings and labels with hard negatives
        """
        batch_size, embedding_dim = embeddings.shape
        num_genres = labels.shape[1]

        # Calculate pairwise similarities
        similarities = torch.matmul(F.normalize(embeddings, dim=1),
                                  F.normalize(embeddings, dim=1).T)

        # Find hard negatives: most similar embeddings with different labels
        hard_negative_embeddings = []
        hard_negative_labels = []

        for i in range(batch_size):
            current_label = labels[i]
            current_embedding = embeddings[i]

            # Find samples with different labels
            label_similarities = torch.matmul(current_label.unsqueeze(0), labels.T)
            different_mask = label_similarities.squeeze(0) < 0.3  # Low label similarity

            if different_mask.sum() == 0:
                continue

            # Among different labels, find most similar embeddings (hard negatives)
            different_similarities = similarities[i][different_mask]
            if len(different_similarities) > 0:
                # Get top hard negatives
                _, top_indices = torch.topk(different_similarities,
                                          min(num_negatives, len(different_similarities)))

                # Map back to original indices
                different_indices = torch.where(different_mask)[0]
                hard_neg_indices = different_indices[top_indices]

                for neg_idx in hard_neg_indices:
                    # Create synthetic hard negative by interpolating
                    alpha = random.uniform(0.3, 0.7)
                    synthetic_embedding = (alpha * current_embedding +
                                         (1 - alpha) * embeddings[neg_idx])

                    # Create negative label (no genres in common)
                    negative_label = torch.zeros_like(current_label)

                    hard_negative_embeddings.append(synthetic_embedding)
                    hard_negative_labels.append(negative_label)

        if hard_negative_embeddings:
            # Combine with original data
            all_embeddings = torch.cat([embeddings] +
                                     [emb.unsqueeze(0) for emb in hard_negative_embeddings])
            all_labels = torch.cat([labels] +
                                 [lbl.unsqueeze(0) for lbl in hard_negative_labels])

            return all_embeddings, all_labels

        return embeddings, labels

    def augment_batch(self,
                     texts: List[str],
                     genres_list: List[List[str]]) -> Tuple[List[str], List[List[str]]]:
        """Augment a batch of training data.

        Args:
            texts: List of text descriptions
            genres_list: List of genre lists for each text

        Returns:
            Augmented texts and corresponding genres
        """
        augmented_texts = []
        augmented_genres = []

        for text, genres in zip(texts, genres_list):
            # Add original
            augmented_texts.append(text)
            augmented_genres.append(genres)

            # Add augmented versions
            aug_texts = self.augment_text(text, genres)
            for aug_text in aug_texts[1:]:  # Skip original
                augmented_texts.append(aug_text)
                augmented_genres.append(genres)

            # Add negative samples
            neg_samples = self.generate_negative_samples([text], genres)
            for neg_text, neg_genres in neg_samples:
                augmented_texts.append(neg_text)
                augmented_genres.append(neg_genres)

        return augmented_texts, augmented_genres