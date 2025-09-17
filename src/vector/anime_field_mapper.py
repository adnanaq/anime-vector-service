# src/vector/anime_field_mapper.py
"""
AnimeFieldMapper - Extract and map anime data fields to 14-vector semantic architecture

Maps anime data from AnimeEntry models to appropriate text/visual embeddings
for each vector type. Implements the comprehensive field mapping strategy
defined in Phase 2.5 architecture with character image semantic separation.
"""

import logging
from typing import Any, Dict, List, Union

from src.models.anime import AnimeEntry

logger = logging.getLogger(__name__)


class AnimeFieldMapper:
    """
    Maps anime data fields to 14-vector semantic architecture.

    Extracts and processes anime data for embedding into:
    - 12 text vectors (BGE-M3, 1024-dim each) for semantic search
    - 2 visual vectors (JinaCLIP v2, 1024-dim each) for image search
      * image_vector: covers, posters, banners, trailer thumbnails
      * character_image_vector: character images for character identification
    """

    def __init__(self) -> None:
        """Initialize the anime field mapper."""
        self.logger = logger

    def map_anime_to_vectors(self, anime: AnimeEntry) -> Dict[str, Union[str, List[str]]]:
        """
        Map complete anime entry to all 14 vectors.

        Args:
            anime: AnimeEntry model with comprehensive anime data

        Returns:
            Dict mapping vector names to their content for embedding
        """
        vector_data: Dict[str, Union[str, List[str]]] = {}

        # Text vectors (12)
        vector_data["title_vector"] = self._extract_title_content(anime)
        vector_data["character_vector"] = self._extract_character_content(anime)
        vector_data["genre_vector"] = self._extract_genre_content(anime)
        vector_data["technical_vector"] = self._extract_technical_content(anime)
        vector_data["staff_vector"] = self._extract_staff_content(anime)
        vector_data["review_vector"] = self._extract_review_content(anime)
        vector_data["temporal_vector"] = self._extract_temporal_content(anime)
        vector_data["streaming_vector"] = self._extract_streaming_content(anime)
        vector_data["related_vector"] = self._extract_related_content(anime)
        vector_data["franchise_vector"] = self._extract_franchise_content(anime)
        vector_data["episode_vector"] = self._extract_episode_content(anime)
        vector_data["identifiers_vector"] = self._extract_identifiers_content(anime)

        # Visual vectors (2)
        vector_data["image_vector"] = self._extract_image_content(anime)
        vector_data["character_image_vector"] = self._extract_character_image_content(anime)

        return vector_data

    # ============================================================================
    # TEXT VECTOR EXTRACTORS (BGE-M3, 1024-dim)
    # ============================================================================

    def _extract_title_content(self, anime: AnimeEntry) -> str:
        """Extract title, synopsis, background, and synonyms for semantic search."""
        content_parts = []

        # Primary titles
        if anime.title:
            content_parts.append(f"Title: {anime.title}")
        if anime.title_english:
            content_parts.append(f"English: {anime.title_english}")
        if anime.title_japanese:
            content_parts.append(f"Japanese: {anime.title_japanese}")

        # Alternative titles
        if anime.synonyms:
            content_parts.append(f"Synonyms: {', '.join(anime.synonyms)}")

        # Descriptive content
        if anime.synopsis:
            content_parts.append(f"Synopsis: {anime.synopsis}")
        if anime.background:
            content_parts.append(f"Background: {anime.background}")

        return " | ".join(content_parts)

    def _extract_character_content(self, anime: AnimeEntry) -> str:
        """Extract character information for semantic character search."""
        content_parts = []

        for char in anime.characters:
            char_info = []

            # Character name and role
            char_info.append(f"Name: {char.name}")
            if char.role:
                char_info.append(f"Role: {char.role}")

            # Name variations
            if char.name_variations:
                char_info.append(f"Variations: {', '.join(char.name_variations)}")
            if char.name_kanji:
                char_info.append(f"Kanji: {char.name_kanji}")
            if char.nicknames:
                char_info.append(f"Nicknames: {', '.join(char.nicknames)}")

            # Character details
            if char.description:
                char_info.append(f"Description: {char.description}")
            if char.age:
                char_info.append(f"Age: {char.age}")
            if char.gender:
                char_info.append(f"Gender: {char.gender}")

            if char_info:
                content_parts.append(" | ".join(char_info))

        return " || ".join(content_parts)

    def _extract_genre_content(self, anime: AnimeEntry) -> str:
        """Extract genres, tags, themes, demographics, and content warnings."""
        content_parts = []

        if anime.genres:
            content_parts.append(f"Genres: {', '.join(anime.genres)}")
        if anime.tags:
            content_parts.append(f"Tags: {', '.join(anime.tags)}")
        if anime.demographics:
            content_parts.append(f"Demographics: {', '.join(anime.demographics)}")
        if anime.content_warnings:
            content_parts.append(
                f"Content Warnings: {', '.join(anime.content_warnings)}"
            )

        # Theme descriptions
        theme_info = []
        for theme in anime.themes:
            if hasattr(theme, "name") and theme.name:
                theme_part = f"Theme: {theme.name}"
                if hasattr(theme, "description") and theme.description:
                    theme_part += f" - {theme.description}"
                theme_info.append(theme_part)
        if theme_info:
            content_parts.append(" | ".join(theme_info))

        return " | ".join(content_parts)

    def _extract_technical_content(self, anime: AnimeEntry) -> str:
        """Extract technical details like rating, status, type, source material."""
        content_parts = []

        # Basic technical info
        content_parts.append(f"Type: {anime.type}")
        content_parts.append(f"Status: {anime.status}")

        if anime.rating:
            content_parts.append(f"Rating: {anime.rating}")
        if anime.source_material:
            content_parts.append(f"Source: {anime.source_material}")
        # NSFW removed - now in payload indexing for filtering

        # Licensing information
        if anime.licensors:
            content_parts.append(f"Licensors: {', '.join(anime.licensors)}")

        return " | ".join(content_parts)

    def _extract_staff_content(self, anime: AnimeEntry) -> str:
        """Extract staff data including directors, composers, studios, voice actors."""
        content_parts = []

        # Extract staff information from staff_data if available
        if hasattr(anime, "staff_data") and anime.staff_data:
            staff_info: List[str] = []

            # Handle different staff_data structures
            if isinstance(anime.staff_data, dict):
                for role, people in anime.staff_data.items():
                    if isinstance(people, list):
                        staff_info.append(f"{role}: {', '.join(people)}")
                    elif isinstance(people, str):
                        staff_info.append(f"{role}: {people}")
            elif isinstance(anime.staff_data, list):
                for staff_entry in anime.staff_data:
                    if hasattr(staff_entry, "role") and hasattr(staff_entry, "name"):
                        staff_info.append(f"{staff_entry.role}: {staff_entry.name}")

            if staff_info:
                content_parts.extend(staff_info)

        return " | ".join(content_parts)

    def _extract_review_content(self, anime: AnimeEntry) -> str:
        """Extract awards, achievements, and recognition for semantic context."""
        content_parts = []

        # Statistics removed - now in payload indexing for precise filtering

        # Awards
        award_info = []
        for award in anime.awards:
            if hasattr(award, "name") and award.name:
                award_part = f"Award: {award.name}"
                if hasattr(award, "year") and award.year:
                    award_part += f" ({award.year})"
                if hasattr(award, "category") and award.category:
                    award_part += f" - {award.category}"
                award_info.append(award_part)
        if award_info:
            content_parts.extend(award_info)

        return " | ".join(content_parts)

    def _extract_temporal_content(self, anime: AnimeEntry) -> str:
        """Extract aired dates, anime season, broadcast, premiere dates."""
        content_parts = []

        # Aired dates
        if anime.aired_dates:
            if hasattr(anime.aired_dates, "from_date") and anime.aired_dates.from_date:
                content_parts.append(f"Aired From: {anime.aired_dates.from_date}")
            if hasattr(anime.aired_dates, "to_date") and anime.aired_dates.to_date:
                content_parts.append(f"Aired To: {anime.aired_dates.to_date}")


        # Broadcast information
        if hasattr(anime, "broadcast") and anime.broadcast:
            if hasattr(anime.broadcast, "day") and anime.broadcast.day:
                broadcast_info = f"Broadcast: {anime.broadcast.day}"
                if hasattr(anime.broadcast, "time") and anime.broadcast.time:
                    broadcast_info += f" at {anime.broadcast.time}"
                content_parts.append(broadcast_info)

        # Premiere month
        if anime.month:
            content_parts.append(f"Premiere Month: {anime.month}")

        return " | ".join(content_parts)

    def _extract_streaming_content(self, anime: AnimeEntry) -> str:
        """Extract streaming platform information and licenses."""
        content_parts = []

        # Streaming platforms
        streaming_info = []
        for stream in anime.streaming_info:
            if hasattr(stream, "name") and stream.name:
                stream_part = f"Platform: {stream.name}"
                if hasattr(stream, "url") and stream.url:
                    stream_part += f" ({stream.url})"
                streaming_info.append(stream_part)
        if streaming_info:
            content_parts.extend(streaming_info)

        # Streaming licenses
        if anime.streaming_licenses:
            content_parts.append(f"Licenses: {', '.join(anime.streaming_licenses)}")

        return " | ".join(content_parts)

    def _extract_related_content(self, anime: AnimeEntry) -> str:
        """Extract related anime and franchise connections."""
        content_parts = []

        # Related anime entries
        related_info = []
        for related in anime.related_anime:
            if hasattr(related, "title") and related.title:
                related_part = f"Related: {related.title}"
                if hasattr(related, "relation_type") and related.relation_type:
                    related_part += f" ({related.relation_type})"
                related_info.append(related_part)
        if related_info:
            content_parts.extend(related_info)

        # Relations with URLs
        relation_info = []
        for relation in anime.relations:
            if hasattr(relation, "title") and relation.title:
                relation_part = f"Relation: {relation.title}"
                if hasattr(relation, "type") and relation.type:
                    relation_part += f" ({relation.type})"
                relation_info.append(relation_part)
        if relation_info:
            content_parts.extend(relation_info)

        return " | ".join(content_parts)

    def _extract_franchise_content(self, anime: AnimeEntry) -> str:
        """Extract trailers, opening themes, ending themes (multimedia content)."""
        content_parts = []

        # Trailers
        trailer_info = []
        for trailer in anime.trailers:
            if hasattr(trailer, "title") and trailer.title:
                trailer_part = f"Trailer: {trailer.title}"
                if hasattr(trailer, "url") and trailer.url:
                    trailer_part += f" ({trailer.url})"
                trailer_info.append(trailer_part)
        if trailer_info:
            content_parts.extend(trailer_info)

        # Opening themes
        opening_info = []
        for opening in anime.opening_themes:
            if hasattr(opening, "title") and opening.title:
                opening_part = f"Opening: {opening.title}"
                if hasattr(opening, "artist") and opening.artist:
                    opening_part += f" by {opening.artist}"
                opening_info.append(opening_part)
        if opening_info:
            content_parts.extend(opening_info)

        # Ending themes
        ending_info = []
        for ending in anime.ending_themes:
            if hasattr(ending, "title") and ending.title:
                ending_part = f"Ending: {ending.title}"
                if hasattr(ending, "artist") and ending.artist:
                    ending_part += f" by {ending.artist}"
                ending_info.append(ending_part)
        if ending_info:
            content_parts.extend(ending_info)

        return " | ".join(content_parts)

    def _extract_episode_content(self, anime: AnimeEntry) -> str:
        """Extract detailed episode information, filler/recap status."""
        content_parts = []

        episode_info = []
        for episode in anime.episode_details:
            if hasattr(episode, "title") and episode.title:
                ep_part = f"Episode: {episode.title}"
                if hasattr(episode, "episode_number") and episode.episode_number:
                    ep_part = f"Episode {episode.episode_number}: {episode.title}"
                if hasattr(episode, "filler") and episode.filler:
                    ep_part += " (Filler)"
                if hasattr(episode, "recap") and episode.recap:
                    ep_part += " (Recap)"
                episode_info.append(ep_part)
        if episode_info:
            content_parts.extend(episode_info)

        return " | ".join(content_parts)


    def _extract_identifiers_content(self, anime: AnimeEntry) -> str:
        """Extract IDs as semantic relationships from List and Dict objects."""
        content_parts = []

        # Extract IDs from various sources
        if hasattr(anime, "mal_id") and anime.mal_id:
            content_parts.append(f"MAL ID: {anime.mal_id}")
        if hasattr(anime, "anilist_id") and anime.anilist_id:
            content_parts.append(f"AniList ID: {anime.anilist_id}")
        if hasattr(anime, "kitsu_id") and anime.kitsu_id:
            content_parts.append(f"Kitsu ID: {anime.kitsu_id}")
        if hasattr(anime, "anidb_id") and anime.anidb_id:
            content_parts.append(f"AniDB ID: {anime.anidb_id}")

        # Character IDs
        id_info = []
        for character in anime.characters:
            if character.character_ids:
                for platform, char_id in character.character_ids.items():
                    id_info.append(f"Character {platform} ID: {char_id}")
        if id_info:
            content_parts.extend(id_info)

        return " | ".join(content_parts)

    # ============================================================================
    # VISUAL VECTOR EXTRACTORS (JinaCLIP v2, 1024-dim)
    # ============================================================================

    def _extract_image_content(self, anime: AnimeEntry) -> List[str]:
        """Extract general anime image URLs (covers, posters, banners, trailers) excluding character images."""
        image_urls = []

        # Process all images from unified images field (now simple URL strings)
        if hasattr(anime, "images") and anime.images:
            # Process covers (highest priority)
            if "covers" in anime.images and anime.images["covers"]:
                for cover_url in anime.images["covers"]:
                    if cover_url:  # Simple URL string
                        image_urls.append(cover_url)

            # Process posters (high quality promotional images)
            if "posters" in anime.images and anime.images["posters"]:
                for poster_url in anime.images["posters"]:
                    if poster_url:  # Simple URL string
                        image_urls.append(poster_url)

            # Process banners (additional visual content)
            if "banners" in anime.images and anime.images["banners"]:
                for banner_url in anime.images["banners"]:
                    if banner_url:  # Simple URL string
                        image_urls.append(banner_url)

            # Process any other image types in the images field
            for image_type, image_list in anime.images.items():
                if image_type not in ["covers", "posters", "banners"] and image_list:
                    for image_url in image_list:
                        if image_url:  # Simple URL string
                            image_urls.append(image_url)

        # Trailer thumbnails (promotional visual content)
        for trailer in anime.trailers:
            if hasattr(trailer, "thumbnail_url") and trailer.thumbnail_url:
                image_urls.append(trailer.thumbnail_url)
            if hasattr(trailer, "image_url") and trailer.image_url:
                image_urls.append(trailer.image_url)

        # Remove duplicates while preserving order
        unique_urls = list(dict.fromkeys(image_urls))
        return unique_urls

    def _extract_character_image_content(self, anime: AnimeEntry) -> List[str]:
        """Extract character image URLs for character-specific visual embedding."""
        character_image_urls = []

        # Extract character images separately for character identification and recommendations
        for character in anime.characters:
            if character.images:
                # character.images is Dict[str, str] with platform keys mapping to URLs
                for platform, image_url in character.images.items():
                    if image_url:
                        character_image_urls.append(image_url)

        # Remove duplicates while preserving order
        unique_urls = list(dict.fromkeys(character_image_urls))
        return unique_urls

    # ============================================================================
    # UTILITY METHODS
    # ============================================================================

    def get_vector_types(self) -> Dict[str, str]:
        """Get mapping of vector names to their types (text/visual)."""
        return {
            # Text vectors (BGE-M3, 1024-dim)
            "title_vector": "text",
            "character_vector": "text",
            "genre_vector": "text",
            "technical_vector": "text",
            "staff_vector": "text",
            "review_vector": "text",
            "temporal_vector": "text",
            "streaming_vector": "text",
            "related_vector": "text",
            "franchise_vector": "text",
            "episode_vector": "text",
            "identifiers_vector": "text",
            # Visual vectors (JinaCLIP v2, 1024-dim)
            "image_vector": "visual",
            "character_image_vector": "visual",
        }

    def validate_mapping(self, vector_data: Dict[str, Any]) -> bool:
        """Validate that vector data contains all expected vectors."""
        expected_vectors = set(self.get_vector_types().keys())
        actual_vectors = set(vector_data.keys())

        missing_vectors = expected_vectors - actual_vectors
        if missing_vectors:
            self.logger.warning(f"Missing vectors: {missing_vectors}")
            return False

        return True

    def _extract_image_url(self, anime: AnimeEntry) -> str:
        """Extract the primary image URL for visual embedding using unified images field.

        Args:
            anime: AnimeEntry instance

        Returns:
            Primary image URL or empty string if not available
        """
        # Use unified images field with priority: covers -> posters -> banners
        if hasattr(anime, "images") and anime.images:
            # Priority 1: covers (best quality cover images)
            if "covers" in anime.images and anime.images["covers"]:
                for cover_url in anime.images["covers"]:
                    if cover_url:  # Simple URL string check
                        return cover_url

            # Priority 2: posters (good quality poster images)
            if "posters" in anime.images and anime.images["posters"]:
                for poster_url in anime.images["posters"]:
                    if poster_url:  # Simple URL string check
                        return poster_url

            # Priority 3: banners (fallback option)
            if "banners" in anime.images and anime.images["banners"]:
                for banner_url in anime.images["banners"]:
                    if banner_url:  # Simple URL string check
                        return banner_url

        return ""
