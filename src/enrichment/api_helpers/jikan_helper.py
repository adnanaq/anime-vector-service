#!/usr/bin/env python3
"""
Reusable script for fetching detailed data from Jikan API with proper rate limiting.

Usage:
    python scripts/fetch_detailed_jikan_data.py episodes <anime_id> <input_file> <output_file>
    python scripts/fetch_detailed_jikan_data.py characters <anime_id> <input_file> <output_file>

Examples:
    python scripts/fetch_detailed_jikan_data.py episodes 21 temp/episodes.json temp/episodes_detailed.json
    python scripts/fetch_detailed_jikan_data.py characters 21 temp/characters.json temp/characters_detailed.json
"""

import json
import requests
import time
import os
import sys
import argparse
from typing import List, Dict, Any, Optional


class JikanDetailedFetcher:
    """
    Fetches detailed data from Jikan API with proper rate limiting.
    Supports episodes and characters endpoints.
    """
    
    def __init__(self, anime_id: str, data_type: str):
        self.anime_id = anime_id
        self.data_type = data_type  # 'episodes' or 'characters'
        self.request_count = 0
        self.start_time = time.time()
        self.batch_size = 50
        
        # Jikan API rate limits: 3 requests per second, 60 per minute
        self.max_requests_per_second = 3
        self.max_requests_per_minute = 60
        
    def respect_rate_limits(self):
        """Ensure we don't exceed Jikan API rate limits."""
        current_time = time.time()
        elapsed = current_time - self.start_time
        
        # Reset counter every minute
        if elapsed >= 60:
            self.request_count = 0
            self.start_time = current_time
            elapsed = 0
        
        # If we've made 60 requests in current minute, wait
        if self.request_count >= self.max_requests_per_minute:
            wait_time = 60 - elapsed
            if wait_time > 0:
                print(f'Rate limit reached. Waiting {wait_time:.1f} seconds...')
                time.sleep(wait_time)
                self.request_count = 0
                self.start_time = time.time()
        
        # Ensure we don't exceed 3 requests per second
        if self.request_count % self.max_requests_per_second == 0 and self.request_count > 0:
            time.sleep(1)
    
    def fetch_episode_detail(self, episode_id: int) -> Optional[Dict[str, Any]]:
        """Fetch detailed episode data from Jikan API."""
        self.respect_rate_limits()
        
        try:
            url = f'https://api.jikan.moe/v4/anime/{self.anime_id}/episodes/{episode_id}'
            response = requests.get(url, timeout=10)
            self.request_count += 1
            
            if response.status_code == 200:
                episode_detail = response.json()['data']
                
                return {
                    'episode_number': episode_id,
                    'url': episode_detail.get('url'),
                    'title': episode_detail.get('title'),
                    'title_japanese': episode_detail.get('title_japanese'),
                    'title_romaji': episode_detail.get('title_romaji'),
                    'aired': episode_detail.get('aired'),
                    'score': episode_detail.get('score'),
                    'filler': episode_detail.get('filler', False),
                    'recap': episode_detail.get('recap', False),
                    'duration': episode_detail.get('duration'),
                    'synopsis': episode_detail.get('synopsis')
                }
            
            elif response.status_code == 429:
                print(f'Rate limit hit for episode {episode_id}. Waiting and retrying...')
                time.sleep(5)
                return self.fetch_episode_detail(episode_id)  # Retry once
            
            else:
                print(f'Error fetching episode {episode_id}: HTTP {response.status_code}')
                return None
                
        except Exception as e:
            print(f'Error fetching episode {episode_id}: {e}')
            return None
    
    def fetch_character_detail(self, character_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Fetch detailed character data from Jikan API."""
        character_id = character_data['character']['mal_id']
        self.respect_rate_limits()
        
        try:
            url = f'https://api.jikan.moe/v4/characters/{character_id}'
            response = requests.get(url, timeout=10)
            self.request_count += 1
            
            if response.status_code == 200:
                character_detail = response.json()['data']
                
                return {
                    'character_id': character_id,
                    'url': character_detail.get('url'),
                    'name': character_detail.get('name'),
                    'name_kanji': character_detail.get('name_kanji'),
                    'nicknames': character_detail.get('nicknames', []),
                    'about': character_detail.get('about'),
                    'images': character_detail.get('images', {}),
                    'favorites': character_detail.get('favorites'),
                    'role': character_data.get('role'),
                    'voice_actors': character_data.get('voice_actors', [])
                }
            
            elif response.status_code == 429:
                print(f'Rate limit hit for character {character_id}. Waiting and retrying...')
                time.sleep(5)
                return self.fetch_character_detail(character_data)  # Retry once
            
            else:
                print(f'Error fetching character {character_id}: HTTP {response.status_code}')
                return None
                
        except Exception as e:
            print(f'Error fetching character {character_id}: {e}')
            return None
    
    def append_batch_to_file(self, batch_data: List[Dict[str, Any]], progress_file: str) -> int:
        """Append batch data to progress file."""
        # Load existing data
        if os.path.exists(progress_file):
            with open(progress_file, 'r', encoding='utf-8') as f:
                all_data = json.load(f)
        else:
            all_data = []
        
        # Append new batch
        all_data.extend(batch_data)
        
        # Save updated data
        with open(progress_file, 'w', encoding='utf-8') as f:
            json.dump(all_data, f, ensure_ascii=False, indent=2)
        
        return len(all_data)
    
    def fetch_detailed_data(self, input_file: str, output_file: str):
        """Main method to fetch detailed data with batch processing. When processing each object should 
        have these properties, example for characters:
        {
            "name": "Character Full Name",
            "role": "Main/Supporting/Minor",
            "name_variations": ["Alternative names"],
            "name_kanji": "漢字名",
            "name_native": "Native Name", 
            "character_ids": {{"mal": 12345, "anilist": null}},
            "images": {{"mal": "image_url", "anilist": null}},
            "description": "Character description",
            "age": "Character age or null",
            "gender": "Male/Female/Other or null",
            "voice_actors": [
                {{"name": "Voice Actor Name", "language": "Japanese"}}
            ]
        }
        """
        # Load input data
        with open(input_file, 'r', encoding='utf-8') as f:
            input_data = json.load(f)
        
        if self.data_type == 'episodes':
            # For episodes, we can use the episode count directly from anime data
            # or process existing episode list
            if isinstance(input_data, list):
                total_items = len(input_data)
                episode_ids = [ep['mal_id'] for ep in input_data]
            elif 'data' in input_data:
                # Handle paginated episode data
                input_data = input_data['data']
                total_items = len(input_data)
                episode_ids = [ep['mal_id'] for ep in input_data]
            else:
                # If input is anime data with episode count
                total_items = input_data.get('episodes', 0)
                episode_ids = list(range(1, total_items + 1))
            print(f'Fetching detailed data for {total_items} episodes...')
        else:  # characters
            if 'data' in input_data:
                input_data = input_data['data']
            total_items = len(input_data)
            print(f'Fetching detailed data for {total_items} characters...')
        
        # Progress tracking
        progress_file = f'{output_file}.progress'
        if os.path.exists(progress_file):
            with open(progress_file, 'r', encoding='utf-8') as f:
                existing_data = json.load(f)
            print(f'Found existing progress: {len(existing_data)} items already fetched')
            start_index = len(existing_data)
        else:
            start_index = 0
        
        batch_data = []
        
        # Process items starting from where we left off
        for i in range(start_index, total_items):
            if self.data_type == 'episodes':
                item_id = episode_ids[i]
                detailed_item = self.fetch_episode_detail(item_id)
                item_type = 'episode'
            else:  # characters
                item = input_data[i]
                detailed_item = self.fetch_character_detail(item)
                item_type = 'character'
                item_id = item['character']['mal_id']
            
            if detailed_item:
                batch_data.append(detailed_item)
            
            # Progress update
            if (i + 1) % 10 == 0:
                print(f'Progress: {i+1}/{total_items} {item_type}s fetched')
            
            # Save progress every batch_size items
            if len(batch_data) >= self.batch_size:
                total_count = self.append_batch_to_file(batch_data, progress_file)
                print(f'Appended batch: {len(batch_data)} {item_type}s (total: {total_count})')
                batch_data = []  # Clear batch
        
        # Save any remaining items in the final batch
        if batch_data:
            total_count = self.append_batch_to_file(batch_data, progress_file)
            print(f'Appended final batch: {len(batch_data)} {item_type}s (total: {total_count})')
        
        # Load final data and create final file
        with open(progress_file, 'r', encoding='utf-8') as f:
            all_detailed_data = json.load(f)
        
        print(f'\\nCompleted fetching {len(all_detailed_data)} detailed {item_type}s')
        
        # Sort data by ID
        if self.data_type == 'episodes':
            all_detailed_data.sort(key=lambda x: x.get('episode_number', 0))
            synopsis_count = sum(1 for ep in all_detailed_data if ep.get('synopsis'))
            print(f'Episodes with synopsis: {synopsis_count}/{len(all_detailed_data)}')
        else:  # characters
            all_detailed_data.sort(key=lambda x: x.get('character_id', 0))
            about_count = sum(1 for char in all_detailed_data if char.get('about'))
            print(f'Characters with about text: {about_count}/{len(all_detailed_data)}')
        
        # Save final detailed data
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(all_detailed_data, f, ensure_ascii=False, indent=2)
        
        print(f'Final detailed data saved to {output_file}')
        
        # Clean up progress file
        if os.path.exists(progress_file):
            os.remove(progress_file)
            print(f'Cleaned up progress file: {progress_file}')


def main():
    parser = argparse.ArgumentParser(description='Fetch detailed data from Jikan API')
    parser.add_argument('data_type', choices=['episodes', 'characters'], 
                       help='Type of data to fetch')
    parser.add_argument('anime_id', help='Anime ID (MAL ID)')
    parser.add_argument('input_file', help='Input file path')
    parser.add_argument('output_file', help='Output file path')
    
    args = parser.parse_args()
    
    # Validate input file exists
    if not os.path.exists(args.input_file):
        print(f'Error: Input file {args.input_file} does not exist')
        sys.exit(1)
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    
    # Create fetcher and run
    fetcher = JikanDetailedFetcher(args.anime_id, args.data_type)
    fetcher.fetch_detailed_data(args.input_file, args.output_file)


if __name__ == '__main__':
    main()