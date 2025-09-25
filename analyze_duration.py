#!/usr/bin/env python3
"""
Analyze duration values in the enrichment database.
"""

import json
from typing import Dict, List, Any, Optional

def analyze_durations():
    """Analyze all duration values in the database."""

    # Load the database
    with open('./data/qdrant_storage/enriched_anime_database.json', 'r', encoding='utf-8') as f:
        data = json.load(f)

    anime_duration_analysis = []
    episode_duration_analysis = []

    print("🔍 ANALYZING ANIME DURATIONS")
    print("=" * 80)

    for i, anime in enumerate(data['data']):
        title = anime.get('title', f'Anime {i}')
        duration = anime.get('duration')

        if duration is not None:
            # Handle structured duration format {value: X, unit: 'SECONDS'}
            if isinstance(duration, dict):
                duration_seconds = duration.get('value', 0)
                unit = duration.get('unit', 'SECONDS')
            else:
                duration_seconds = duration
                unit = 'SECONDS'

            duration_minutes = duration_seconds / 60
            anime_duration_analysis.append({
                'title': title,
                'duration_seconds': duration_seconds,
                'duration_minutes': duration_minutes,
                'sub_minute': duration_minutes < 1.0
            })

            print(f"{title}")
            print(f"  Duration: {duration_seconds} {unit.lower()} = {duration_minutes:.2f} minutes")
            if duration_minutes < 1.0:
                print(f"  ⚠️  SUB-MINUTE CONTENT")
            print()
        else:
            print(f"{title}")
            print(f"  Duration: None")
            print()

    print("\n🎬 ANALYZING EPISODE DURATIONS")
    print("=" * 80)

    for anime in data['data']:
        title = anime.get('title', 'Unknown')
        episode_details = anime.get('episode_details', [])

        if episode_details:
            print(f"\n📺 {title}")
            for ep in episode_details:
                ep_num = ep.get('episode_number', 'Unknown')
                ep_title = ep.get('title', 'Untitled')
                duration = ep.get('duration')

                if duration is not None:
                    # Handle structured or simple duration format
                    if isinstance(duration, dict):
                        duration_seconds = duration.get('value', 0)
                        unit = duration.get('unit', 'SECONDS')
                    else:
                        duration_seconds = duration
                        unit = 'SECONDS'

                    duration_minutes = duration_seconds / 60
                    episode_duration_analysis.append({
                        'anime_title': title,
                        'episode_number': ep_num,
                        'episode_title': ep_title,
                        'duration_seconds': duration_seconds,
                        'duration_minutes': duration_minutes,
                        'sub_minute': duration_minutes < 1.0
                    })

                    print(f"  Episode {ep_num}: {ep_title}")
                    print(f"    Duration: {duration_seconds} {unit.lower()} = {duration_minutes:.2f} minutes")
                    if duration_minutes < 1.0:
                        print(f"    ⚠️  SUB-MINUTE EPISODE")
                else:
                    print(f"  Episode {ep_num}: {ep_title}")
                    print(f"    Duration: None")

    # Summary Statistics
    print("\n" + "=" * 80)
    print("📊 SUMMARY STATISTICS")
    print("=" * 80)

    # Anime-level stats
    total_anime_with_duration = len(anime_duration_analysis)
    sub_minute_anime = [a for a in anime_duration_analysis if a['sub_minute']]

    print(f"\n🎭 ANIME-LEVEL DURATIONS:")
    print(f"Total anime with duration: {total_anime_with_duration}")
    print(f"Sub-minute anime: {len(sub_minute_anime)}")
    print(f"Percentage sub-minute: {len(sub_minute_anime)/total_anime_with_duration*100:.1f}%")

    if sub_minute_anime:
        print(f"\nSUB-MINUTE ANIME:")
        for anime in sub_minute_anime:
            print(f"  • {anime['title']}: {anime['duration_seconds']}s ({anime['duration_minutes']:.2f}min)")

    # Episode-level stats
    total_episodes_with_duration = len(episode_duration_analysis)
    sub_minute_episodes = [e for e in episode_duration_analysis if e['sub_minute']]

    print(f"\n🎬 EPISODE-LEVEL DURATIONS:")
    print(f"Total episodes with duration: {total_episodes_with_duration}")
    print(f"Sub-minute episodes: {len(sub_minute_episodes)}")
    if total_episodes_with_duration > 0:
        print(f"Percentage sub-minute: {len(sub_minute_episodes)/total_episodes_with_duration*100:.1f}%")

    if sub_minute_episodes:
        print(f"\nSUB-MINUTE EPISODES:")
        for ep in sub_minute_episodes:
            print(f"  • {ep['anime_title']} - Episode {ep['episode_number']}: {ep['duration_seconds']}s ({ep['duration_minutes']:.2f}min)")

    # Duration ranges
    if anime_duration_analysis:
        durations = [a['duration_minutes'] for a in anime_duration_analysis]
        print(f"\n📈 ANIME DURATION RANGES:")
        print(f"Minimum: {min(durations):.2f} minutes ({min(durations)*60:.0f} seconds)")
        print(f"Maximum: {max(durations):.2f} minutes ({max(durations)*60:.0f} seconds)")
        print(f"Average: {sum(durations)/len(durations):.2f} minutes")

    if episode_duration_analysis:
        ep_durations = [e['duration_minutes'] for e in episode_duration_analysis]
        print(f"\n📈 EPISODE DURATION RANGES:")
        print(f"Minimum: {min(ep_durations):.2f} minutes ({min(ep_durations)*60:.0f} seconds)")
        print(f"Maximum: {max(ep_durations):.2f} minutes ({max(ep_durations)*60:.0f} seconds)")
        print(f"Average: {sum(ep_durations)/len(ep_durations):.2f} minutes")

if __name__ == "__main__":
    analyze_durations()