# Gemini Self-Instruction Manual for Anime Data Enrichment

This document outlines the PRODUCTION-LEVEL process for enriching anime data with **MULTI-AGENT CONCURRENT PROCESSING** capabilities.

## 1. Objective

The primary goal is to take a raw anime data object (from an offline database) and enrich it with additional information from external APIs (Jikan, AnimSchedule, Kitsu, Anime-Planet, AniList, AniDB) and AI processing. This is a PRODUCTION implementation of the `enrich_anime_from_offline_data` method with **concurrent multi-agent support** for enhanced performance.

**CRITICAL: This is NOT a simulation. This is a production-level enrichment process that must use REAL API calls and REAL data.**

## 2. Inputs

- `offline_anime_data`: A JSON object representing a single anime. This will be loaded from `data/current_anime.json` in Step 0 and used throughout the enrichment process.

## 3. The Process

**CRITICAL RULE: I must not proceed to the next step unless the current step has been completed successfully.**

### Step 0: Load Next Anime Entry (Multi-Agent Concurrent Processing)

**MULTI-AGENT PROCESSING MODEL**: This system supports concurrent processing by multiple agents using numbered processing files to prevent conflicts and enable scalability.

1.  **Agent Slot Detection:**
    - Scan for existing processing files in `temp/` directory at the root of this rpeo: `current_anime.json`, `current_anime_2.json`, `current_anime_3.json`, etc.
    - Determine next available agent slot (e.g., if `current_anime.json` and `current_anime_2.json` exist, create `current_anime_3.json`)
    - This enables multiple agents to work simultaneously without conflicts

2.  **Load Source Database:**
    - Load `data/anime-offline-database.json` from the project root
    - Extract anime entries sequentially from the `data` array (first available entry not being processed)
    - If the `data` array is empty, all anime have been processed - stop here
    - Skip entries that match any currently processing anime (check against all existing `current_anime*.json` files)

3.  **Prepare Current Anime (Agent-Specific):**
    - Create temp directory using agent-specific naming: `temp/current_anime_<N>.json`
    - Extract the first entry from the `data` array and save it to your agent's processing file you created above
    - Add `processing: true` to the entry to indicate this agent is currently processing it
    - This numbered file becomes the `offline_anime_data` for all subsequent processing steps
    - Each agent has its own processing file, enabling parallel processing without conflicts
    - The file enables manual review and debugging if needed

4.  **Verification:**
    - Confirm your agent's processing file contains a valid anime object
    - Verify the anime has required fields (title, sources, etc.)
    - Log which anime is being processed and which agent slot is being used
    - Create temp directory using agent-specific naming and N you selected above: `temp/<first word from anime title>_agent<N>/`

### Step 1: Extract Platform IDs

1.  Check the `sources` field in the `offline_anime_data`.
2.  **MAL ID Extraction:**
    - Find the URL containing "myanimelist.net/anime/".
    - Extract the numerical ID from this URL.
    - If no MAL ID is found, Jikan data fetching will be skipped.
3.  **Kitsu ID Extraction:**
    - Find the URL containing "kitsu.io/anime/" or "kitsu.app/anime/".
    - Extract the numerical ID from this URL.
    - If no Kitsu ID is found, Kitsu data fetching will be skipped.
4.  **Anime-Planet Slug Extraction:**
    - Find the URL containing "anime-planet.com/anime/".
    - Extract the slug from this URL (e.g., "dandadan" from "https://www.anime-planet.com/anime/dandadan").
    - If no Anime-Planet URL is found, Anime-Planet data fetching will be skipped.
5.  **AniList ID Extraction:**
    - Find the URL containing "anilist.co/anime/".
    - Extract the numerical ID from this URL.
    - If no AniList ID is found, AniList data fetching will be skipped.
6.  **AniDB ID Extraction:**
    - Find the URL containing "anidb.net/anime/" or "anidb.info/a".
    - Extract the numerical ID from this URL.
    - If no AniDB ID is found, AniDB data fetching will be skipped.

### Step 2: Concurrent External API Data Fetching

Use each services' respective helper funciton wheere possible.

1.  **Jikan Anime Full Data:** Fetch full anime data from Jikan API using the MAL ID. Save in temporary file in temp/<first word from anime title>/jikan.json to be used later
    - URL: `https://api.jikan.moe/v4/anime/{mal_id}/full`
2.  **Jikan Episodes Data:** Fetch episode data from Jikan API. Use episodes property from the `offline_anime_data` (from base_anime_sample.json). Do not skip any episode. Save in temporary file in temp/<first word from anime title>/episodes.json to be used later
    - **FIRST:** Create temp/<first word from anime title>/episodes.json with the episode count from offline_anime_data: `{"episodes": <episode_count>}`
    - **IMPORTANT:** For anime with >100 episodes, use the reusable script:
      `python src/enrichment/api_helpers/jikan_helper.py episodes {mal_id} temp/<first word from anime title>/episodes.json temp/<first word from anime title>/episodes_detailed.json`
    - **CRITICAL:**
      - NEVER give up on fetching all episodes regardless of time taken. Wait for the reusable script to complete fully before proceeding. ALL episodes MUST be fetched - no exceptions.
      - The reusable script will read the episode count from temp/<first word from anime title>/episodes.json and fetch detailed data for each episode from the Jikan API endpoints
    - URL: `https://api.jikan.moe/v4/anime/{mal_id}/episodes/{episode_num}`
3.  **Jikan Characters Data:** Fetch character data from Jikan API. Do not skip any character. Save in temporary file in temp/<first word from anime title>/characters.json to be used later
    - **IMPORTANT:** For anime with >50 characters, use the reusable script: `python src/enrichment/api_helpers/jikan_helper.py characters {mal_id} temp/<first word from anime title>/characters.json temp/<first word from anime title>/characters_detailed.json`
    - **CRITICAL:** NEVER give up on fetching all characters regardless of time taken. Wait for the reusable script to complete fully before proceeding. ALL characters MUST be fetched - no exceptions.
    - URL: `https://api.jikan.moe/v4/anime/{mal_id}/characters`
4.  **AnimSchedule Data:** Find a matching anime on AnimSchedule using REAL API calls. Save in temporary file in temp/<first word from anime title>/as.json to be used later
    - URL: `https://animeschedule.net/api/v3/anime?q={search_term}`
    - This involves a smart search using title, synonyms, and other metadata from `offline_anime_data`. Follow the logic in `animeschedule_helper.py` to implement proper search strategy.
    - **NEVER mock this data** - Always make real API calls to AnimSchedule to get accurate, up-to-date information including statistics, images, and external links.
5.  **Kitsu Data:** Fetch comprehensive Kitsu data using the extracted Kitsu ID. Save in temporary file in temp/<first word from anime title>/kitsu.json to be used later
    - **ONLY if Kitsu ID was found in Step 1** - otherwise skip this step entirely
    - Use the `KitsuEnrichmentHelper.fetch_all_data(kitsu_id)` method from `src/enrichment/api_helpers/kitsu_helper.py`
    - **NEVER mock this data** - Always make real API calls to Kitsu to get accurate information including categories, statistics, images, and NSFW flags.
6.  **Anime-Planet Data:** Fetch comprehensive Anime-Planet data using web scraping. Save in temporary file in temp/<first word from anime title>/animeplanet.json to be used later
    - **ONLY if Anime-Planet URL was found in Step 1** - otherwise skip this step entirely
    - Use the `AnimePlanetEnrichmentHelper.fetch_all_data(offline_anime_data)` method from `src/enrichment/api_helpers/animeplanet_helper.py`
    - **NEVER mock this data** - Always make real web scraping calls to Anime-Planet to get accurate information including ratings, images, rankings, and genre data.
7.  **AniList Data:** Fetch comprehensive AniList data using the extracted AniList ID. Save in temporary file in temp/<first word from anime title>/anilist.json to be used later
    - **ONLY if AniList ID was found in Step 1** - otherwise skip this step entirely
    - use the reusable script: `python src/enrichment/api_helpers/anilist_helper.py`
    - **NEVER mock this data** - Always make real GraphQL calls to AniList to get accurate information including detailed character data, staff information, and comprehensive statistics.
8.  **AniDB Data:** Fetch comprehensive AniDB data using the extracted AniDB ID. Save in temporary file in temp/<first word from anime title>/anidb.json to be used later
    - **IMPORTANT:**
    - **ONLY if AniDB ID was found in Step 1** - otherwise skip this step entirely
    - Use the reusable script: `python src/enrichment/api_helpers/anidb_helper.py`
    - **NEVER mock this data** - Always make real web scraping calls to AniDB to get accurate information including comprehensive staff data, episode information, and technical details.

### Step 3: Pre-process Episode Data

1.  From the fetched Jikan episodes data, create a simplified list of episodes, extracting only the following fields for each episode: `url`, `title`, `title_japanese`, `title_romaji`, `aired`, `score`, `filler`, `recap`, `duration`, `synopsis`.

### Step 4: Execute 6-Stage Enrichment Pipeline (Multi-Agent Parallel Processing)

This is the core of the process, where AI is used to process the collected data. Act as expert data scientist who is collecting, sanitizing and organizing anime data, and generate the expected JSON output for each stage based on the provided data and the logic in the corresponding prompt templates. **OPTIMIZATION: Run/Spawn 4 concurrent agents to parallelize the 6 stages.** When creating script, DO NOT use ChatGPT or ANthropic API.

**IMPORTANT: Strictly follow AnimeEtry schema from `src/models/anime.py` at each stage**

**AGENT EXECUTION REQUIREMENTS:**

**CRITICAL INSTRUCTIONS FOR ALL AGENTS:**

1. **Read Prompt Files Completely**: Each agent MUST read their assigned prompt file(s) from top to bottom before starting processing
2. **Handle Large Data Files**: If input data files are large (>25k tokens), use chunking to read them completely - DO NOT skip data due to size limits
3. **Process ALL Data Sources**: Every data source mentioned in the prompt must be processed - do not leave any arrays empty unless the source data is genuinely unavailable
4. **No "Pending" Status**: Do not mark anything as "pending" or "to be processed later" - perform all processing steps immediately
5. **Verification Before Output**: Before saving stage output files, verify that all expected data sections contain actual data, not placeholders

**COMPLETION VERIFICATION:**

- Stage 1: Metadata must include themes from multiple sources, organized images, external links
- Stage 2: Episode details must include data from all available episode sources
- Stage 3: Relationships must process EVERY URL with intelligent title extraction
- Stage 4: Statistics must include data from ALL available API sources
- Stage 5: Characters must include data from multiple sources with proper matching
- Stage 6: Staff must include production staff, studios, producers, AND voice actors

**IF ANY EXPECTED SECTION IS EMPTY OR INCOMPLETE, THE AGENT MUST RE-EXAMINE THE DATA SOURCES AND PROCESSING LOGIC**

**AGENT DISTRIBUTION:**

**Agent 1 - Metadata Specialist:**

1.  **Stage 1: Metadata Extraction** PROMPT: src/enrichment/prompts/stages/01_metadata_extraction.txt
    - **Inputs:** `offline_anime_data`, core Jikan data, AnimSchedule data, Kitsu data, Anime-Planet data, AniList data, AniDB data.
    - **Action:** Generate a JSON object containing `synopsis`, `genres`, `demographics`, `themes`, `source_material`, `rating`, `content_warnings`, `nsfw`, `title_japanese`, `title_english`, `background`, `aired_dates`, `broadcast`, `broadcast_schedule`, `premiere_dates`, `delay_information`, `episode_overrides`, `external_links`, `statistics`, `images`, `month`.
    - **Output:** `temp/<first word from anime title>/stage1_metadata.json`

**Agent 2 - Episode Specialist:** 2. **Stage 2: Episode Processing** PROMPT: src/enrichment/prompts/stages/02_episode_processing.txt - **Inputs:** The pre-processed episode list. - **Action:** Process episodes in batches. For each batch, generate a list of `episode_details`. DO NOT skip any episode - **Output:** `temp/<first word from anime title>/stage2_episodes.json`

**Agent 3 - Relationship & Media Specialist:** 3. **Stage 3: Relationship Analysis** PROMPT: src/enrichment/prompts/stages/03_relationship_analysis.txt
**Inputs:** `relatedAnime` URLs from `offline_anime_data`, and `relations` from Jikan data. - **Action:** Generate a JSON object with `relatedAnime` and `relations` fields. - **CRITICAL RULES:** - Process EVERY URL. The number of output `relatedAnime` entries must exactly match the number of input URLs. - Use "Intelligent Title Extraction": - Scan all URLs to find explicit titles (e.g., from anime-planet). - Visit each site to find the approprioate title and relation - Do not use numeric ID from url as the title. - **FORBIDDEN PATTERNS:** Do not use generic titles like "Anime [ID]", "Unknown Title", or "Anime 19060". - **Output:** `temp/<first word from anime title>/stage3_relationships.json`

4.  **Stage 4: Statistics** PROMPT: src/enrichment/prompts/stages/04_statistics_media.txt
    **Inputs:** Jikan statistics and media data, AniList statistics, AniDB statistics and staff data.
    **Action:** Generate a JSON object with `statistics`.
    **CRITICAL RULES:**
    - The `statistics` field must be a nested object with source as a key, like `mal`, `animeschedule`, `kitsu`, `animeplanet`, `anilist`, `anidb` key (e.g., `{"statistics": {"mal": {...}, "anilist": {...}, "anidb": {...}}}`). There could be multiple sources.
    - Prioritize AniDB for comprehensive staff data merging including detailed roles and credits.
      **Output:** `temp/<first word from anime title>/stage4_statistics_media.json`

**Agent 4 - Character & Staff Specialist:** 5. **Stage 5: Character Processing** PROMPT: src/enrichment/prompts/stages/05_character_processing.txt - **Inputs:** Jikan characters data, AniList characters data. - **Action:** Process characters in batches. For each batch, generate a list of `characters`. DO NOT skip any character. Merge character data from multiple sources for comprehensive character profiles. - **Output:** `temp/<first word from anime title>/stage5_characters.json`

6.  **Stage 6: Staff Processing** PROMPT: src/enrichment/prompts/stages/06_staff_processing.txt
    - **Inputs:** AniDB staff data, AniList staff data, Jikan company data.
    - **Action:** Generate a JSON object with comprehensive `staff_data` including production staff (directors, music composers, character designers), studios, producers, and voice actors with multi-source integration and biographical enhancement.
    - **Output:** `temp/<first word from anime title>/stage6_staff.json`

**SYNCHRONIZATION POINT:** All 4 agents must complete their assigned stages before proceeding to Step 5. Verify all stage output files exist:

- `temp/<first word from anime title>/stage1_metadata.json`
- `temp/<first word from anime title>/stage2_episodes.json`
- `temp/<first word from anime title>/stage3_relationships.json`
- `temp/<first word from anime title>/stage4_statistics_media.json`
- `temp/<first word from anime title>/stage5_characters.json`
- `temp/<first word from anime title>/stage6_staff.json`

### Step 5: Programmatic Assembly

1.  **Synchronization Check:** Verify all 6 stage output files from the 4 agents exist before proceeding:
    - `temp/<first word from anime title>/stage1_metadata.json` (Agent 1)
    - `temp/<first word from anime title>/stage2_episodes.json` (Agent 2)
    - `temp/<first word from anime title>/stage3_relationships.json` (Agent 3)
    - `temp/<first word from anime title>/stage4_statistics_media.json` (Agent 3)
    - `temp/<first word from anime title>/stage5_characters.json` (Agent 4)
    - `temp/<first word from anime title>/stage6_staff.json` (Agent 4)

2.  Merge the results from all six stages into a single JSON object.
3.  Start with the original `offline_anime_data`, and append animeschedule url for the relevent anime in the sources proeprty.
4.  Update the fields with the data from each stage's output following AnimeEtry schema from `src/models/anime.py`
5.  **CRITICAL: Schema-Compliant Field Ordering**
    - Order all fields according to AnimeEntry schema defined in `src/models/anime.py`
    - **FINAL FIELD**: enrichment_metadata (MUST ALWAYS BE LAST)
    - **Nested Objects**: Ensure characters, staff_data, episode_details, etc. also follow their respective schema ordering
6.  **CRITICAL: Data Quality Cleanup**
    - **Remove null values**: Any property with `null` value must be removed from final output
    - **Remove empty strings**: Any property with empty string `""` value must be removed from final output
    - **Preserve empty collections**: Empty arrays `[]` and empty objects `{}` must be preserved as they indicate intentional empty state
    - **Exception**: Required fields (title, status, type, episodes, sources) must never be removed even if empty
    - **Nested cleanup**: Apply cleanup rules recursively to all nested objects and arrays
    - **Character cleanup**: Remove null voice_actors, descriptions, ages, genders from character entries
    - **Staff cleanup**: Remove null biographies, birth_dates, images from staff member entries
7.  **CRITICAL: Unicode Character Handling** - When saving the final JSON output, always use `ensure_ascii=False` and `encoding='utf-8'` to properly display international characters (Greek, Cyrillic, Japanese, etc.) instead of Unicode escape sequences.
8.  **CRITICAL: Final Field Placement** - Add "enrichment_metadata" property as the VERY LAST field in the JSON object with the following schema, example:
    json

```
"enrichment_metadata": {
        "source": "multi-source",
        "enriched_at": "2025-07-24T19:42:09.282039Z",
        "success": true, (if successful)
        "error_message": null
      }
```

### Step 6: Database Persistence & Cleanup (Multi-Agent Safe)

Receive confirmation from user before procesing this step.

1.  **Update Enriched Database:**
    - Load `data/enriched_anime_database.json`
    - If file doesn't exist, create with initial structure following anime-offline-database.json format
    - Append the final enriched anime object to the `data` array
    - Update `lastUpdate` field to current date (YYYY-MM-DD format)
    - Increment `totalAnime` count in `enrichmentInfo`
    - Save file with proper JSON formatting using `ensure_ascii=False` and `encoding='utf-8'`

2.  **Update Source Database (Agent-Safe Removal):**
    - Load `anime-offline-database.json`
    - **CRITICAL**: Remove the processed anime entry by matching against the exact anime from your agent's processing file
    - **Verification**: Verify removal by matching both `title` and `sources` array against your processed anime
    - **Safety Check**: Ensure you're removing the correct anime - not just the first entry
    - Update `lastUpdate` field to current date (YYYY-MM-DD format)
    - Save the updated source database with proper JSON formatting

3.  **Agent Cleanup (Multi-Agent Environment):**
    - Remove your agent's processing file (e.g., `data/current_anime_2.json`)
    - Clean up your agent's temp directory (e.g., `temp/<anime_title>/`)
    - **Do NOT** remove other agents' processing files or temp directories
    - Verify cleanup completed successfully

4.  **Completion Confirmation:**
    - Confirm the anime has been added to the enriched database
    - Confirm the anime has been removed from the source database (with title verification)
    - Report remaining anime count in the source database
    - Confirm agent-specific cleanup completed
    - Log successful completion for this agent's processing session

**MULTI-AGENT COORDINATION:**

- Each agent manages its own processing files and cleanup
- Database updates are atomic and safe for concurrent access
- Verification prevents accidental removal of wrong anime entries
- Clean separation ensures agents don't interfere with each other

## 4. Output Schema

The final output of this process must be a single JSON object that validates against the `AnimeEntry` Pydantic model defined in `src/models/anime.py`. I will ensure that the final, merged object adheres strictly to this schema, including all specified fields and their data types.

## 5. My Role

You will act as the Data Enrichment Expert. You will go through each step of the process, making REAL requests to the external APIs and then generating the expected JSON output for each of the five AI-driven stages. I will then perform the final programmatic merge to produce the final, enriched anime data object.

**PRODUCTION REQUIREMENTS:**

- Always make real API calls to Jikan, AnimSchedule, Kitsu (when IDs are available), and Anime-Planet (when URLs are available)
- Never mock or simulate data - this is production-level enrichment
- Handle API rate limits and errors gracefully
- Save all API responses to temporary files for reproducibility
- Use proper error handling for failed API calls
- Follow the exact API endpoints and data structures

I am now ready to begin. Please provide the `offline_anime_data` JSON object.
