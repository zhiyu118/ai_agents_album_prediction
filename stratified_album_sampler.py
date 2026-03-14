"""
Stratified Album Sampler with RapidAPI Audio Features
Performs stratified sampling of albums by genre and prepares data for the workflow.
Fetches audio features from RapidAPI Track Analysis API.
"""

import json
import os
import shutil
from pathlib import Path
from collections import defaultdict
import random
import urllib.request
from typing import Dict, List, Set, Optional
import time
import re

# Parent genre mapping
PARENT_GENRES = {
    'pop': ['pop', 'pop rock', 'synth-pop', 'synthpop', 'electropop', 'indie pop',
            'dream pop', 'art pop', 'dance-pop', 'power pop', 'k-pop', 'j-pop',
            'britpop', 'bubblegum pop'],
    'hip-hop': ['hip-hop', 'hip hop', 'rap', 'trap', 'gangsta rap', 'conscious hip hop',
                'underground hip-hop', 'old school hip hop', 'east coast hip hop',
                'west coast hip hop', 'southern hip hop', 'alternative hip hop'],
    'rock': ['rock', 'hard rock', 'punk rock', 'punk', 'garage rock', 'blues rock',
             'folk rock', 'psychedelic rock', 'progressive rock', 'classic rock',
             'alternative rock', 'indie rock', 'post-rock', 'post-punk', 'new wave',
             'grunge', 'emo', 'metalcore', 'shoegaze', 'math rock', 'noise rock',
             'experimental rock', 'glam rock', 'art rock', 'atmospheric rock'],
    'electronic': ['electronic', 'techno', 'house', 'trance', 'dubstep', 'drum and bass',
                   'ambient', 'idm', 'edm', 'downtempo', 'chillout', 'breakbeat',
                   'electronica', 'electro', 'industrial', 'trip-hop', 'glitch',
                   'vaporwave', 'future bass', 'hardstyle', 'uk garage'],
    'r&b': ['r&b', 'rnb', 'rhythm and blues', 'soul', 'funk', 'neo-soul',
            'contemporary r&b', 'quiet storm', 'new jack swing', 'motown'],
    'indie': ['indie', 'indie folk', 'chamber pop', 'lo-fi', 'bedroom pop',
              'slowcore', 'sadcore', 'freak folk'],
    'jazz': ['jazz', 'bebop', 'cool jazz', 'free jazz', 'fusion', 'smooth jazz',
             'jazz fusion', 'contemporary jazz', 'avant-garde jazz', 'swing',
             'big band', 'bossa nova', 'latin jazz', 'ragtime'],
    'classical': ['classical', 'baroque', 'romantic', 'contemporary classical',
                  'opera', 'orchestral', 'chamber music', 'symphony', 'piano',
                  'choral', 'avant-garde classical', 'minimalism', 'neoclassical']
}

# Reverse mapping for quick lookup
GENRE_TO_PARENT = {}
for parent, genres in PARENT_GENRES.items():
    for genre in genres:
        GENRE_TO_PARENT[genre.lower()] = parent

# For genres not in the mapping, we'll need to search online
AMBIGUOUS_GENRES = set()


def classify_genre(genre_tags: List[str]) -> Set[str]:
    """
    Classify album into parent genre categories.
    Returns set of parent genres the album belongs to.
    """
    parent_genres = set()

    for tag in genre_tags:
        tag_lower = tag.lower().strip()

        # Direct match
        if tag_lower in GENRE_TO_PARENT:
            parent_genres.add(GENRE_TO_PARENT[tag_lower])
        else:
            # Check if tag contains any parent genre name
            for parent in PARENT_GENRES.keys():
                if parent in tag_lower or tag_lower in parent:
                    parent_genres.add(parent)
                    break
            else:
                # Mark for manual review if needed
                AMBIGUOUS_GENRES.add(tag_lower)

    return parent_genres


def load_album_data(json_folder: str) -> Dict[str, List[dict]]:
    """
    Load all albums and organize by parent genre.
    Returns dict mapping parent genre to list of album data.
    """
    albums_by_genre = defaultdict(list)
    json_path = Path(json_folder)

    print(f"Scanning albums in {json_folder}...")

    for json_file in json_path.glob("*.json"):
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Extract genre tags
            album_info = data.get('album_info', {}).get('album', {})
            genres = album_info.get('genres', [])

            if not genres:
                continue

            # Classify into parent genres
            parent_genres = classify_genre(genres)

            # Store album data with file path
            album_data = {
                'file_path': str(json_file),
                'data': data,
                'parent_genres': parent_genres
            }

            # Add to all matching parent genre buckets
            for parent_genre in parent_genres:
                albums_by_genre[parent_genre].append(album_data)

        except Exception as e:
            print(f"Error processing {json_file}: {e}")
            continue

    # Print statistics
    print("\nAlbum distribution by genre:")
    for genre in sorted(albums_by_genre.keys()):
        print(f"  {genre}: {len(albums_by_genre[genre])} albums")

    if AMBIGUOUS_GENRES:
        print(f"\nFound {len(AMBIGUOUS_GENRES)} ambiguous genre tags (may need manual classification)")
        print(f"Sample: {list(AMBIGUOUS_GENRES)[:10]}")

    return albums_by_genre


def stratified_sample(albums_by_genre: Dict[str, List[dict]],
                     samples_per_genre: int = 8,
                     seed: int = 42) -> Dict[str, List[dict]]:
    """
    Perform stratified sampling with RETRY POOL.
    Samples 3x the target to ensure we have enough albums for retries.
    """
    random.seed(seed)
    sampled = {}

    target_genres = ['pop', 'hip-hop', 'rock', 'electronic', 'r&b', 'indie', 'jazz', 'classical']

    # Sample 3x the target for retry pool (or all available, whichever is smaller)
    pool_size = samples_per_genre * 3

    print(f"\nPerforming stratified sampling (target: {samples_per_genre}, pool: {pool_size} per genre)...")

    for genre in target_genres:
        available = albums_by_genre.get(genre, [])

        if len(available) < samples_per_genre:
            print(f"WARNING: {genre} only has {len(available)} albums (need {samples_per_genre})")
            sampled[genre] = available
        elif len(available) < pool_size:
            print(f"  {genre}: using all {len(available)} available albums")
            sampled[genre] = available
        else:
            sampled[genre] = random.sample(available, pool_size)

        print(f"  {genre}: sampled {len(sampled[genre])} albums (pool for retries)")

    return sampled


def download_album_cover(image_urls: List[str], save_path: Path) -> bool:
    """
    Download album cover from URL list (tries highest quality first).
    """
    # Try to get the highest quality image (usually the last or 'mega' size)
    for url in reversed(image_urls):
        if url:
            try:
                urllib.request.urlretrieve(url, save_path)
                return True
            except Exception as e:
                print(f"    Failed to download from {url}: {e}")
                continue
    return False


def assign_popularity_tier(playcount: int) -> Dict:
    """
    Assign categorical popularity tier based on playcount.
    Uses industry-aligned brackets for better interpretability.

    Returns dict with tier, label, description, and raw playcount.
    """
    if playcount < 100_000:
        tier_info = {
            'tier': 1,
            'tier_label': 'Underground/Cult',
            'tier_description': 'Small dedicated fanbase, very limited reach',
            'playcount_range': '< 100K plays'
        }
    elif playcount < 1_000_000:
        tier_info = {
            'tier': 2,
            'tier_label': 'Indie Success',
            'tier_description': 'Moderate indie following, some streaming presence',
            'playcount_range': '100K - 1M plays'
        }
    elif playcount < 10_000_000:
        tier_info = {
            'tier': 3,
            'tier_label': 'Moderate Hit',
            'tier_description': 'Solid commercial success, genre recognition',
            'playcount_range': '1M - 10M plays'
        }
    elif playcount < 100_000_000:
        tier_info = {
            'tier': 4,
            'tier_label': 'Major Success',
            'tier_description': 'Mainstream hit, widespread recognition',
            'playcount_range': '10M - 100M plays'
        }
    else:
        tier_info = {
            'tier': 5,
            'tier_label': 'Cultural Phenomenon',
            'tier_description': 'Legendary status, multi-platinum level',
            'playcount_range': '> 100M plays'
        }

    tier_info['actual_playcount'] = playcount
    return tier_info


def get_rapidapi_client(api_key: str = None):
    """
    Get RapidAPI credentials for Track Analysis API.
    """
    if not api_key:
        api_key = os.environ.get('RAPIDAPI_KEY')

    if not api_key:
        print("WARNING: RapidAPI key not found.")
        return None

    return {
        'x-rapidapi-key': api_key,
        'x-rapidapi-host': 'track-analysis.p.rapidapi.com'
    }


def get_track_audio_features_rapidapi(headers: Dict, song: str, artist: str) -> Optional[Dict]:
    """
    Fetch audio features for a specific track using RapidAPI Track Analysis API.

    API endpoint: https://track-analysis.p.rapidapi.com/pktx/analysis
    Parameters: song, artist
    """
    if not headers:
        return None

    try:
        import requests
    except ImportError:
        print("      ERROR: requests library not installed. Run: pip install requests")
        return None

    try:
        url = "https://track-analysis.p.rapidapi.com/pktx/analysis"

        querystring = {
            "song": song,
            "artist": artist
        }

        response = requests.get(url, headers=headers, params=querystring, timeout=15)

        if response.status_code == 200:
            data = response.json()
            return data
        elif response.status_code == 404:
            return None
        elif response.status_code == 429:
            print(f"      Rate limited - waiting...")
            time.sleep(2)
            return None
        else:
            print(f"      API status {response.status_code}: {response.text[:100]}")
            return None

    except Exception as e:
        print(f"      Error: {e}")
        return None


def get_album_audio_features_rapidapi(headers: Dict, artist_name: str, album_name: str, track_list: List[str]) -> Optional[Dict]:
    """
    Fetch audio features for first 3 tracks in an album.
    Returns individual track features (for LLM to analyze) plus summary stats.
    """
    if not headers or not track_list:
        return None

    try:
        track_features_list = []

        # Analyze first 3 tracks
        tracks_to_analyze = track_list[:3] if len(track_list) > 3 else track_list

        for i, track_name in enumerate(tracks_to_analyze):
            clean_track = re.sub(r'\(.*?\)|\[.*?\]|feat\..*|ft\..*', '', track_name).strip()

            print(f"      Track {i+1}/3: {clean_track[:30]}...")

            features = get_track_audio_features_rapidapi(headers, clean_track, artist_name)

            if features:
                # Normalize all values to proper numeric types
                # REMOVE popularity to prevent data leakage (correlated with playcount)
                normalized = {}
                for key, value in features.items():
                    if key == 'popularity':
                        continue  # Skip popularity field
                    if value is not None:
                        try:
                            normalized[key] = float(value) if isinstance(value, str) else value
                        except (ValueError, TypeError):
                            normalized[key] = value
                    else:
                        normalized[key] = None

                track_features_list.append({
                    'track_name': track_name,
                    'track_number': i + 1,
                    **normalized  # Unpack features directly
                })

                energy = normalized.get('energy', 'N/A')
                tempo = normalized.get('tempo', 'N/A')
                print(f"        ✓ energy={energy}, tempo={tempo}")
            else:
                print(f"        ✗ No features")

            time.sleep(0.6)

        # STRICT REQUIREMENT: ALL 3 tracks must be retrieved
        if len(track_features_list) < 3:
            print(f"      ✗ Only {len(track_features_list)}/3 tracks retrieved - REJECTED")
            return None

        # VALIDATE: Each track must have complete audio features (no missing data)
        required_features = ['energy', 'tempo', 'danceability', 'acousticness',
                            'instrumentalness', 'speechiness', 'liveness']

        for track in track_features_list:
            missing_features = []
            for feature in required_features:
                value = track.get(feature)
                if value is None or (isinstance(value, str) and value.lower() == 'n/a'):
                    missing_features.append(feature)

            if missing_features:
                print(f"      ✗ Track {track.get('track_number')} missing features: {missing_features} - REJECTED")
                return None

        # Calculate summary statistics (for context, tracks are primary)
        # EXCLUDE popularity to prevent data leakage
        numeric_keys = ['danceability', 'energy', 'valence', 'acousticness',
                       'instrumentalness', 'speechiness', 'liveness',
                       'loudness', 'tempo']

        summary_stats = {}
        for key in numeric_keys:
            values = [t[key] for t in track_features_list if t.get(key) is not None and isinstance(t.get(key), (int, float))]

            if values:
                import statistics
                summary_stats[key] = {
                    'mean': round(statistics.mean(values), 2),
                    'min': round(min(values), 2),
                    'max': round(max(values), 2),
                    'range': round(max(values) - min(values), 2),
                    'std': round(statistics.stdev(values), 2) if len(values) > 1 else 0.0
                }

        result = {
            'tracks': track_features_list,  # PRIMARY: Individual tracks for LLM
            'summary_stats': summary_stats,  # Secondary: Diversity metrics
            'num_tracks_retrieved': len(track_features_list),
            'data_quality': 'complete'  # All 3 tracks with full features validated
        }

        print(f"      ✓ COMPLETE: 3/3 tracks with full audio features")
        return result

    except Exception as e:
        import traceback
        print(f"      ✗ Error: {e}")
        print(f"      Trace: {traceback.format_exc()[:150]}")
        return None


def format_audio_features_description(features: Dict) -> str:
    """
    Convert Spotify audio features to natural language description for LLM consumption.
    """
    if not features:
        return "Audio features not available for this album."

    # Energy interpretation
    energy = features.get('energy', 0)
    if energy > 0.7:
        energy_desc = "high energy and intensity"
    elif energy > 0.4:
        energy_desc = "moderate energy"
    else:
        energy_desc = "low energy and calm atmosphere"

    # Danceability interpretation
    dance = features.get('danceability', 0)
    if dance > 0.7:
        dance_desc = "highly danceable with strong rhythmic elements"
    elif dance > 0.4:
        dance_desc = "moderately danceable"
    else:
        dance_desc = "not particularly suited for dancing"

    # Valence interpretation (musical positivity)
    valence = features.get('valence', 0)
    if valence > 0.65:
        mood_desc = "positive, upbeat, and cheerful mood"
    elif valence > 0.35:
        mood_desc = "neutral emotional tone"
    else:
        mood_desc = "melancholic, somber, or introspective mood"

    # Acousticness interpretation
    acoustic = features.get('acousticness', 0)
    if acoustic > 0.6:
        acoustic_desc = "predominantly acoustic with minimal electronic production"
    elif acoustic > 0.3:
        acoustic_desc = "balanced mix of acoustic and electronic elements"
    else:
        acoustic_desc = "heavily produced with electronic instrumentation and effects"

    # Tempo interpretation
    tempo = features.get('tempo', 120)
    if tempo > 140:
        tempo_desc = f"fast tempo ({tempo:.0f} BPM), suggesting high-energy music"
    elif tempo > 100:
        tempo_desc = f"moderate tempo ({tempo:.0f} BPM), typical of popular music"
    else:
        tempo_desc = f"slow tempo ({tempo:.0f} BPM), creating a relaxed pace"

    # Compile description
    description = (
        f"This album exhibits {energy_desc} (energy: {energy:.2f}). "
        f"It is {dance_desc} (danceability: {dance:.2f}). "
        f"The overall emotional character conveys a {mood_desc} (valence: {valence:.2f}). "
        f"The production style is {acoustic_desc} (acousticness: {acoustic:.2f}). "
        f"The album has a {tempo_desc}. "
    )

    # Add instrumentalness if significant
    instrumental = features.get('instrumentalness', 0)
    if instrumental > 0.5:
        description += f"The album is predominantly instrumental (instrumentalness: {instrumental:.2f}), with minimal vocals. "

    # Add speechiness if significant
    speechiness = features.get('speechiness', 0)
    if speechiness > 0.33:
        description += f"Notable spoken word, rap, or vocal-heavy content is present (speechiness: {speechiness:.2f}). "

    # Add liveness if significant
    liveness = features.get('liveness', 0)
    if liveness > 0.6:
        description += f"The album has a live performance quality (liveness: {liveness:.2f}). "

    # Add loudness info
    loudness = features.get('loudness', -10)
    description += f"Average loudness is {loudness:.1f} dB. "

    # Add track count
    num_tracks = features.get('num_tracks', 0)
    description += f"The album contains {num_tracks} tracks."

    return description


def transform_to_workflow_format(album_data: dict, album_id: str, cover_relative_path: str,
                                spotify_features: Optional[Dict] = None) -> dict:
    """
    Transform raw album JSON to workflow format.
    """
    album_info = album_data.get('album_info', {}).get('album', {})

    # Extract metadata
    title = album_info.get('name', 'Unknown')
    artist = album_info.get('artist', 'Unknown')

    # Extract year from release_date (format: YYYY-MM-DD)
    release_date = album_data.get('release_date', '')
    release_year = int(release_date.split('-')[0]) if release_date and '-' in release_date else None

    # Get primary genre
    genres = album_info.get('genres', [])
    primary_genre = genres[0] if genres else 'Unknown'

    # Get description from wiki
    wiki = album_info.get('wiki', {})
    description = wiki.get('summary', '') or wiki.get('content', '')

    # Clean HTML tags from description if needed
    if description:
        import re
        description = re.sub(r'<[^>]+>', '', description)

    # Get listeners/playcount for ground truth
    listeners = int(album_info.get('listeners', 0))
    playcount = int(album_info.get('playcount', 0))

    # Assign categorical popularity tier (ground truth for prediction)
    popularity_tier = assign_popularity_tier(playcount)

    # Prepare audio features (NEW FORMAT: individual tracks + summary)
    if spotify_features:
        audio_features = spotify_features  # Contains 'tracks', 'summary_stats', 'num_tracks_retrieved'
    else:
        audio_features = {
            "tracks": [],
            "summary_stats": {},
            "num_tracks_retrieved": 0
        }

    # Create workflow format
    workflow_data = {
        "album_id": album_id,
        "metadata": {
            "title": title,
            "artist": artist,
            "release_year": release_year,
            "genre": primary_genre,
            "all_genres": genres,
            "description": description[:500] if description else f"Album by {artist}",  # Limit description length
            "listeners": listeners,
            "playcount": playcount
        },
        "audio_features": audio_features,
        "cover_path": cover_relative_path,
        "ground_truth": popularity_tier,  # Categorical tier (1-5)
        "original_mbid": album_data.get('mbid', ''),
        "url": album_info.get('url', '')
    }

    return workflow_data


def create_output_structure(sampled_albums: Dict[str, List[dict]],
                           output_folder: str,
                           json_folder: str,
                           fetch_audio_features: bool = True,
                           rapidapi_headers: Dict = None,
                           target_per_genre: int = 8):
    """
    Create output with RETRY LOGIC: keep sampling albums until we get target_per_genre with features per genre.
    """
    output_path = Path(output_folder)
    covers_path = output_path / "covers"

    output_path.mkdir(exist_ok=True)
    covers_path.mkdir(exist_ok=True)

    print(f"\nCreating output structure in {output_folder}...")
    print(f"Target: {target_per_genre} albums with features per genre")

    if fetch_audio_features and not rapidapi_headers:
        print("WARNING: Continuing without audio features.")
        fetch_audio_features = False

    all_albums = []
    album_counter = 1
    audio_success = 0
    audio_failed = 0

    for genre, albums in sampled_albums.items():
        print(f"\n{'='*70}")
        print(f"Processing {genre.upper()} albums")
        print(f"{'='*70}")

        successful_albums = []
        attempted_albums = []
        album_pool = albums.copy()  # Work with a copy

        while len(successful_albums) < target_per_genre and len(album_pool) > 0:
            # Get next album from pool
            album = album_pool.pop(0)
            album_data = album['data']
            album_info = album_data['album_info']['album']

            artist_name = album_info.get('artist', 'Unknown')
            album_name = album_info.get('name', 'Unknown')
            attempted_albums.append(f"{artist_name} - {album_name}")

            print(f"\n  [{len(successful_albums)+1}/{target_per_genre}] Trying: {artist_name} - {album_name}")

            # Download album cover
            image_urls = album_data.get('album_image_url', [])
            cover_relative_path = None
            if image_urls:
                album_id = f"{album_counter:03d}"
                cover_filename = f"album_{album_id}.jpg"
                cover_path = covers_path / cover_filename

                print(f"    - Downloading cover...")
                if download_album_cover(image_urls, cover_path):
                    cover_relative_path = f"covers/{cover_filename}"

            # Fetch audio features
            audio_features = None
            if fetch_audio_features and rapidapi_headers:
                print(f"    - Fetching audio features...")
                try:
                    tracks_data = album_info.get('tracks', {}).get('track', [])
                    track_names = [track['name'] for track in tracks_data if 'name' in track]

                    if track_names:
                        audio_features = get_album_audio_features_rapidapi(
                            rapidapi_headers,
                            artist_name,
                            album_name,
                            track_names
                        )

                        if audio_features:
                            print(f"    ✓ SUCCESS - Album #{len(successful_albums)+1} secured")
                            audio_success += 1

                            # Transform and save
                            album_id = f"{album_counter:03d}"
                            workflow_data = transform_to_workflow_format(
                                album_data,
                                album_id,
                                cover_relative_path,
                                audio_features
                            )

                            album_json_path = output_path / f"album_{album_id}.json"
                            with open(album_json_path, 'w', encoding='utf-8') as f:
                                json.dump(workflow_data, f, indent=2, ensure_ascii=False)

                            all_albums.append(workflow_data)
                            successful_albums.append(workflow_data)
                            album_counter += 1
                        else:
                            print(f"    ✗ FAILED - Trying next album...")
                            audio_failed += 1
                    else:
                        print(f"    ✗ No track list - Trying next album...")
                        audio_failed += 1

                    time.sleep(0.5)

                except Exception as e:
                    print(f"    ✗ ERROR: {e} - Trying next album...")
                    audio_failed += 1
            else:
                # No features requested, just save the album
                album_id = f"{album_counter:03d}"
                workflow_data = transform_to_workflow_format(
                    album_data,
                    album_id,
                    cover_relative_path,
                    None
                )

                album_json_path = output_path / f"album_{album_id}.json"
                with open(album_json_path, 'w', encoding='utf-8') as f:
                    json.dump(workflow_data, f, indent=2, ensure_ascii=False)

                all_albums.append(workflow_data)
                successful_albums.append(workflow_data)
                album_counter += 1

        print(f"\n{genre.upper()} complete: {len(successful_albums)}/{target_per_genre} albums")
        if len(successful_albums) < target_per_genre:
            print(f"  WARNING: Only got {len(successful_albums)} albums (ran out of candidates)")

    # Create master dataset
    master_path = output_path / "albums_dataset.json"
    with open(master_path, 'w', encoding='utf-8') as f:
        json.dump(all_albums, f, indent=2, ensure_ascii=False)

    print(f"\n{'=' * 70}")
    print(f"✓ Dataset created with {len(all_albums)} albums")
    print(f"✓ Output: {output_folder}")
    print(f"✓ Master: {master_path}")

    if fetch_audio_features:
        print(f"\nAudio Features Status:")
        print(f"  ✓ Successful: {audio_success} albums")
        print(f"  ✗ Failed: {audio_failed} albums")
        print(f"  Success rate: {audio_success / (audio_success + audio_failed) * 100:.1f}%")

    # Create summary
    summary = {
        "total_albums": len(all_albums),
        "target_per_genre": target_per_genre,
        "albums_per_genre": {
            genre: len([a for a in all_albums if a['metadata']['genre'] in sampled_albums[genre][0]['parent_genres']])
            for genre in sampled_albums.keys()
        },
        "audio_features": {
            "source": "RapidAPI Track Analysis",
            "success": audio_success,
            "failed": audio_failed
        } if fetch_audio_features else None
    }

    summary_path = output_path / "dataset_summary.json"
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)

    print(f"✓ Summary: {summary_path}")

    return all_albums


def main():
    """Main execution function."""

    # Configuration
    JSON_FOLDER = "/Users/river/Downloads/album_json"
    OUTPUT_FOLDER = "/Users/river/Desktop/MACS 37005/sampled_albums_with_audio"
    SAMPLES_PER_GENRE = 8
    RANDOM_SEED = 42
    FETCH_AUDIO_FEATURES = True  # Set to False to skip audio features

    print("=" * 70)
    print("STRATIFIED ALBUM SAMPLER WITH RAPIDAPI INTEGRATION")
    print("=" * 70)

    # Check for RapidAPI credentials if needed
    rapidapi_headers = None

    if FETCH_AUDIO_FEATURES:
        # Check environment variable first
        rapidapi_key = os.environ.get('RAPIDAPI_KEY')

        if not rapidapi_key:
            print("\n🎵 RapidAPI Track Analysis Credentials Required")
            print("=" * 70)
            print("\nTo fetch audio features and popularity scores, you need a RapidAPI key.")
            print("\nYou mentioned you have a subscription - please provide your API key.")
            print("\nAPI: Track Analysis API by SoundNet on RapidAPI")
            print("URL: https://rapidapi.com/soundnet/api/track-analysis")
            print("\n" + "=" * 70)
            print("\nOptions:")
            print("  1. Enter RapidAPI key now")
            print("  2. Skip audio features (will be set to null)")

            try:
                choice = input("\nEnter your choice (1 or 2): ").strip()

                if choice == '1':
                    print("\nPlease paste your RapidAPI key:")
                    rapidapi_key = input("  API Key: ").strip()

                    if not rapidapi_key:
                        print("\n❌ Invalid API key provided. Continuing without audio features.")
                        FETCH_AUDIO_FEATURES = False
                    else:
                        # Set environment variable for the session
                        os.environ['RAPIDAPI_KEY'] = rapidapi_key
                        rapidapi_headers = get_rapidapi_client(rapidapi_key)
                        print("\n✓ API key set successfully!")
                else:
                    print("\n⏭  Skipping audio features.")
                    FETCH_AUDIO_FEATURES = False
            except EOFError:
                print("\n⚠️  Running in non-interactive mode.")
                print("Please set RAPIDAPI_KEY environment variable and run again.")
                print("\nExample:")
                print("  export RAPIDAPI_KEY='your_key_here'")
                print("  python stratified_album_sampler.py")
                return
        else:
            rapidapi_headers = get_rapidapi_client(rapidapi_key)
            print("\n✓ RapidAPI key found in environment variables")

    # Step 1: Load all albums
    albums_by_genre = load_album_data(JSON_FOLDER)

    # Step 2: Check if we have enough albums in each genre
    target_genres = ['pop', 'hip-hop', 'rock', 'electronic', 'r&b', 'indie', 'jazz', 'classical']
    missing_genres = [g for g in target_genres if len(albums_by_genre.get(g, [])) < SAMPLES_PER_GENRE]

    if missing_genres:
        print(f"\nWARNING: Insufficient albums for genres: {missing_genres}")
        try:
            response = input("Continue anyway? (y/n): ")
            if response.lower() != 'y':
                print("Aborted.")
                return
        except EOFError:
            print("\nNon-interactive mode: continuing with available albums.")

    # Step 3: Perform stratified sampling
    sampled_albums = stratified_sample(albums_by_genre, SAMPLES_PER_GENRE, RANDOM_SEED)

    # Step 4: Create output structure with audio features and retry logic
    create_output_structure(
        sampled_albums,
        OUTPUT_FOLDER,
        JSON_FOLDER,
        fetch_audio_features=FETCH_AUDIO_FEATURES,
        rapidapi_headers=rapidapi_headers,
        target_per_genre=SAMPLES_PER_GENRE  # Will keep trying until we get 8 with features
    )

    print("\n" + "=" * 70)
    print("SAMPLING COMPLETE!")
    print("=" * 70)
    print(f"\nNext steps:")
    print(f"1. Review the sampled albums in: {OUTPUT_FOLDER}")
    print(f"2. Check albums_dataset.json for complete dataset with audio features")
    print(f"3. Use for Modal deployment with LangGraph")
    print(f"4. Individual album JSONs ready for parallel processing")

    if FETCH_AUDIO_FEATURES:
        print(f"\n💡 Tip: Audio features stored as individual tracks (up to 3 per album)")
        print(f"   Each album JSON contains 'tracks' array with features for each song")
        print(f"   Plus 'summary_stats' with mean/min/max/std for diversity analysis")


if __name__ == "__main__":
    main()
