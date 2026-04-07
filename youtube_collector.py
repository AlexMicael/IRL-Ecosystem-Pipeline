import os
import sys
import time
import random
import logging
import requests
import pandas as pd
from dotenv import load_dotenv
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

# --- CONFIGURATION ---
load_dotenv()

# Get Credentials
api_keys_str = os.getenv("YOUTUBE_API_KEYS")
API_KEYS = [key.strip() for key in api_keys_str.split(',')] if api_keys_str else []
PERSPECTIVE_API_KEY = os.getenv("PERSPECTIVE_API_KEY")

# --- FILE PATHS ---
MAP_FILE = "streamer_map.csv"
OUTPUT_VIDEOS_CSV = "youtube_videos_data.csv"
OUTPUT_COMMENTS_CSV = "youtube_comments_data.csv"
LOG_FILE = "youtube_collector.log"
LOCK_FILE = "youtube_collector.lock"

# --- LOGGING SETUP ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', filename=LOG_FILE, filemode='a')
logger = logging.getLogger(__name__)

if not API_KEYS:
    logger.critical("YOUTUBE_API_KEYS not found or empty in .env file.")
    sys.exit("Exiting: Missing API Keys.")
if not PERSPECTIVE_API_KEY:
    logger.warning("PERSPECTIVE_API_KEY not found. Toxicity scores will be None.")

# --- Custom exception for final quota error ---
class QuotaExceededError(Exception):
    """Raised when all available API keys have exhausted their quota."""
    pass

# --- API Key Manager ---
class YouTubeKeyManager:
    """Manages a list of API keys and switches them when a quota is exceeded."""
    def __init__(self, keys):
        self._keys = keys
        self._current_key_index = 0
        self.service = self._build_service()

    def _build_service(self):
        if self._current_key_index >= len(self._keys):
            logger.critical("All API keys have been exhausted.")
            return None
        key = self._keys[self._current_key_index]
        logger.info(f"Building YouTube service with API Key #{self._current_key_index + 1}")
        return build('youtube', 'v3', developerKey=key, cache_discovery=False)

    def get_next_key(self):
        """Switches to the next available API key."""
        logger.warning(f"API Key #{self._current_key_index + 1} has exceeded its quota.")
        self._current_key_index += 1
        self.service = self._build_service()
        return self.service is not None

# --- TOXICITY ANALYSIS FUNCTIONS ---
PERSPECTIVE_URL = "https://commentanalyzer.googleapis.com/v1alpha1/comments:analyze"

def get_toxicity_score(comment_text):
    """Fetch toxicity score using Google Perspective API."""
    if not comment_text or not PERSPECTIVE_API_KEY:
        return None
    retries = 5
    backoff = 1
    while retries > 0:
        try:
            data = {
                "comment": {"text": comment_text},
                "languages": ["en"],
                "requestedAttributes": {"TOXICITY": {}}
            }
            response = requests.post(
                f"{PERSPECTIVE_URL}?key={PERSPECTIVE_API_KEY}",
                json=data,
                timeout=5
            )
            response.raise_for_status()
            result = response.json()
            score = result["attributeScores"]["TOXICITY"]["summaryScore"]["value"]
            return round(score, 4)
        except requests.exceptions.HTTPError as e:
            if e.response is not None and e.response.status_code == 429:  # Rate Limited Error
                logger.warning(f"Perspective API rate-limited. Retrying in {backoff} seconds.")
                time.sleep(backoff + random.uniform(0, 1)) # Add Jitter
                retries -= 1
                backoff *= 2  # Exponential Backoff
            else:
                logger.error(f"Perspective API HTTP error: {e}")
                return None # Don't Retry on Other HTTP Errors
        except Exception as e:
            logger.error(f"Error during toxicity analysis: {e}")
            return None # Don't Retry on Other HTTP Errors
    logger.error("Perspective API rate limit exceeded after retries.")
    return None

# --- HELPER & DATA FETCHING FUNCTIONS ---
def load_target_channels():
    try:
        df = pd.read_csv(MAP_FILE)
        channel_ids = df['youtube_channel_id'].dropna().unique().tolist()
        logger.info(f"Loaded {len(channel_ids)} target channels from {MAP_FILE}.")
        return channel_ids
    except FileNotFoundError:
        logger.critical(f"Mapping file not found: {MAP_FILE}. Please create it.")
        return []

def load_processed_video_ids():
    """Loads video IDs from the VIDEOS CSV to prevent re-fetching them."""
    if not os.path.exists(OUTPUT_VIDEOS_CSV): return set()
    try:
        df = pd.read_csv(OUTPUT_VIDEOS_CSV, dtype={'video_id': str})
        return set(df['video_id'].unique())
    except (FileNotFoundError, pd.errors.EmptyDataError):
        return set()

def load_all_video_ids():
    """Helper to load all video IDs from the videos CSV."""
    return load_processed_video_ids()

def load_processed_comment_video_ids():
    """Loads video IDs from the COMMENTS CSV to find videos we've already processed."""
    if not os.path.exists(OUTPUT_COMMENTS_CSV): return set()
    try:
        df = pd.read_csv(OUTPUT_COMMENTS_CSV, dtype={'video_id': str})
        return set(df['video_id'].unique())
    except (FileNotFoundError, pd.errors.EmptyDataError):
        return set()

def get_recent_videos(key_manager, channel_id, processed_ids):
    """Fetches recent videos, with retry logic for API keys."""
    while True:
        try:
            res = key_manager.service.channels().list(id=channel_id, part='contentDetails').execute()
            
            if not res.get('items'):
                logger.warning(f"Channel not found or empty: {channel_id}. Skipping.")
                return []
            
            playlist_id = res['items'][0]['contentDetails']['relatedPlaylists']['uploads']
            
            videos = []
            res = key_manager.service.playlistItems().list(playlistId=playlist_id, part='snippet', maxResults=10).execute()
            
            for item in res.get('items', []):
                video_id = item['snippet']['resourceId']['videoId']
                if video_id not in processed_ids:
                    videos.append({'video_id': video_id, 'channel_id': item['snippet']['channelId'], 'published_at': item['snippet']['publishedAt'], 'video_title': item['snippet']['title'], 'video_description': item['snippet']['description']})
            return videos
        
        except HttpError as e:
            if 'quotaExceeded' in str(e):
                if not key_manager.get_next_key():
                    raise QuotaExceededError
            else:
                logger.error(f"HTTP error fetching videos for channel {channel_id}: {e}")
                return []

def get_video_comments(key_manager, video_id):
    """Fetches comments, adds toxicity score, and handles quota errors."""
    while True:
        try:
            comments = []
            res = key_manager.service.commentThreads().list(part='snippet', videoId=video_id, textFormat='plainText', maxResults=100).execute()
            
            comment_batch = [] # Collect All Comments First
            for item in res.get('items', []):
                comment_batch.append(item)

            if not comment_batch:
                logger.info(f"No comments found for video {video_id}.")
                return []
            
            for item in comment_batch:
                comment = item['snippet']['topLevelComment']['snippet']
                text = comment['textDisplay']
                toxicity = get_toxicity_score(text) # Get Score

                comments.append({
                    'video_id': video_id,
                    'comment_id': item['snippet']['topLevelComment']['id'],
                    'author_name': comment['authorDisplayName'],
                    'comment_text': text,
                    'published_at': comment['publishedAt'],
                    'like_count': comment['likeCount'],
                    'toxicity_score': toxicity
                })
                
            return comments
        except HttpError as e:
            if 'quotaExceeded' in str(e):
                if not key_manager.get_next_key():
                    raise QuotaExceededError
            elif "disabled comments" in str(e).lower() or "commentsDisabled" in str(e):
                logger.warning(f"Could not fetch comments for video {video_id} (comments disabled).")
                return [] # Return Empty List (Final State)
            else:
                logger.error(f"HTTP error fetching comments for video {video_id}: {e}")
                return [] # Return Empty List (Maybe Temporary Error)

def main():
    if os.path.exists(LOCK_FILE):
        logger.warning(f"Lockfile {LOCK_FILE} exists. Another instance is likely running. Exiting.")
        sys.exit("Process already running.")
        
    try:
        with open(LOCK_FILE, 'w') as f:
            f.write(str(os.getpid()))
        
        logger.info("--- Starting YouTube data collection job ---")
        
        key_manager = YouTubeKeyManager(API_KEYS)
        if not key_manager.service:
            sys.exit()

        target_channel_ids = load_target_channels()
        if not target_channel_ids:
            sys.exit("No channels to process. Exiting.")

        # Initialize List to Hold Temporary Comments
        all_comments = []
        
        try:
            # Find and Save NEW Videos
            processed_video_ids = load_processed_video_ids()
            
            new_videos = []
            for channel_id in target_channel_ids:
                logger.info(f"Checking for new videos in channel: {channel_id}")
                new_videos.extend(get_recent_videos(key_manager, channel_id, processed_video_ids))
            
            if not new_videos:
                logger.info("No new videos found.")
            else:
                logger.info(f"Found {len(new_videos)} new videos to save.")
                videos_df = pd.DataFrame(new_videos)
                file_exists = os.path.exists(OUTPUT_VIDEOS_CSV)
                videos_df.to_csv(OUTPUT_VIDEOS_CSV, mode='a', index=False, header=not file_exists, encoding='utf-8-sig')

            # Find Videos That Need Comments Processed (New + Backlog)
            logger.info("Checking for any videos that need comment processing...")
            all_video_ids = load_all_video_ids()
            processed_comment_video_ids = load_processed_comment_video_ids()
            
            videos_to_process = list(all_video_ids - processed_comment_video_ids)
            
            if not videos_to_process:
                logger.info("All videos have their comments processed. Nothing to backfill.")
            else:
                logger.info(f"Found {len(videos_to_process)} videos needing comment processing (this includes any new videos).")
                
                # Process Videos One-by-One
                for video_id in videos_to_process:
                    logger.info(f"Fetching comments for video ID: {video_id}")
                    video_comments = get_video_comments(key_manager, video_id)
                    
                    if video_comments:
                        all_comments.extend(video_comments)
                    
                    if len(all_comments) > 500:
                        comments_df = pd.DataFrame(all_comments)
                        file_exists = os.path.exists(OUTPUT_COMMENTS_CSV)
                        comments_df.to_csv(OUTPUT_COMMENTS_CSV, mode='a', index=False, header=not file_exists, encoding='utf-8-sig')
                        logger.info(f"Saved a batch of {len(all_comments)} comments.")
                        all_comments = []

        except QuotaExceededError:
            logger.warning("All YouTube API keys have been exhausted for this run. Exiting gracefully.")
        except Exception as e:
            logger.error(f"An unexpected error occurred in main: {e}", exc_info=True)
        finally:
            if all_comments:
                logger.info(f"Saving last batch of {len(all_comments)} comments before exiting...")
                comments_df = pd.DataFrame(all_comments)
                file_exists = os.path.exists(OUTPUT_COMMENTS_CSV)
                comments_df.to_csv(OUTPUT_COMMENTS_CSV, mode='a', index=False, header=not file_exists, encoding='utf-8-sig')

        logger.info("--- YouTube data collection job finished ---")

    finally:
        if os.path.exists(LOCK_FILE):
            os.remove(LOCK_FILE)
            logger.info("Removed lockfile.")

if __name__ == "__main__":
    main()