import os
import sys
import logging
import pandas as pd
from dotenv import load_dotenv
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

# --- CONFIGURATION ---
load_dotenv()
api_keys_str = os.getenv("YOUTUBE_API_KEYS")
API_KEYS = [key.strip() for key in api_keys_str.split(',')] if api_keys_str else []

# --- FILE PATHS ---
USERS_FILE = "twitch_users_data.csv"
MAP_FILE = "streamer_map.csv"
LOG_FILE = "map_creator.log"

# --- LOGGING SETUP ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', filename=LOG_FILE, filemode='w')
logger = logging.getLogger(__name__)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

if not API_KEYS:
    logger.critical("YOUTUBE_API_KEYS not found or empty in .env file.")
    sys.exit("Exiting: Missing API Keys.")

# --- Custom exception for final quota error ---
class QuotaExceededError(Exception):
    """Raised when all available API keys have exhausted their quota."""
    pass

# --- Key Manager Class ---
class YouTubeKeyManager:
    """Manages a list of API keys and rotates them when a quota is exceeded."""
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
        return build('youtube', 'v3', developerKey=key)

    def get_next_key(self):
        """Switches to the next available API key and rebuilds the service."""
        logger.warning(f"API Key #{self._current_key_index + 1} has exceeded its quota.")
        self._current_key_index += 1
        self.service = self._build_service()
        return self.service is not None

def find_youtube_channel(key_manager, streamer_name):
    """Searches for a YouTube channel and returns the top result, with key rotation."""
    while True:
        try:
            search_response = key_manager.service.search().list(
                q=streamer_name,
                part="snippet",
                maxResults=1,
                type="channel"
            ).execute()

            if not search_response.get("items"):
                logger.warning(f"No YouTube channel found for '{streamer_name}'.")
                return None
            
            top_result = search_response["items"][0]
            channel_id = top_result["id"]["channelId"]
            channel_title = top_result["snippet"]["title"]
            
            logger.info(f"Found channel for '{streamer_name}': '{channel_title}' (ID: {channel_id})")
            return channel_id
            
        except HttpError as e:
            if 'quotaExceeded' in str(e):
                if not key_manager.get_next_key():
                    raise QuotaExceededError
            else:
                logger.error(f"An HTTP error occurred while searching for '{streamer_name}': {e}")
                return None

def main():
    logger.info("--- Starting Streamer Map Creation Utility ---")
    
    try:
        users_df = pd.read_csv(USERS_FILE)
        streamers_to_map = users_df[['login_name', 'display_name']].drop_duplicates('login_name')
    except FileNotFoundError:
        logger.critical(f"Source file not found: {USERS_FILE}. Please run the Twitch collector first.")
        sys.exit()

    existing_mappings = set()
    if os.path.exists(MAP_FILE):
        map_df = pd.read_csv(MAP_FILE)
        existing_mappings = set(map_df['twitch_login_name'])
        logger.info(f"Loaded {len(existing_mappings)} existing mappings from {MAP_FILE}.")

    streamers_to_find = streamers_to_map[~streamers_to_map['login_name'].isin(existing_mappings)]
    
    if streamers_to_find.empty:
        logger.info("All streamers are already mapped. No new searches needed.")
        sys.exit()
    
    logger.info(f"Attempting to find YouTube channels for {len(streamers_to_find)} new streamers.")
    
    key_manager = YouTubeKeyManager(API_KEYS)
    if not key_manager.service:
        sys.exit()

    new_mappings = []
    try:
        for index, row in streamers_to_find.iterrows():
            channel_id = find_youtube_channel(key_manager, row['display_name'])
            if channel_id:
                new_mappings.append({
                    'twitch_login_name': row['login_name'],
                    'youtube_channel_id': channel_id
                })
    except QuotaExceededError:
        logger.warning("All YouTube API keys have been exhausted for the day. Saving progress and exiting.")
    
    if new_mappings:
        new_mappings_df = pd.DataFrame(new_mappings)
        file_exists = os.path.exists(MAP_FILE)
        new_mappings_df.to_csv(MAP_FILE, mode='a', index=False, header=not file_exists)
        logger.info(f"Successfully added {len(new_mappings)} new mappings to {MAP_FILE} before exiting.")
    else:
        logger.info("No new mappings were found in this run.")
        
    logger.info("--- Streamer Map Creation Finished ---")

if __name__ == "__main__":
    main()