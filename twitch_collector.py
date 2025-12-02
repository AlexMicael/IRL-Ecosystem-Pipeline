import os
import sys
import logging
import requests
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv

# --- CONFIGURATION ---
load_dotenv()

# Get Credentials
CLIENT_ID = os.getenv("TWITCH_CLIENT_ID")
CLIENT_SECRET = os.getenv("TWITCH_CLIENT_SECRET")

# Categories to Track
TARGET_CATEGORIES = ["Just Chatting", "IRL", "Travel & Outdoors"]

# Collected Data Output
OUTPUT_STREAMS_CSV = "twitch_streams_data.csv"
OUTPUT_USERS_CSV = "twitch_users_data.csv"
LOG_FILE = "twitch_collector.log"

# --- LOGGING SETUP ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename=LOG_FILE,
    filemode='a'
)

logger = logging.getLogger(__name__)

if not CLIENT_ID or not CLIENT_SECRET:
    logger.critical("CLIENT_ID or CLIENT_SECRET not found in .env file. Please create a .env file.")
    sys.exit("Please create a .env file with your credentials.")

# --- HELPER FUNCTIONS ---
def get_twitch_access_token():
    """Acquires an OAuth access token from the Twitch API."""
    auth_url = "https://id.twitch.tv/oauth2/token"
    params = {
        'client_id': CLIENT_ID,
        'client_secret': CLIENT_SECRET,
        'grant_type': 'client_credentials'
    }

    try:
        response = requests.post(auth_url, params=params)
        response.raise_for_status()
        return response.json()['access_token']
    except requests.exceptions.RequestException as e:
        logger.error(f"Error getting access token: {e}")
        return None

def load_processed_ids(file_path, id_column):
    """Loads already processed IDs from a CSV file to avoid re-fetching data."""
    if not os.path.exists(file_path):
        return set()
    
    try:
        df = pd.read_csv(file_path, dtype={id_column: str})

        if id_column not in df.columns:
            return set()
            
        return set(df[id_column].unique())
    except (FileNotFoundError, pd.errors.EmptyDataError):
        return set()

# --- DATA FETCHING FUNCTIONS ---
def get_category_ids(category_names, token):
    """Gets the official Twitch IDs for a list of category names."""
    search_url = "https://api.twitch.tv/helix/games"
    headers = {'Client-ID': CLIENT_ID, 'Authorization': f'Bearer {token}'}
    params = {'name': category_names}

    try:
        response = requests.get(search_url, headers=headers, params=params)
        response.raise_for_status()
        data = response.json()['data']
        return [item['id'] for item in data]
    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching category IDs: {e}")
        return []

def fetch_live_streams(token, category_ids):
    """Fetches current top 100 streams based on categories provided."""
    streams_url = "https://api.twitch.tv/helix/streams"
    headers = {'Client-ID': CLIENT_ID, 'Authorization': f'Bearer {token}'}
    params = {'game_id': category_ids, 'first': 100}

    try:
        response = requests.get(streams_url, headers=headers, params=params)
        response.raise_for_status()
        streams_data = response.json()['data']
        collection_timestamp = datetime.now().isoformat()
        
        processed_data = [
            {
                'collection_timestamp': collection_timestamp,
                'user_id': stream.get('user_id'),
                'user_name': stream.get('user_name'),
                'stream_id': stream.get('id'),
                'stream_title': stream.get('title'),
                'viewer_count': stream.get('viewer_count'),
                'category_id': stream.get('game_id'),
                'category_name': stream.get('game_name'),
                'language': stream.get('language'),
                'started_at': stream.get('started_at'),
                'tags': stream.get('tags', [])
            } for stream in streams_data
        ]
        return pd.DataFrame(processed_data)
    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching stream data: {e}")
        return pd.DataFrame()

def fetch_user_info(token, user_ids):
    """Fetches profile information for a list of user IDs."""
    if not user_ids:
        return pd.DataFrame()
    
    users_url = "https://api.twitch.tv/helix/users"
    headers = {'Client-ID': CLIENT_ID, 'Authorization': f'Bearer {token}'}
    params = {'id': user_ids}

    try:
        response = requests.get(users_url, headers=headers, params=params)
        response.raise_for_status()
        users_data = response.json()['data']
        processed_users = [
            {
                'user_id': user.get('id'), 'login_name': user.get('login'),
                'display_name': user.get('display_name'), 'description': user.get('description'),
                'profile_image_url': user.get('profile_image_url'), 'created_at': user.get('created_at')
            } for user in users_data
        ]
        return pd.DataFrame(processed_users)
    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching user data: {e}")
        return pd.DataFrame()

# --- MAIN EXECUTION ---
def main():
    logger.info("--- Starting data collection job ---")
    
    access_token = get_twitch_access_token()
    if not access_token:
        logger.critical("Failed to get access token. Exiting.")
        sys.exit()
        
    category_ids_to_track = get_category_ids(TARGET_CATEGORIES, access_token)
    if not category_ids_to_track:
        logger.critical("Failed to get category IDs. Exiting.")
        sys.exit()
    
    # Fetch Current Live Streams
    live_streams_df = fetch_live_streams(access_token, category_ids_to_track)
    if live_streams_df.empty:
        logger.info("No live stream data fetched in this run.")
    else:
        file_exists = os.path.exists(OUTPUT_STREAMS_CSV)
        live_streams_df.to_csv(OUTPUT_STREAMS_CSV, mode='a', index=False, header=not file_exists, encoding='utf-8-sig')
        logger.info(f"Appended {len(live_streams_df)} records to {OUTPUT_STREAMS_CSV}")

    # Identify New Streamers
    processed_user_ids = load_processed_ids(OUTPUT_USERS_CSV, 'user_id')
    
    # Get User Info for New Users
    new_user_ids = list(live_streams_df[~live_streams_df['user_id'].isin(processed_user_ids)]['user_id'].unique())
    if new_user_ids:
        users_df = fetch_user_info(access_token, new_user_ids)
        if not users_df.empty:
            file_exists = os.path.exists(OUTPUT_USERS_CSV)
            users_df.to_csv(OUTPUT_USERS_CSV, mode='a', index=False, header=not file_exists, encoding='utf-8-sig')
            logger.info(f"Fetched and saved info for {len(users_df)} new users.")

    logger.info("--- Data collection job finished ---")

if __name__ == "__main__":
    main()