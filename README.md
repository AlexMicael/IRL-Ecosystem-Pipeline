# IRL Streaming Ecosystem — Data Collection Pipeline
**Authors:** Alex Chen Hsieh & Derek Li  
**Course:** CS 415: Social Media Data Science Pipelines (Binghamton University)

## Overview
This repository contains the automated data collection pipeline powering the
[IRL Streaming Ecosystem Dashboard](https://alexmicael-irl-ecosystem-dashboard-app-wgmceg.streamlit.app/).
It continuously ingests live stream metadata from Twitch and video/comment data
from YouTube, then enriches each comment with a toxicity score via the Google
Perspective API — all managed by `cron` on a university-provided VM.

Over a 15-week semester, the pipeline collected:
- **455,535** Twitch stream snapshots across **5,597** unique streamers
- **56,291** YouTube videos and **1,527,771** comments with toxicity scores

🔗 [Dashboard Repo](https://github.com/AlexMicael/IRL-Ecosystem-Dashboard)

## Pipeline Architecture
| Script | What it does |
|---|---|
| `twitch_collector.py` | Fetches the top 100 live IRL streams from the Twitch Helix API. Appends time-series snapshots to `twitch_streams_data.csv` and new streamer profiles to `twitch_users_data.csv`. |
| `map_creator.py` | Reads new streamers from `twitch_users_data.csv` and uses the YouTube Search API to find their channels, saving results to `streamer_map.csv`. |
| `youtube_collector.py` | Polls mapped channels for new videos, fetches comments via the YouTube Data API v3, and scores each comment for toxicity using the Perspective API. Uses a lock file to prevent concurrent cron collisions and fails gracefully on quota exhaustion. |

## Output Files

| File | Description |
|---|---|
| `twitch_streams_data.csv` | Time-series snapshots of live IRL streams (viewer count, title, duration, etc.) |
| `twitch_users_data.csv` | Streamer profile metadata |
| `streamer_map.csv` | Twitch username → YouTube channel ID mappings |
| `youtube_videos_data.csv` | Video metadata for all mapped channels |
| `youtube_comments_data.csv` | Comments with Perspective API toxicity scores |

## Setup & Usage

### Prerequisites
- Python 3.x with [Conda](https://docs.conda.io/)
- API credentials for:
  - [Twitch Helix API](https://dev.twitch.tv/docs/api/)
  - [YouTube Data API v3](https://developers.google.com/youtube/v3)
  - [Google Perspective API](https://developers.perspectiveapi.com/)

### Installation

1. **Clone the repository:**
```sh
    git clone https://github.com/AlexMicael/IRL-Ecosystem-Pipeline.git
    cd IRL-Ecosystem-Pipeline
```

2. **Create and activate the Conda environment:**
```sh
    conda env create -f environment.yml
    conda activate irl-pipeline
```

3. **Configure API credentials:**
```sh
    cp .env.example .env
    # Fill in your API keys in .env
```

4. **Run a script manually (optional):**
```sh
    python twitch_collector.py
    python youtube_collector.py
    python map_creator.py
```

### Automating with cron

Add the following entries via `crontab -e`:
```cron
# Twitch — every 15 minutes
*/15 * * * * /path/to/conda/envs/irl-pipeline/bin/python /path/to/twitch_collector.py

# YouTube — every 4 hours
0 */4 * * * /path/to/conda/envs/irl-pipeline/bin/python /path/to/youtube_collector.py

# Streamer mapping — daily at 3 AM
0 3 * * * /path/to/conda/envs/irl-pipeline/bin/python /path/to/map_creator.py
```

## Implementation Notes

- **API quota management:** The YouTube collector rotates across multiple API keys and exits gracefully on quota exhaustion, resuming from where it left off on the next run.
- **Deduplication:** Streamer IDs are read with `dtype=str` to prevent type-mismatch duplicates across collection runs.
- **Backfill logic:** Videos collected before toxicity scoring was integrated are retroactively processed by the YouTube collector on each run until the backlog is cleared.
- **Security:** All API keys are stored in a `.env` file via `python-dotenv` and excluded from version control.

## Acknowledgments
This work was supported by Professor Yang at Binghamton University. We also thank the Computer Science Department for providing the virtual machine resources used to run this pipeline.