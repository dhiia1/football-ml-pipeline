"""
Historical backfill stage.

Pulls one or more PAST seasons for a competition and saves each as its own
dated, season-labeled snapshot. Run this once (or whenever you want to add
another season) — it's separate from fetch_football_data.py, which just
grabs the current season on a schedule.

Run manually, e.g. to backfill 2024-25 and 2025-26 for La Liga:
    uv run python src/ingestion/backfill_seasons.py --seasons 2024 2025

TODO once this works:
- If you hit the free-tier rate limit (10 req/min) with more seasons or
  competitions, the sleep between calls below already handles it, but bump
  SLEEP_SECONDS up if you see 429 responses.
"""
import os
import time
import json
import yaml
import argparse
import requests
from datetime import datetime, timezone
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parents[2]
CONFIG = yaml.safe_load(open(ROOT / "config" / "config.yaml"))

API_KEY = os.getenv("FOOTBALL_DATA_API_KEY")
BASE_URL = os.getenv("FOOTBALL_DATA_BASE_URL", "https://api.football-data.org/v4")
SLEEP_SECONDS = 7  # free tier is 10 req/min -> stay comfortably under that


def fetch_season(competition_code: str, season: int) -> dict:
    url = f"{BASE_URL}/competitions/{competition_code}/matches"
    headers = {"X-Auth-Token": API_KEY}
    params = {"season": season}
    response = requests.get(url, headers=headers, params=params, timeout=30)
    response.raise_for_status()
    return response.json()


def save_snapshot(data: dict, raw_dir: Path, season: int) -> Path:
    raw_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = raw_dir / f"matches_season{season}_{timestamp}.json"
    with open(out_path, "w") as f:
        json.dump(data, f)
    return out_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seasons", type=int, nargs="+", required=True,
        help="Season start years to backfill, e.g. --seasons 2024 2025"
    )
    args = parser.parse_args()

    if not API_KEY:
        raise RuntimeError(
            "FOOTBALL_DATA_API_KEY not set. Copy .env.example to .env and add your key."
        )

    raw_dir = ROOT / CONFIG["paths"]["raw_dir"]

    for i, season in enumerate(args.seasons):
        data = fetch_season(CONFIG["competition_code"], season)
        out_path = save_snapshot(data, raw_dir, season)
        n_matches = len(data.get("matches", []))
        print(f"Season {season}: saved {n_matches} matches to {out_path}")

        if i < len(args.seasons) - 1:
            time.sleep(SLEEP_SECONDS)


if __name__ == "__main__":
    main()