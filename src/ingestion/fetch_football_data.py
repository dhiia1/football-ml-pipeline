"""
Ingestion stage.

Pulls finished + upcoming matches for one competition from football-data.org
and saves a dated, immutable raw snapshot. Never overwrite old snapshots —
that's what makes results reproducible (you can always answer "what did the
data look like when I trained model X").

Run manually:
    python src/ingestion/fetch_football_data.py

TODO once this works:
- Handle pagination if a competition returns more matches than one page.
- Add retry/backoff for rate limits (free tier is limited requests/min).
- Consider also pulling standings/team data for extra features.
"""
import os
import json
import yaml
import requests
from datetime import datetime, timezone
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parents[2]
CONFIG = yaml.safe_load(open(ROOT / "config" / "config.yaml"))

API_KEY = os.getenv("FOOTBALL_DATA_API_KEY")
BASE_URL = os.getenv("FOOTBALL_DATA_BASE_URL", "https://api.football-data.org/v4")


def fetch_matches(competition_code: str) -> dict:
    url = f"{BASE_URL}/competitions/{competition_code}/matches"
    headers = {"X-Auth-Token": API_KEY}
    response = requests.get(url, headers=headers, timeout=30)
    response.raise_for_status()
    return response.json()


def save_snapshot(data: dict, raw_dir: Path) -> Path:
    raw_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = raw_dir / f"matches_{timestamp}.json"
    with open(out_path, "w") as f:
        json.dump(data, f)
    return out_path


def main():
    if not API_KEY:
        raise RuntimeError(
            "FOOTBALL_DATA_API_KEY not set. Copy .env.example to .env and add your key."
        )

    raw_dir = ROOT / CONFIG["paths"]["raw_dir"]
    data = fetch_matches(CONFIG["competition_code"])
    out_path = save_snapshot(data, raw_dir)

    n_matches = len(data.get("matches", []))
    print(f"Saved {n_matches} matches to {out_path}")


if __name__ == "__main__":
    main()
