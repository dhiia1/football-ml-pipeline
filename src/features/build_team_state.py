"""
Team state snapshot stage.

Serving needs to answer "what does this team look like RIGHT NOW" for a
match that hasn't happened yet — but build_features.py only ever computes
features for FINISHED matches (correctly, since training needs known
outcomes). This script bridges that gap by producing two small artifacts:

1. team_state.json — each team's most recent known form + Elo rating,
   pulled from wherever they last appeared in the historical data. This
   is NOT new data collection — it's the same numbers already sitting in
   features.parquet, just extracted into a fast per-team lookup instead
   of scanning the whole historical table on every prediction request.

2. upcoming_fixtures.json — SCHEDULED matches from the raw snapshots,
   which matches_to_dataframe() throws away entirely (by design — training
   only wants FINISHED matches). Serving needs these to know what to
   predict on.

Run manually, after build_features.py:
    uv run python src/features/build_team_state.py

TODO once this works:
- h2h between two specific teams isn't precomputed here — serving will
  need to compute it on demand from the historical matches, since it
  depends on WHICH two teams are asked about, not a single team snapshot.
- Re-run this alongside build_features.py in the Prefect flow, since both
  depend on the same raw snapshots being fresh.
"""
import json
import yaml
import pandas as pd
from pathlib import Path

from src.features.build_features import (
    load_all_snapshots,
    matches_to_dataframe,
    add_rolling_form,
    add_elo_ratings,
)

ROOT = Path(__file__).resolve().parents[2]
CONFIG = yaml.safe_load(open(ROOT / "config" / "config.yaml"))


def extract_scheduled_matches(matches: list) -> list:
    """
    Pull SCHEDULED (not yet played) matches out of the raw match list —
    the exact opposite filter of matches_to_dataframe, which keeps only
    FINISHED ones.
    """
    scheduled = []
    for m in matches:
        if m.get("status") not in ("SCHEDULED", "TIMED"):
            continue
        scheduled.append({
            "match_id": m["id"],
            "date": m["utcDate"],
            "matchday": m.get("matchday"),
            "home_team": m["homeTeam"]["name"],
            "away_team": m["awayTeam"]["name"],
        })
    return sorted(scheduled, key=lambda x: x["date"])


def build_team_state(df: pd.DataFrame) -> dict:
    """
    For each team, take their MOST RECENT row (by date) across either the
    home or away side, and pull out whichever form/elo value corresponds
    to the side they played. This is "their rating as of today," derived
    from the same historical computation used for training — not a new
    calculation.
    """
    state = {}

    home_rows = df[["date", "home_team", "home_form", "home_elo"]].rename(
        columns={"home_team": "team", "home_form": "form", "home_elo": "elo"}
    )
    away_rows = df[["date", "away_team", "away_form", "away_elo"]].rename(
        columns={"away_team": "team", "away_form": "form", "away_elo": "elo"}
    )
    all_rows = pd.concat([home_rows, away_rows]).sort_values("date")

    latest = all_rows.groupby("team").last()

    for team, row in latest.iterrows():
        state[team] = {
            "form": float(row["form"]),
            "elo": float(row["elo"]),
            "as_of": row["date"].isoformat(),
        }

    return state


def main():
    raw_dir = ROOT / CONFIG["paths"]["raw_dir"]
    processed_dir = ROOT / CONFIG["paths"]["processed_dir"]

    all_matches = load_all_snapshots(raw_dir)

    finished_df = matches_to_dataframe(all_matches)
    finished_df = add_elo_ratings(finished_df)
    finished_df = add_rolling_form(finished_df, CONFIG["features"]["rolling_form_window"])

    team_state = build_team_state(finished_df)

    scheduled = extract_scheduled_matches(all_matches)

    # Backfilled seasons include teams no longer in the league this season
    # (relegated/promoted). Only teams with an UPCOMING fixture are
    # actually in the current season — filter team_state down to those,
    # so serving never predicts using stale data for a team that's left
    # the league.
    current_teams = {f["home_team"] for f in scheduled} | {f["away_team"] for f in scheduled}
    dropped = set(team_state) - current_teams
    if dropped:
        print(f"Dropping {len(dropped)} team(s) not in the current season: {sorted(dropped)}")
    team_state = {team: stats for team, stats in team_state.items() if team in current_teams}

    team_state_path = processed_dir / "team_state.json"
    with open(team_state_path, "w") as f:
        json.dump(team_state, f, indent=2)
    print(f"Saved current state for {len(team_state)} teams to {team_state_path}")

    fixtures_path = processed_dir / "upcoming_fixtures.json"
    with open(fixtures_path, "w") as f:
        json.dump(scheduled, f, indent=2)
    print(f"Saved {len(scheduled)} upcoming fixtures to {fixtures_path}")


if __name__ == "__main__":
    main()