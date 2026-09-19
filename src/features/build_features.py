"""
Feature building stage.

Turns raw match JSON into a leakage-safe feature table: for every match,
features are computed ONLY from matches that happened before it
chronologically. This is the single most important rule in this file —
break it and your model will look great and be useless.

Run manually:
    python src/features/build_features.py

TODO once this works:
- Add head-to-head (H2H) features: past results between these two teams.
- Add home/away split form (a team's home form vs away form differ a lot).
- Add rest days between matches (fixture congestion affects performance).
- Consider Elo-style rating as a single strong baseline feature.
"""

import json
import yaml
import pandas as pd
from pathlib import Path
from glob import glob
from collections import defaultdict

ROOT = Path(__file__).resolve().parents[2]
CONFIG = yaml.safe_load(open(ROOT / "config" / "config.yaml"))


def load_all_snapshots(raw_dir: Path) -> list:
    """
    Load every raw snapshot (current-season incremental pulls AND
    backfilled past seasons), merged into one flat list of match dicts.

    The same match can appear in more than one snapshot (e.g. two current-
    season pulls a week apart both contain last week's matches) — we
    dedupe by match id, keeping whichever copy has the most recent
    `lastUpdated` timestamp, since that one reflects the latest known score.
    """
    files = sorted(glob(str(raw_dir / "matches_*.json")))
    if not files:
        raise FileNotFoundError("No raw snapshots found — run ingestion first.")

    by_id = {}
    for file in files:
        with open(file) as f:
            data = json.load(f)
        for m in data.get("matches", []):
            match_id = m["id"]
            existing = by_id.get(match_id)
            if existing is None or m.get("lastUpdated", "") >= existing.get(
                "lastUpdated", ""
            ):
                by_id[match_id] = m

    return list(by_id.values())


def matches_to_dataframe(matches: list) -> pd.DataFrame:
    rows = []
    for m in matches:
        if m.get("status") != "FINISHED":
            continue  # only finished matches have a known result to train on
        score = m["score"]["fullTime"]
        home_goals, away_goals = score["home"], score["away"]
        if home_goals is None or away_goals is None:
            continue
        result = (
            "H"
            if home_goals > away_goals
            else ("A" if away_goals > home_goals else "D")
        )
        rows.append(
            {
                "match_id": m["id"],
                "date": m["utcDate"],
                "home_team": m["homeTeam"]["name"],
                "away_team": m["awayTeam"]["name"],
                "home_goals": home_goals,
                "away_goals": away_goals,
                "result": result,
            }
        )
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").reset_index(drop=True)


def add_rolling_form(df: pd.DataFrame, window: int) -> pd.DataFrame:
    """
    For each match, compute each team's average points-per-game over their
    last `window` matches BEFORE this match's date. This loop is simple
    and O(n^2)-ish on purpose — get correctness first, optimize later.
    """
    df = df.copy()
    df["home_form"] = None
    df["away_form"] = None

    def points(row, team):
        if row["home_team"] == team:
            g_for, g_against = row["home_goals"], row["away_goals"]
        else:
            g_for, g_against = row["away_goals"], row["home_goals"]
        if g_for > g_against:
            return 3
        elif g_for == g_against:
            return 1
        return 0

    for idx, row in df.iterrows():
        for side, team in [
            ("home_form", row["home_team"]),
            ("away_form", row["away_team"]),
        ]:
            past = df[
                (df["date"] < row["date"])
                & ((df["home_team"] == team) | (df["away_team"] == team))
            ].tail(window)
            if len(past) < CONFIG["features"]["min_matches_for_prediction"]:
                df.at[idx, side] = None
                continue
            pts = past.apply(lambda r: points(r, team), axis=1)
            df.at[idx, side] = pts.mean()

    return df.dropna(subset=["home_form", "away_form"])


def add_head_to_head(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each match, computes the home team's average points-per-game from
    PAST meetings against this exact opponent (regardless of which side
    they were on back then) — strictly before this match's date, same
    leakage-safe rule as add_rolling_form.

    Unlike rolling form, head-to-head history is sparse (two teams might
    only meet once or twice a season), so instead of dropping matches with
    no prior meetings, this fills them with 0.5 — a neutral "no history,
    assume even" prior — rather than losing rows you can't afford to lose.
    """
    df = df.copy()
    df["h2h_home_advantage"] = None

    def points(g_for, g_against):
        if g_for > g_against:
            return 3
        elif g_for == g_against:
            return 1
        return 0

    for idx, row in df.iterrows():
        home_team, away_team = row["home_team"], row["away_team"]
        past_meetings = df[
            (df["date"] < row["date"])
            & (
                ((df["home_team"] == home_team) & (df["away_team"] == away_team))
                | ((df["home_team"] == away_team) & (df["away_team"] == home_team))
            )
        ]

        if past_meetings.empty:
            df.at[idx, "h2h_home_advantage"] = 0.5  # no history — neutral prior
            continue

        pts = []
        for _, m in past_meetings.iterrows():
            if m["home_team"] == home_team:
                pts.append(points(m["home_goals"], m["away_goals"]))
            else:
                pts.append(points(m["away_goals"], m["home_goals"]))
        df.at[idx, "h2h_home_advantage"] = sum(pts) / len(pts) / 3  # normalize to 0-1

    return df


from collections import defaultdict


def add_elo_ratings(
    df: pd.DataFrame,
    k: float = 20,
    base_rating: float = 1500,
    home_advantage: float = 100,
) -> pd.DataFrame:
    """
    Computes each team's Elo rating as it enters every match — a single
    number reflecting long-run team strength, derived purely from actual
    results in this dataset (no external opinion injected).

    Core idea: after each match, the winner gains rating points and the
    loser loses them, but the SIZE of the swing depends on how surprising
    the result was (beating a much stronger team gains a lot; beating a
    much weaker team gains almost nothing). This naturally separates elite
    teams from mid-table ones over time, purely from results — unlike
    home_form, it has memory across the ENTIRE history, not just 5 games.

    `home_advantage` (added only when computing the expected outcome, not
    stored in the rating itself) reflects that home teams win more often
    even between equally-rated sides — a standard Elo-for-football
    adjustment.

    Requires df to already be sorted chronologically (matches_to_dataframe
    already does this) — ratings are computed as a running single pass,
    so out-of-order rows would corrupt the sequence.

    New/promoted teams with no prior matches simply start at `base_rating`
    — correct behavior, not a bug: they haven't earned a rating yet.
    """
    df = df.copy()
    ratings = defaultdict(lambda: base_rating)
    home_elos, away_elos = [], []

    for _, row in df.iterrows():
        home, away = row["home_team"], row["away_team"]
        home_elo, away_elo = ratings[home], ratings[away]

        # Record the rating EACH TEAM ENTERED this match with — this is
        # the leakage-safe part: we use pre-match ratings as the feature,
        # then update AFTER, so this match's own result never influences
        # its own feature values.
        home_elos.append(home_elo)
        away_elos.append(away_elo)

        expected_home = 1 / (1 + 10 ** ((away_elo - (home_elo + home_advantage)) / 400))

        home_goals, away_goals = row["home_goals"], row["away_goals"]
        if home_goals > away_goals:
            actual_home = 1.0
        elif home_goals == away_goals:
            actual_home = 0.5
        else:
            actual_home = 0.0

        ratings[home] = home_elo + k * (actual_home - expected_home)
        ratings[away] = away_elo + k * ((1 - actual_home) - (1 - expected_home))

    df["home_elo"] = home_elos
    df["away_elo"] = away_elos
    return df


def main():
    raw_dir = ROOT / CONFIG["paths"]["raw_dir"]
    processed_dir = ROOT / CONFIG["paths"]["processed_dir"]
    processed_dir.mkdir(parents=True, exist_ok=True)

    raw = load_all_snapshots(raw_dir)
    df = matches_to_dataframe(raw)
    df = add_rolling_form(df, CONFIG["features"]["rolling_form_window"])
    df = add_head_to_head(df)
    df = add_elo_ratings(df)
    out_path = processed_dir / "features.parquet"
    df.to_parquet(out_path, index=False)
    print(f"Saved {len(df)} feature rows to {out_path}")


if __name__ == "__main__":
    main()
