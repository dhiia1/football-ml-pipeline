"""
Serving stage.

Three endpoints, all built on the same idea: take an upcoming (SCHEDULED)
fixture, assemble the exact feature vector the model was trained on using
each team's CURRENT known state (team_state.json) plus on-demand
head-to-head history, and return a decoded, human-readable prediction.

    GET /round/{matchday}       -> predictions for every scheduled match
                                    in that matchday
    GET /game?home=X&away=Y     -> prediction for one specific matchup
                                    (works even for a hypothetical pairing
                                    not currently on the fixture list)
    GET /team/{team_name}/next  -> prediction for that team's next
                                    scheduled match, whichever side they're on

Run:
    uv run uvicorn src.serving.app:app --reload

TODO once this works:
- Persist the LabelEncoder from train.py instead of hardcoding the A/D/H
  mapping below — it happens to be stable (sklearn sorts classes
  alphabetically, and all 3 always appear in training data), but relying
  on that is fragile if it ever changes.
- MODEL_URI loading assumes the production model is XGBoost specifically
  (mlflow.xgboost.load_model) — would break if a different model type
  (e.g. LogisticRegression) ever gets promoted instead.
- Team name matching is exact-string only (case-insensitive) — no fuzzy
  matching for typos or alternate spellings (e.g. "Barca" vs "FC Barcelona").
- These artifacts (team_state.json, upcoming_fixtures.json,
  historical matches) are loaded once at startup — add a /reload endpoint
  so a fresh pipeline run doesn't require restarting the whole app.
"""
import os
import json
import yaml
import mlflow
import mlflow.xgboost
import pandas as pd
from pathlib import Path
from fastapi import FastAPI, HTTPException, Query

from src.features.build_features import load_all_snapshots, matches_to_dataframe

ROOT = Path(__file__).resolve().parents[2]
CONFIG = yaml.safe_load(open(ROOT / "config" / "config.yaml"))

from fastapi.middleware.cors import CORSMiddleware

mlflow.set_tracking_uri("sqlite:///" + str(ROOT / "mlflow.db"))

app = FastAPI(title="Football Outcome Predictor")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://football-ml-pipeline.onrender.com"],
    #allow_origins=["http://localhost:3000"]
    allow_methods=["GET"],
    allow_headers=["*"],
)

REGISTRY_NAME = CONFIG["model"]["registry_name"]
ALIAS = "production"
MODEL_URI = f"models:/{REGISTRY_NAME}@{ALIAS}"

# Matches the FEATURE_COLS order in train.py — must stay in sync.
FEATURE_ORDER = ["home_form", "away_form", "h2h_home_advantage", "home_elo", "away_elo"]

# sklearn's LabelEncoder sorts classes alphabetically: A, D, H -> 0, 1, 2.
# See TODO above about persisting this properly instead of hardcoding it.
LABEL_MAP = {0: "Away Win", 1: "Draw", 2: "Home Win"}
MODEL_PATH = os.getenv("MODEL_PATH")

_model = None
_team_state = None
_upcoming_fixtures = None
_historical_df = None


def get_model():
    global _model
    if _model is None:
        source = MODEL_PATH if MODEL_PATH else MODEL_URI
        try:
            _model = mlflow.xgboost.load_model(source)
        except Exception as e:
            raise RuntimeError(
                f"Could not load model from '{source}'. Has a model been "
                f"promoted yet? Run src/evaluation/evaluate.py first. "
                f"Original error: {e}"
            )
    return _model


def get_team_state():
    global _team_state
    if _team_state is None:
        path = ROOT / CONFIG["paths"]["processed_dir"] / "team_state.json"
        if not path.exists():
            raise RuntimeError("team_state.json not found — run build_team_state.py first.")
        with open(path) as f:
            _team_state = json.load(f)
    return _team_state


def get_upcoming_fixtures():
    global _upcoming_fixtures
    if _upcoming_fixtures is None:
        path = ROOT / CONFIG["paths"]["processed_dir"] / "upcoming_fixtures.json"
        if not path.exists():
            raise RuntimeError("upcoming_fixtures.json not found — run build_team_state.py first.")
        with open(path) as f:
            _upcoming_fixtures = json.load(f)
    return _upcoming_fixtures


def get_historical_df():
    """
    Loaded once for on-demand head-to-head lookups. This is the same raw
    match history build_features.py uses — h2h is pairwise, so unlike
    form/Elo it can't be precomputed into a flat per-team snapshot.
    """
    global _historical_df
    if _historical_df is None:
        raw_dir = ROOT / CONFIG["paths"]["raw_dir"]
        matches = load_all_snapshots(raw_dir)
        _historical_df = matches_to_dataframe(matches)
    return _historical_df


def compute_h2h(historical_df: pd.DataFrame, home_team: str, away_team: str) -> float:
    """
    Same formula as add_head_to_head in build_features.py: average points
    (win=3, draw=1, loss=0) the "home" side earned across ALL past meetings
    between these two teams (regardless of which side they were on back
    then), normalized to 0-1. Defaults to 0.5 (neutral) if they've never
    met in the available history.
    """
    past = historical_df[
        ((historical_df["home_team"] == home_team) & (historical_df["away_team"] == away_team)) |
        ((historical_df["home_team"] == away_team) & (historical_df["away_team"] == home_team))
    ]
    if past.empty:
        return 0.5

    def points(g_for, g_against):
        if g_for > g_against:
            return 3
        elif g_for == g_against:
            return 1
        return 0

    pts = []
    for _, m in past.iterrows():
        if m["home_team"] == home_team:
            pts.append(points(m["home_goals"], m["away_goals"]))
        else:
            pts.append(points(m["away_goals"], m["home_goals"]))
    return (sum(pts) / len(pts)) / 3


def find_team(team_state: dict, team_name: str) -> str:
    """Case-insensitive lookup — returns the canonical name as stored."""
    for name in team_state:
        if name.lower() == team_name.lower():
            return name
    raise HTTPException(status_code=404, detail=f"No current data for team '{team_name}'.")


def predict_match(home_team: str, away_team: str) -> dict:
    team_state = get_team_state()
    home_key = find_team(team_state, home_team)
    away_key = find_team(team_state, away_team)

    home_stats = team_state[home_key]
    away_stats = team_state[away_key]
    h2h = compute_h2h(get_historical_df(), home_key, away_key)

    features = pd.DataFrame([{
        "home_form": home_stats["form"],
        "away_form": away_stats["form"],
        "h2h_home_advantage": h2h,
        "home_elo": home_stats["elo"],
        "away_elo": away_stats["elo"],
    }])[FEATURE_ORDER]

    model = get_model()
    pred_class = int(model.predict(features)[0])
    proba = model.predict_proba(features)[0].tolist()

    return {
        "home_team": home_key,
        "away_team": away_key,
        "prediction": LABEL_MAP[pred_class],
        "probabilities": {LABEL_MAP[i]: round(p, 3) for i, p in enumerate(proba)},
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/teams")
def list_teams():
    return {"teams": sorted(get_team_state().keys())}


@app.get("/round/{matchday}")
def predict_round(matchday: int):
    fixtures = [f for f in get_upcoming_fixtures() if f["matchday"] == matchday]
    if not fixtures:
        raise HTTPException(status_code=404, detail=f"No scheduled fixtures found for matchday {matchday}.")

    results = []
    for fixture in fixtures:
        try:
            prediction = predict_match(fixture["home_team"], fixture["away_team"])
        except HTTPException:
            continue  # skip fixtures where a team has no current state yet (e.g. new promotion)
        prediction["date"] = fixture["date"]
        prediction["matchday"] = fixture["matchday"]
        results.append(prediction)

    return {"matchday": matchday, "predictions": results}


@app.get("/game")
def predict_game(
    home: str = Query(..., description="Home team name"),
    away: str = Query(..., description="Away team name"),
):
    return predict_match(home, away)

@app.get("/current-matchday")
def current_matchday():
    fixtures = get_upcoming_fixtures()
    if not fixtures:
        raise HTTPException(status_code=404, detail="No upcoming fixtures found.")
    return {"matchday": fixtures[0]["matchday"]}

@app.get("/team/{team_name}/next")
def predict_team_next(team_name: str):
    team_state = get_team_state()
    canonical = find_team(team_state, team_name)

    fixtures = [
        f for f in get_upcoming_fixtures()
        if f["home_team"] == canonical or f["away_team"] == canonical
    ]
    if not fixtures:
        raise HTTPException(status_code=404, detail=f"No scheduled fixtures found for '{canonical}'.")

    next_fixture = fixtures[0]  # already sorted by date in build_team_state.py
    prediction = predict_match(next_fixture["home_team"], next_fixture["away_team"])
    prediction["date"] = next_fixture["date"]
    prediction["matchday"] = next_fixture["matchday"]
    return prediction