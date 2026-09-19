"""
Basic sanity tests. These aren't exhaustive — the point is to have SOME
automated gate before merging changes, which is what CI/CD actually means
in an ML context (it's rarely about testing model accuracy in CI; it's
about testing that the pipeline code still runs and produces sane shapes).

Run:
    pytest tests/
"""
import pandas as pd
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.features.build_features import matches_to_dataframe


def test_matches_to_dataframe_basic_shape():
    fake_raw = {
        "matches": [
            {
                "id": 1,
                "status": "FINISHED",
                "utcDate": "2026-01-01T15:00:00Z",
                "homeTeam": {"name": "Team A"},
                "awayTeam": {"name": "Team B"},
                "score": {"fullTime": {"home": 2, "away": 1}},
            },
            {
                "id": 2,
                "status": "SCHEDULED",  # should be excluded — no result yet
                "utcDate": "2026-01-08T15:00:00Z",
                "homeTeam": {"name": "Team A"},
                "awayTeam": {"name": "Team C"},
                "score": {"fullTime": {"home": None, "away": None}},
            },
        ]
    }
    df = matches_to_dataframe(fake_raw)
    assert len(df) == 1
    assert df.iloc[0]["result"] == "H"


def test_chronological_ordering_preserved():
    fake_raw = {
        "matches": [
            {
                "id": 1, "status": "FINISHED", "utcDate": "2026-02-01T15:00:00Z",
                "homeTeam": {"name": "A"}, "awayTeam": {"name": "B"},
                "score": {"fullTime": {"home": 1, "away": 1}},
            },
            {
                "id": 2, "status": "FINISHED", "utcDate": "2026-01-01T15:00:00Z",
                "homeTeam": {"name": "A"}, "awayTeam": {"name": "B"},
                "score": {"fullTime": {"home": 0, "away": 2}},
            },
        ]
    }
    df = matches_to_dataframe(fake_raw)
    assert df.iloc[0]["date"] < df.iloc[1]["date"], "Rows must be sorted chronologically"
