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
    fake_matches = [
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
    df = matches_to_dataframe(fake_matches)
    assert len(df) == 1
    assert df.iloc[0]["result"] == "H"


def test_chronological_ordering_preserved():
    fake_matches = [
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
    df = matches_to_dataframe(fake_matches)
    assert df.iloc[0]["date"] < df.iloc[1]["date"], "Rows must be sorted chronologically"


def test_dedup_keeps_latest_lastupdated(tmp_path):
    """Two snapshots with the same match id should dedupe to one, keeping
    whichever has the more recent lastUpdated timestamp."""
    from src.features.build_features import load_all_snapshots
    import json

    raw_dir = tmp_path
    older = {"matches": [{
        "id": 1, "status": "SCHEDULED", "utcDate": "2026-01-01T15:00:00Z",
        "homeTeam": {"name": "A"}, "awayTeam": {"name": "B"},
        "score": {"fullTime": {"home": None, "away": None}},
        "lastUpdated": "2026-01-01T10:00:00Z",
    }]}
    newer = {"matches": [{
        "id": 1, "status": "FINISHED", "utcDate": "2026-01-01T15:00:00Z",
        "homeTeam": {"name": "A"}, "awayTeam": {"name": "B"},
        "score": {"fullTime": {"home": 2, "away": 0}},
        "lastUpdated": "2026-01-01T17:00:00Z",
    }]}
    (raw_dir / "matches_a.json").write_text(json.dumps(older))
    (raw_dir / "matches_b.json").write_text(json.dumps(newer))

    merged = load_all_snapshots(raw_dir)
    assert len(merged) == 1
    assert merged[0]["status"] == "FINISHED"