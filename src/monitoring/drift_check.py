"""
Monitoring stage.

Compares the feature distribution of new (current) data against the data
the current model was trained on (reference). A big shift (e.g. a new
season starting, major transfers, a team's form swinging wildly) is a
signal the model may need retraining even if you haven't measured an
accuracy drop yet.

This is what closes the loop: deployment isn't the finish line, it's the
start of a feedback cycle.

Note: Evidently's API changed significantly in recent versions. This uses
the current top-level `evidently.Report` / `evidently.presets` structure,
not the older `evidently.report.Report` / `evidently.metric_preset` paths
you may see in older tutorials.

Run manually:
    uv run python src/monitoring/drift_check.py

TODO once this works:
- Also log actual outcomes vs predictions once matches finish, and compute
  rolling accuracy — that's a more direct signal than feature drift alone.
- Replace the naive halfway split below with something meaningful, e.g.
  "all data used to train the current production model" vs "matches from
  the last 2 weeks" — that actually simulates "did the world change since
  training," rather than just splitting the dataset in half.
- Wire this into pipelines/flow.py as a step after evaluate.
"""
import yaml
import pandas as pd
from pathlib import Path
from evidently import Report
from evidently.presets import DataDriftPreset

ROOT = Path(__file__).resolve().parents[2]
CONFIG = yaml.safe_load(open(ROOT / "config" / "config.yaml"))

FEATURE_COLS = ["home_form", "away_form"]


def main():
    processed_dir = ROOT / CONFIG["paths"]["processed_dir"]
    df = pd.read_parquet(processed_dir / "features.parquet")

    # Naive split for demo purposes: first half = "reference" (what the
    # model was trained on), second half = "current" (recent matches).
    # See the TODO above — this should become a more meaningful split.
    midpoint = len(df) // 2
    reference = df.iloc[:midpoint][FEATURE_COLS]
    current = df.iloc[midpoint:][FEATURE_COLS]

    report = Report([DataDriftPreset()], include_tests=True)
    my_eval = report.run(current, reference)

    out_path = ROOT / "data" / "processed" / "drift_report.html"
    my_eval.save_html(str(out_path))
    print(f"Drift report saved to {out_path} — open it in a browser.")


if __name__ == "__main__":
    main()