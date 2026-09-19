"""
Monitoring stage.

Compares the feature distribution of new (reference) data against the data
the current model was trained on. A big shift (e.g. a new season starting,
major transfers, a team's form swinging wildly) is a signal the model may
need retraining even if you haven't measured accuracy drop yet.

This is what closes the loop: deployment isn't the finish line, it's the
start of a feedback cycle.

Run manually (after you have at least two feature snapshots to compare):
    python src/monitoring/drift_check.py

TODO once this works:
- Also log actual outcomes vs predictions once matches finish, and compute
  rolling accuracy — that's a more direct signal than feature drift alone.
- Wire this into the orchestrated pipeline (pipelines/flow.py) so it runs
  automatically after each ingestion cycle.
"""
import yaml
import pandas as pd
from pathlib import Path
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

ROOT = Path(__file__).resolve().parents[2]
CONFIG = yaml.safe_load(open(ROOT / "config" / "config.yaml"))

FEATURE_COLS = ["home_form", "away_form"]


def main():
    processed_dir = ROOT / CONFIG["paths"]["processed_dir"]
    df = pd.read_parquet(processed_dir / "features.parquet")

    # Naive split for demo purposes: first half = "reference" (what the
    # model was trained on), second half = "current" (recent matches).
    # TODO: replace with an actual "training window vs last N days" split.
    midpoint = len(df) // 2
    reference = df.iloc[:midpoint][FEATURE_COLS]
    current = df.iloc[midpoint:][FEATURE_COLS]

    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=reference, current_data=current)

    out_path = ROOT / "data" / "processed" / "drift_report.html"
    report.save_html(str(out_path))
    print(f"Drift report saved to {out_path} — open it in a browser.")


if __name__ == "__main__":
    main()
