"""
Orchestration stage.

Chains ingestion -> feature build -> train -> evaluate into one Prefect
flow. Run it manually first, then schedule it (e.g. daily, or right after
matchday) once you trust each stage individually.

Run manually:
    python pipelines/flow.py

Schedule it (once comfortable):
    prefect deployment build pipelines/flow.py:football_pipeline -n daily
    prefect deployment apply football_pipeline-deployment.yaml
    # then set a cron schedule in the Prefect UI or CLI

TODO once this works:
- Add the monitoring stage as a step after evaluate.
- Add failure handling: if ingestion fails (e.g. rate limit), the flow
  should retry rather than silently produce stale features.
- Parameterize the competition code so one flow can run for multiple
  leagues.
"""
from prefect import flow, task
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run_script(relative_path: str):
    result = subprocess.run(
        [sys.executable, str(ROOT / relative_path)],
        capture_output=True, text=True
    )
    print(result.stdout)
    if result.returncode != 0:
        print(result.stderr)
        raise RuntimeError(f"{relative_path} failed")


@task
def ingest():
    run_script("src/ingestion/fetch_football_data.py")


@task
def build_features():
    run_script("src/features/build_features.py")


@task
def train():
    run_script("src/training/train.py")


@task
def evaluate():
    run_script("src/evaluation/evaluate.py")


@flow(name="football-outcome-pipeline")
def football_pipeline():
    ingest()
    build_features()
    train()
    evaluate()


if __name__ == "__main__":
    football_pipeline()
