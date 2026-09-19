"""
Orchestration stage.

Chains ingestion -> feature build -> train -> evaluate -> monitor into one
Prefect flow. Run it manually first, then schedule it (e.g. daily, or
right after matchday) once you trust each stage individually.

Run manually:
    uv run python pipelines/flow.py

Schedule it (once comfortable):
    prefect deployment build pipelines/flow.py:football_pipeline -n daily
    prefect deployment apply football_pipeline-deployment.yaml
    # then set a cron schedule in the Prefect UI or CLI

TODO once this works:
- Parameterize the competition code so one flow can run for multiple
  leagues.
- Replace the naive drift_check.py split (see its own TODO) with a
  meaningful one, then decide what the flow should DO when drift is
  detected (e.g. trigger a Slack/email alert, or force a retrain even
  if evaluate.py would otherwise hold).
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


@task(retries=2, retry_delay_seconds=15)
def ingest():
    # Retries handle transient issues like the free-tier rate limit
    # (10 req/min) or a momentary network blip — not a fix for a bad
    # API key or a genuinely broken request, which will just fail 3
    # times and then raise as before.
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


@task
def monitor():
    run_script("src/monitoring/drift_check.py")


@flow(name="football-outcome-pipeline")
def football_pipeline():
    ingest()
    build_features()
    train()
    evaluate()
    monitor()


if __name__ == "__main__":
    football_pipeline()