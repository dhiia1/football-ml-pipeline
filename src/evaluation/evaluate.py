"""
Evaluation / promotion gate — now wired to MLflow directly.

Pulls the most recent training run's accuracy from MLflow (instead of you
reading train.py's terminal output), computes the same baseline as before,
and if the model wins, registers it in the MLflow Model Registry and points
the "production" alias at it. Serving (app.py) always loads whatever the
"production" alias currently points to — so promoting here is what makes
a new model live, with no file copying or manual wiring.

Note: this uses aliases, not the older "stages" (Staging/Production) API —
MLflow deprecated stages in 2.9 in favor of aliases + tags, which are more
flexible (an alias is just a named pointer to any version, not a fixed
4-stage lifecycle).

Run manually:
    uv run python src/evaluation/evaluate.py

TODO once this works:
- Track baseline accuracy over time too, and log it to MLflow as a metric
  on each run — useful once the league's competitiveness shifts season to
  season.
- Add a log_loss comparison too, not just accuracy — a model could win on
  accuracy while being worse-calibrated (see PROMOTION_METRIC below).
"""
import argparse
import yaml
import pandas as pd
from pathlib import Path
from sklearn.metrics import accuracy_score
from mlflow.tracking import MlflowClient
import mlflow

ROOT = Path(__file__).resolve().parents[2]
CONFIG = yaml.safe_load(open(ROOT / "config" / "config.yaml"))

ALIAS = "production"
PROMOTION_METRIC = "accuracy"  # the metric name we compare against baseline


def baseline_accuracy(test_df: pd.DataFrame) -> float:
    always_home = ["H"] * len(test_df)
    return accuracy_score(test_df[CONFIG["model"]["target"]], always_home)


def get_latest_run(client: MlflowClient, experiment_name: str):
    """
    Return the most recent run in the experiment, ordered by start time.
    This is "the run we just produced with train.py", not necessarily the
    best one ever — the promotion decision below is what decides if it's
    good enough to become the new production model.
    """
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise RuntimeError(
            f"No experiment named '{experiment_name}' found — run train.py first."
        )
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["start_time DESC"],
        max_results=1,
    )
    if not runs:
        raise RuntimeError("Experiment exists but has no runs yet — run train.py first.")
    return runs[0]


def get_current_champion_accuracy(client: MlflowClient, registry_name: str) -> float | None:
    """
    Return the accuracy of whatever model currently holds the "production"
    alias — the real bar a new model needs to clear. Returns None if
    nothing has ever been promoted yet (first-ever promotion just needs to
    clear the sanity floor below).
    """
    try:
        current = client.get_model_version_by_alias(registry_name, ALIAS)
    except Exception:
        return None  # no registered model / no alias set yet — first promotion
    run = client.get_run(current.run_id)
    return run.data.metrics.get(PROMOTION_METRIC)


def promote(client: MlflowClient, run, registry_name: str):
    """
    Register the run's model artifact as a new version of `registry_name`,
    then point the "production" alias at it. If a version is already
    aliased "production", this simply moves the pointer — the old version
    still exists in the registry (nothing is deleted), just no longer
    aliased.
    """
    model_uri = f"runs:/{run.info.run_id}/model"
    model_version = mlflow.register_model(model_uri=model_uri, name=registry_name)
    client.set_registered_model_alias(
        name=registry_name, alias=ALIAS, version=model_version.version
    )
    print(f"Promoted version {model_version.version} to alias '{ALIAS}'.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--force", action="store_true",
        help="Promote the latest run regardless of baseline. For testing the "
             "registry/serving wiring only — never use this for a real decision."
    )
    args = parser.parse_args()

    processed_dir = ROOT / CONFIG["paths"]["processed_dir"]
    df = pd.read_parquet(processed_dir / "features.parquet")

    test_frac = 0.2
    split_idx = int(len(df) * (1 - test_frac))
    test_df = df.sort_values("date").iloc[split_idx:]
    baseline_acc = baseline_accuracy(test_df)

    mlflow.set_tracking_uri("sqlite:///" + str(ROOT / "mlflow.db"))
    client = MlflowClient()

    latest_run = get_latest_run(client, CONFIG["model"]["experiment_name"])
    model_acc = latest_run.data.metrics.get(PROMOTION_METRIC)
    if model_acc is None:
        raise RuntimeError(
            f"Latest run has no '{PROMOTION_METRIC}' metric logged — check train.py."
        )

    print(f"Baseline ('always home win') accuracy: {baseline_acc:.3f}")
    print(f"Latest run accuracy: {model_acc:.3f}  (run_id={latest_run.info.run_id})")

    beats_floor = model_acc > baseline_acc
    champion_acc = get_current_champion_accuracy(client, CONFIG["model"]["registry_name"])

    if champion_acc is not None:
        print(f"Current production model accuracy: {champion_acc:.3f}")
        beats_champion = model_acc > champion_acc
    else:
        print("No production model exists yet — this would be the first promotion.")
        beats_champion = True  # nothing to beat yet, floor check alone decides

    if beats_floor and beats_champion:
        reason = "beats baseline and current production model" if champion_acc is not None else "beats baseline (first promotion)"
        print(f"Model {reason} — promoting.")
        promote(client, latest_run, CONFIG["model"]["registry_name"])
    elif args.force:
        print("Model does NOT clear the bar, but --force was passed — promoting anyway (TEST ONLY).")
        promote(client, latest_run, CONFIG["model"]["registry_name"])
    elif not beats_floor:
        print("Model does NOT beat the baseline sanity floor — holding. Something may be broken.")
    else:
        print("Model beats baseline but does NOT beat the current production model — holding.")


if __name__ == "__main__":
    main()