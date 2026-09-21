"""
Evaluation / promotion gate — now compares using the BACKTEST mean, not a
single noisy split.

Why this changed: a single chronological 80/20 split can make a genuinely
better model look worse (or a genuinely worse one look better) purely
because of which specific matches land in the test slice. This was
proven directly in this project — a bug-fixed model with a strictly
better backtest mean (0.505 vs 0.491) still lost a single-split
comparison to a champion whose 0.530 turned out to be a lucky draw, not
a stable measure of quality. train.py now logs backtest_accuracy_mean
alongside the single-split accuracy specifically so this script can make
a trustworthy comparison instead.

Backward compatibility: a champion promoted BEFORE this change won't have
backtest_accuracy_mean logged on its run. In that case, this script falls
back to the older "accuracy" metric for the champion side of the
comparison and prints a clear warning that the comparison isn't fully
apples-to-apples for that one transitional promotion.

Run manually:
    uv run python src/evaluation/evaluate.py
"""
import argparse
import yaml
from pathlib import Path
from mlflow.tracking import MlflowClient
import mlflow

ROOT = Path(__file__).resolve().parents[2]
CONFIG = yaml.safe_load(open(ROOT / "config" / "config.yaml"))

ALIAS = "production"
PROMOTION_METRIC = "backtest_accuracy_mean"
FALLBACK_METRIC = "accuracy"  # for champions promoted before backtest logging existed


def get_latest_run(client: MlflowClient, experiment_name: str):
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise RuntimeError(f"No experiment named '{experiment_name}' found — run train.py first.")
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["start_time DESC"],
        max_results=1,
    )
    if not runs:
        raise RuntimeError("Experiment exists but has no runs yet — run train.py first.")
    return runs[0]


def get_current_champion_metrics(client: MlflowClient, registry_name: str):
    """
    Returns (accuracy_value, metric_name_used, is_fallback) for whatever
    model currently holds the "production" alias, or (None, None, False)
    if nothing has ever been promoted yet.
    """
    try:
        current = client.get_model_version_by_alias(registry_name, ALIAS)
    except Exception:
        return None, None, False

    run = client.get_run(current.run_id)
    if PROMOTION_METRIC in run.data.metrics:
        return run.data.metrics[PROMOTION_METRIC], PROMOTION_METRIC, False
    elif FALLBACK_METRIC in run.data.metrics:
        return run.data.metrics[FALLBACK_METRIC], FALLBACK_METRIC, True
    else:
        return None, None, False


def promote(client: MlflowClient, run, registry_name: str):
    model_uri = f"runs:/{run.info.run_id}/model"
    model_version = mlflow.register_model(model_uri=model_uri, name=registry_name)
    client.set_registered_model_alias(
        name=registry_name, alias=ALIAS, version=model_version.version
    )
    print(f"Promoted version {model_version.version} to alias '{ALIAS}'.")

    export_path = ROOT / "model_export"
    if export_path.exists():
        import shutil
        shutil.rmtree(export_path)
    mlflow.artifacts.download_artifacts(
        artifact_uri=f"models:/{registry_name}@{ALIAS}",
        dst_path=str(export_path),
    )
    print(f"Exported production model to {export_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--force", action="store_true",
        help="Promote the latest run regardless of the comparison. Use only "
             "when you have separate, trustworthy evidence (e.g. a manual "
             "backtest run) that the standard automated comparison is "
             "misleading for a known, understood reason."
    )
    args = parser.parse_args()

    mlflow.set_tracking_uri("sqlite:///" + str(ROOT / "mlflow.db"))
    client = MlflowClient()

    latest_run = get_latest_run(client, CONFIG["model"]["experiment_name"])
    if PROMOTION_METRIC not in latest_run.data.metrics:
        raise RuntimeError(
            f"Latest run has no '{PROMOTION_METRIC}' metric — make sure "
            f"train.py's backtest logging ran successfully."
        )
    challenger_acc = latest_run.data.metrics[PROMOTION_METRIC]
    challenger_baseline = latest_run.data.metrics.get("backtest_baseline_mean")

    print(f"Challenger backtest mean accuracy: {challenger_acc:.3f}  "
          f"(run_id={latest_run.info.run_id})")
    if challenger_baseline is not None:
        print(f"Challenger backtest baseline mean: {challenger_baseline:.3f}")
        beats_floor = challenger_acc > challenger_baseline
    else:
        beats_floor = True  # older run without baseline logged — skip this check

    champion_acc, metric_used, is_fallback = get_current_champion_metrics(
        client, CONFIG["model"]["registry_name"]
    )

    if champion_acc is None:
        print("No production model exists yet — this would be the first promotion.")
        beats_champion = True
    else:
        if is_fallback:
            print(f"WARNING: current champion has no '{PROMOTION_METRIC}' logged "
                  f"(promoted before backtest logging existed). Falling back to "
                  f"comparing against its single-split '{metric_used}' = {champion_acc:.3f}. "
                  f"This comparison is NOT fully apples-to-apples.")
        else:
            print(f"Current production model backtest mean accuracy: {champion_acc:.3f}")
        beats_champion = challenger_acc > champion_acc

    if beats_floor and beats_champion:
        print("Challenger beats the floor and the current champion — promoting.")
        promote(client, latest_run, CONFIG["model"]["registry_name"])
    elif args.force:
        print("Challenger does NOT clear the bar, but --force was passed — promoting anyway.")
        promote(client, latest_run, CONFIG["model"]["registry_name"])
    elif not beats_floor:
        print("Challenger does NOT beat its own backtest baseline — holding. Something may be broken.")
    else:
        print("Challenger does NOT beat the current champion — holding.")


if __name__ == "__main__":
    main()