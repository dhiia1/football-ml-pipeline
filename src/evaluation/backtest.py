"""
Rolling-origin backtest — the honest way to compare feature/model changes.

A single chronological 80/20 split (what train.py/evaluate.py use day to
day) is fast but noisy: whichever specific matches happen to land in that
one test slice can swing accuracy by several points on their own, with
nothing to do with whether a feature change actually helped. We proved
this directly — adding a season shifted the split boundary and changed
the accuracy reading with no feature even touched.

This script instead trains and evaluates across SEVERAL expanding
chronological windows (via sklearn's TimeSeriesSplit: fold 1 trains on the
earliest chunk and tests on the next chunk, fold 2 trains on everything
before an even later cutoff and tests on the next chunk after that, and so
on) and reports the AVERAGE across all folds. That average is a much more
stable number to compare feature/hyperparameter changes against than any
single split.

This is a diagnostic tool, not part of the promotion pipeline — it doesn't
register or promote anything in MLflow. Use it to decide whether a change
is worth keeping BEFORE running train.py/evaluate.py for real.

Run manually:
    uv run python src/evaluation/backtest.py

TODO once this works:
- Try different n_splits values — more folds = more stable estimate but
  smaller/noisier early folds; fewer folds = closer to the current
  single-split approach.
- Log summary results to MLflow as a separate "backtest" experiment, so
  backtest history is tracked the same way training runs are.
"""
import yaml
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, log_loss
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[2]
CONFIG = yaml.safe_load(open(ROOT / "config" / "config.yaml"))

FEATURE_COLS = ["home_form", "away_form", "h2h_home_advantage", "home_elo", "away_elo"]
N_SPLITS = 5


def baseline_accuracy(test_df: pd.DataFrame) -> float:
    always_home = ["H"] * len(test_df)
    return accuracy_score(test_df[CONFIG["model"]["target"]], always_home)


def main():
    processed_dir = ROOT / CONFIG["paths"]["processed_dir"]
    df = pd.read_parquet(processed_dir / "features.parquet").sort_values("date").reset_index(drop=True)

    label_encoder = LabelEncoder()
    y_all = label_encoder.fit_transform(df[CONFIG["model"]["target"]])

    tscv = TimeSeriesSplit(n_splits=N_SPLITS)

    accuracies, log_losses, baselines = [], [], []

    print(f"Running {N_SPLITS}-fold rolling-origin backtest on {len(df)} rows...\n")

    for fold, (train_idx, test_idx) in enumerate(tscv.split(df), start=1):
        train_df, test_df = df.iloc[train_idx], df.iloc[test_idx]
        X_train, X_test = train_df[FEATURE_COLS], test_df[FEATURE_COLS]
        y_train, y_test = y_all[train_idx], y_all[test_idx]

        model = XGBClassifier(
            n_estimators=30,
            max_depth=2,
            learning_rate=0.1,
            reg_alpha=1.0,
            reg_lambda=1.0,
            objective="multi:softprob",
            eval_metric="mlogloss",
        )
        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        probs = model.predict_proba(X_test)
        acc = accuracy_score(y_test, preds)
        loss = log_loss(y_test, probs)
        baseline = baseline_accuracy(test_df)

        accuracies.append(acc)
        log_losses.append(loss)
        baselines.append(baseline)

        date_range = f"{test_df['date'].min().date()} to {test_df['date'].max().date()}"
        print(f"Fold {fold}: train={len(train_df):4d}  test={len(test_df):4d}  "
              f"({date_range})  acc={acc:.3f}  baseline={baseline:.3f}  log_loss={loss:.3f}")

    print(f"\n{'='*60}")
    print(f"Mean accuracy:    {np.mean(accuracies):.3f}  (± {np.std(accuracies):.3f})")
    print(f"Mean baseline:    {np.mean(baselines):.3f}")
    print(f"Mean log loss:    {np.mean(log_losses):.3f}  (± {np.std(log_losses):.3f})")
    print(f"Folds beating baseline: {sum(a > b for a, b in zip(accuracies, baselines))}/{N_SPLITS}")


if __name__ == "__main__":
    main()