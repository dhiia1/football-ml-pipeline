"""
Training stage.

Trains a classifier on the processed feature table, logs it to MLflow, and
ALSO runs a rolling-origin backtest (multiple chronological train/test
splits, not just one) to get a trustworthy mean accuracy — logged as
separate metrics on the same run.

Why both: a single chronological split is fast and gives you the actual
deployable model artifact, but it's noisy — which specific matches land
in the test slice can swing accuracy by several points with nothing to do
with whether the model is actually better. The backtest mean is what
evaluate.py uses to decide whether to promote; the single-split model
here is still what actually gets deployed (retraining 5 separate models
for every promotion isn't practical — the backtest is for JUDGING quality,
the single split is for PRODUCING the artifact).

Run manually:
    uv run python src/training/train.py

TODO once this works:
- Try tuning XGBoost's hyperparameters further and compare backtest means
  across runs in the MLflow UI (`mlflow ui`).
- Add more features to the feature table and re-run.
"""
import yaml
import numpy as np
import mlflow
import mlflow.xgboost
import pandas as pd
from pathlib import Path
from xgboost import XGBClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, log_loss
from sklearn.preprocessing import LabelEncoder

ROOT = Path(__file__).resolve().parents[2]
CONFIG = yaml.safe_load(open(ROOT / "config" / "config.yaml"))

FEATURE_COLS = ["home_form", "away_form", "h2h_home_advantage", "home_elo", "away_elo"]
MODEL_PARAMS = dict(max_depth=2, n_estimators=30, reg_alpha=1.0, reg_lambda=1.0, learning_rate=0.1)
N_BACKTEST_SPLITS = 5


def chronological_split(df: pd.DataFrame, test_frac: float = 0.2):
    df = df.sort_values("date")
    split_idx = int(len(df) * (1 - test_frac))
    return df.iloc[:split_idx], df.iloc[split_idx:]


def run_backtest(df: pd.DataFrame, target_col: str):
    """
    Rolling-origin backtest: N expanding chronological folds, each trained
    and evaluated independently. Returns per-fold accuracies and baseline
    accuracies, for the caller to summarize.
    """
    df = df.sort_values("date").reset_index(drop=True)
    label_encoder = LabelEncoder()
    y_all = label_encoder.fit_transform(df[target_col])

    tscv = TimeSeriesSplit(n_splits=N_BACKTEST_SPLITS)
    accs, baselines = [], []

    for train_idx, test_idx in tscv.split(df):
        train_df, test_df = df.iloc[train_idx], df.iloc[test_idx]
        X_train, X_test = train_df[FEATURE_COLS], test_df[FEATURE_COLS]
        y_train, y_test = y_all[train_idx], y_all[test_idx]

        model = XGBClassifier(**MODEL_PARAMS, objective="multi:softprob", eval_metric="mlogloss")
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        accs.append(accuracy_score(y_test, preds))

        always_home = ["H"] * len(test_df)
        baselines.append(accuracy_score(test_df[target_col], always_home))

    return accs, baselines


def main():
    processed_dir = ROOT / CONFIG["paths"]["processed_dir"]
    df = pd.read_parquet(processed_dir / "features.parquet")

    train_df, test_df = chronological_split(df)

    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(train_df[CONFIG["model"]["target"]])
    y_test = label_encoder.transform(test_df[CONFIG["model"]["target"]])
    X_train = train_df[FEATURE_COLS]
    X_test = test_df[FEATURE_COLS]

    mlflow.set_tracking_uri("sqlite:///" + str(ROOT / "mlflow.db"))
    mlflow.set_experiment(CONFIG["model"]["experiment_name"])

    with mlflow.start_run():
        # The deployable artifact: one model on one chronological split.
        model = XGBClassifier(**MODEL_PARAMS, objective="multi:softprob", eval_metric="mlogloss")
        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        probs = model.predict_proba(X_test)
        acc = accuracy_score(y_test, preds)
        loss = log_loss(y_test, probs)

        # The trustworthy quality signal: mean over several chronological
        # splits. This is what evaluate.py uses to decide on promotion.
        backtest_accs, backtest_baselines = run_backtest(df, CONFIG["model"]["target"])
        backtest_mean = float(np.mean(backtest_accs))
        backtest_std = float(np.std(backtest_accs))
        backtest_baseline_mean = float(np.mean(backtest_baselines))

        mlflow.log_param("model_type", "XGBClassifier")
        mlflow.log_param("features", FEATURE_COLS)
        mlflow.log_param("train_size", len(train_df))
        mlflow.log_param("test_size", len(test_df))
        mlflow.log_params({f"model_{k}": v for k, v in MODEL_PARAMS.items()})
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("log_loss", loss)
        mlflow.log_metric("backtest_accuracy_mean", backtest_mean)
        mlflow.log_metric("backtest_accuracy_std", backtest_std)
        mlflow.log_metric("backtest_baseline_mean", backtest_baseline_mean)
        mlflow.xgboost.log_model(model, name="model")

        print(f"Single-split accuracy: {acc:.3f}  |  Log loss: {loss:.3f}")
        print(f"Backtest mean accuracy: {backtest_mean:.3f} (±{backtest_std:.3f})  "
              f"vs baseline mean: {backtest_baseline_mean:.3f}")
        print("Run logged to MLflow — inspect with: mlflow ui")


if __name__ == "__main__":
    main()