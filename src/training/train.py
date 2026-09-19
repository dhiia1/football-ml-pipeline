"""
Training stage.

Trains a classifier on the processed feature table and logs the run to
MLflow: params, metrics, and the model artifact itself. This is what
"experiment tracking" buys you in practice — a queryable history of every
run instead of one overwritten notebook.

CRITICAL: the split is chronological (train on earlier matches, test on
later ones), never random. Random shuffling here would leak future team
form into predictions for earlier matches.

Run manually:
    python src/training/train.py

TODO once this works:
- Swap LogisticRegression for GradientBoosting/XGBoost and compare runs
  in the MLflow UI (`mlflow ui`).
- Add more features to the feature table and re-run — that's the workflow
  you'll repeat dozens of times, and MLflow is what makes it not chaos.
- Try a rolling-origin backtest (multiple chronological splits) instead of
  one split, for a more honest accuracy estimate.
"""
from os import name

import yaml
import mlflow
import mlflow.sklearn
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from sklearn.preprocessing import LabelEncoder

ROOT = Path(__file__).resolve().parents[2]
CONFIG = yaml.safe_load(open(ROOT / "config" / "config.yaml"))

FEATURE_COLS = ["home_form", "away_form"]  # TODO: expand as you add features


def chronological_split(df: pd.DataFrame, test_frac: float = 0.2):
    df = df.sort_values("date")
    split_idx = int(len(df) * (1 - test_frac))
    return df.iloc[:split_idx], df.iloc[split_idx:]


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
        model = LogisticRegression(max_iter=1000)
        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        probs = model.predict_proba(X_test)
        acc = accuracy_score(y_test, preds)
        loss = log_loss(y_test, probs)

        mlflow.log_param("model_type", "LogisticRegression")
        mlflow.log_param("features", FEATURE_COLS)
        mlflow.log_param("train_size", len(train_df))
        mlflow.log_param("test_size", len(test_df))
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("log_loss", loss)
        mlflow.sklearn.log_model(model, name="model")

        print(f"Accuracy: {acc:.3f}  |  Log loss: {loss:.3f}")
        print("Run logged to MLflow — inspect with: mlflow ui")


if __name__ == "__main__":
    main()
