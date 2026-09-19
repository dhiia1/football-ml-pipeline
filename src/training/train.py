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
    uv run python src/training/train.py

TODO once this works:
- Try tuning XGBoost's hyperparameters (n_estimators, max_depth,
  learning_rate) and compare runs in the MLflow UI (`mlflow ui`).
- Add more features to the feature table and re-run — that's the workflow
  you'll repeat dozens of times, and MLflow is what makes it not chaos.
- Try a rolling-origin backtest (multiple chronological splits) instead of
  one split, for a more honest accuracy estimate.
"""

import yaml
import mlflow
import mlflow.xgboost
import pandas as pd
from pathlib import Path
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, log_loss
from sklearn.preprocessing import LabelEncoder

ROOT = Path(__file__).resolve().parents[2]
CONFIG = yaml.safe_load(open(ROOT / "config" / "config.yaml"))

FEATURE_COLS = ["home_form", "away_form", "h2h_home_advantage", "home_elo", "away_elo"]


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
        train_acc = accuracy_score(y_train, model.predict(X_train))
        print(f"Train accuracy: {train_acc:.3f}  |  Test accuracy: {acc:.3f}")
        loss = log_loss(y_test, probs)

        mlflow.log_param("model_type", "XGBClassifier")
        mlflow.log_param("n_estimators", 100)
        mlflow.log_param("max_depth", 3)
        mlflow.log_param("learning_rate", 0.1)
        mlflow.log_param("features", FEATURE_COLS)
        mlflow.log_param("train_size", len(train_df))
        mlflow.log_param("test_size", len(test_df))
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("log_loss", loss)
        mlflow.xgboost.log_model(model, name="model")

        print(f"Accuracy: {acc:.3f}  |  Log loss: {loss:.3f}")
        print("Run logged to MLflow — inspect with: mlflow ui")


if __name__ == "__main__":
    main()
