"""
Serving stage.

Wraps the trained model in a FastAPI endpoint. This is the line between
"I trained a model" and "I shipped a model" — a URL someone (or something)
else can actually call.

The model is loaded via the "production" alias from the MLflow Model
Registry — not a hardcoded run ID or file path. This means promoting a new
model (via evaluate.py) makes it live here automatically: restart the app
(or add a refresh endpoint later) and it picks up whatever's currently
aliased "production", with zero code changes.

Run:
    uv run uvicorn src.serving.app:app --reload

Then:
    curl -X POST http://localhost:8000/predict \
      -H "Content-Type: application/json" \
      -d '{"home_form": 2.0, "away_form": 1.2}'

TODO once this works:
- Add a /reload endpoint that re-fetches the aliased model without
  restarting the whole app — useful once retraining runs on a schedule.
- Add request logging (predictions + inputs) so the monitoring stage has
  something to compare against actual outcomes later.
"""
import yaml
import mlflow
import mlflow.sklearn
from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parents[2]
CONFIG = yaml.safe_load(open(ROOT / "config" / "config.yaml"))

mlflow.set_tracking_uri("sqlite:///" + str(ROOT / "mlflow.db"))

app = FastAPI(title="Football Outcome Predictor")

REGISTRY_NAME = CONFIG["model"]["registry_name"]
ALIAS = "production"
MODEL_URI = f"models:/{REGISTRY_NAME}@{ALIAS}"
_model = None


class MatchFeatures(BaseModel):
    home_form: float
    away_form: float


def get_model():
    global _model
    if _model is None:
        try:
            _model = mlflow.sklearn.load_model(MODEL_URI)
        except Exception as e:
            raise RuntimeError(
                f"Could not load model at '{MODEL_URI}'. Has a model been "
                f"promoted yet? Run src/evaluation/evaluate.py first. "
                f"Original error: {e}"
            )
    return _model


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(features: MatchFeatures):
    try:
        model = get_model()
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))

    X = [[features.home_form, features.away_form]]
    pred = model.predict(X)[0]
    proba = model.predict_proba(X)[0].tolist()
    return {"prediction": int(pred), "probabilities": proba}