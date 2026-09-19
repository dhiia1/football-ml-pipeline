# Football Match Outcome Pipeline

An end-to-end ML pipeline that predicts football match outcomes (Home Win /
Draw / Away Win) using live data from football-data.org, built to practice
real MLOps: experiment tracking, versioning, orchestration, serving, and
monitoring — not just training a model in a notebook.

## Architecture

```
football-data.org API
        |
        v
[INGEST]  -> raw JSON snapshots (dated, versioned)
        |
        v
[FEATURE BUILD]  -> rolling form, H2H, home/away splits -> processed parquet
        |
        v
[TRAIN]  -> chronological split -> MLflow logs run
        |
        v
[EVALUATE]  -> compare vs baseline -> only register if it beats it
        |
        v
[REGISTRY]  -> MLflow Model Registry holds "Production" model
        |
        v
[SERVE]  -> FastAPI loads latest Production model -> /predict endpoint
        |
        v
[MONITOR]  -> log prediction vs actual result -> rolling accuracy + drift check
        |
        +--> feeds back into next [TRAIN] cycle
```

## Why it's built this way

- **Chronological splits only.** Football data is time-ordered. Random
  train/test splits leak future team form into the past and give you fake
  accuracy. Always train on matches before date X, evaluate on matches after.
- **Baseline-gated registry.** A new model is only promoted to "Production"
  in MLflow if it beats a dumb baseline (e.g. "always predict home win",
  which is typically ~45-46% accurate in most leagues). This is what makes
  the pipeline auditable instead of "trust me, it's a good model."
- **Ingestion is the trigger, not a one-off script.** Because the API updates
  as real matches are played, ingestion running on a schedule is what forces
  you to build proper orchestration and monitoring, not fake it.

## Getting started

1. Get a free API key from https://www.football-data.org (Free tier covers
   top competitions with a reasonable rate limit).
2. Copy `.env.example` to `.env` and add your key.
3. `pip install -r requirements.txt`
4. Run stages manually first, in order:
   ```
   python src/ingestion/fetch_football_data.py
   python src/features/build_features.py
   python src/training/train.py
   python src/evaluation/evaluate.py
   ```
5. Once that works end to end, wire them into `pipelines/flow.py` (Prefect)
   and schedule it.
6. Serve the latest Production model:
   ```
   uvicorn src.serving.app:app --reload
   ```

## Stages (fill these in as you learn each concept)

| Stage | File | MLOps concept it teaches |
|---|---|---|
| Ingest | `src/ingestion/fetch_football_data.py` | reproducible, versioned data snapshots |
| Features | `src/features/build_features.py` | leakage-safe feature engineering |
| Train | `src/training/train.py` | experiment tracking (MLflow) |
| Evaluate | `src/evaluation/evaluate.py` | baseline-gated promotion |
| Serve | `src/serving/app.py` | packaging a model as a service |
| Orchestrate | `pipelines/flow.py` | scheduled, chained pipeline runs |
| Monitor | `src/monitoring/drift_check.py` | drift detection, feedback loop |
| CI | `.github/workflows/ci.yml` | automated gating before merge |

Each file below has a working skeleton with TODOs — the scaffolding runs,
but the actual modeling choices are intentionally left for you to build out
so it actually sticks.
