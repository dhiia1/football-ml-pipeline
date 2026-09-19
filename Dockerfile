# Serves the trained model + API only — NOT the training/ingestion pipeline.
# Training, MLflow tracking, and promotion all happen on the dev machine;
# this image is the "production" bundle: model artifacts + the data
# snapshots serving needs, baked in at build time.
#
# NOTE: this is the "bake it in" pattern, not a shared MLflow registry.
# Promoting a new model means rebuilding this image — there's no live
# hot-swap. See the app.py TODO about a /reload endpoint if you ever want
# to move to a volume-mounted, hot-swappable version instead.

FROM python:3.11-slim

WORKDIR /app

# Install uv itself (fast, reproducible installs from pyproject.toml/uv.lock)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Install dependencies first (better layer caching — this only re-runs if
# pyproject.toml/uv.lock change, not on every code edit)
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

# Application code
COPY src/ src/
COPY config/ config/

# MLflow tracking DB (registry metadata: which version is "production")
# and the actual artifact folder (model weights) — both required, since
# they're separate stores even though they're both "MLflow."
COPY model_export/ model_export/
ENV MODEL_PATH=/app/model_export

# Serving-time data artifacts: current team state, upcoming fixtures, and
# raw match history (needed for on-demand head-to-head lookups).
COPY data/processed/team_state.json data/processed/team_state.json
COPY data/processed/upcoming_fixtures.json data/processed/upcoming_fixtures.json
COPY data/raw/ data/raw/

EXPOSE 8000

CMD ["uv", "run", "--no-sync", "uvicorn", "src.serving.app:app", "--host", "0.0.0.0", "--port", "8000"]