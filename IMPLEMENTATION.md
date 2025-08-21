# Implementation Guide

This guide explains the architecture, key components, and how they work together from data to serving.

## Architecture Overview

- Data → Proxy Label → Features → Model Training → Registry → Serving (API & Dashboard) → Explainability (SHAP) → Monitoring via MLflow
- All preprocessing is inside sklearn Pipelines to prevent training/serving skew.

## Components

### Data & Proxy Labeling (`src/rfm_proxy.py`)

- Loads raw data from `data/raw/` and produces `data/processed/processed_data.csv`.
- Computes RFM metrics per customer:
  - Recency: days since last transaction
  - Frequency: number of transactions
  - Monetary: total spend (or sum of `Amount`/`Value`)
- Clusters RFM with KMeans and marks the high-risk cluster as `is_high_risk`.
- Streaming/batched computation using `utils.data_generator` to handle large inputs.

### Feature Engineering (`src/data_processing.py`)

- AggregateFeatures: per-customer aggregations (e.g., rolling counts, sums if applicable).
- DateTimeFeatures: extracts components (hour, day of week, etc.).
- WOETransformer: encodes selected categoricals using Weight of Evidence (monotonic binning).
  - Safe no-op if there are no suitable columns.
- DropColumns: removes target, IDs, and timestamps to prevent leakage.
- ColumnTransformer: numeric impute + scale; categorical impute + one-hot; passthrough remainder.
- Returns DataFrame-friendly outputs for traceability.

### Modeling & Training (`src/train.py`)

- End-to-end sklearn Pipelines:
  - Pipeline(steps=[('features', build_feature_engineering_pipeline(...)), ('clf', Estimator)])
- GridSearchCV with ROC-AUC scoring.
- Metrics logged to MLflow (accuracy, precision, recall, f1, roc_auc) and business metrics:
  - `calculate_credit_score(prob)` → 300–850 mapping.
  - `predict_optimal_loan(prob)` → amount/duration suggestions.
- Attempts model registration in MLflow Model Registry; guarded with try/except.

### Serving (`src/api/`)

- FastAPI service loads the Production model from the MLflow Registry.
- `/health` endpoint for liveness checks.
- `/predict` accepts transaction records and returns risk probability, credit score, and loan terms.
- Pydantic schemas relaxed to accept partially filled inputs for demos.

### Explainability (`src/explainability.py`)

- Generates SHAP values.
- If inside an active MLflow run, logs SHAP plots as artifacts.
- Works with Pipelines by transforming through the feature step.

### Dashboard (`src/dashboard.py`)

- Streamlit app for non-technical stakeholders.
- Upload CSV → computes aggregate risk/score/loan; shows SHAP summary.
- Loads the Production model from the Registry.

### Utilities (`src/utils.py`, `src/model_utils.py`)

- Timing and profiling utilities to measure performance.
- Business mapping utilities for credit score and loan suggestions (cached functions).

## MLflow Setup

- Docker Compose launches MLflow with SQLite backend and `--serve-artifacts`.
- `MLFLOW_TRACKING_URI` should be set to `http://localhost:5000` for local runs.
- Promote a model to Production in the MLflow UI for the API/Dashboard to pick it up.

## Testing & CI

- Unit tests for transformers and business metrics; integration test for a minimal pipeline.
- GitHub Actions (`.github/workflows/ci-cd.yml`) runs tests/lint and builds Docker images on push/PR.

## Runbook (Local)

1. Prepare data: `python src/rfm_proxy.py`
2. Train models: `python src/train.py`
3. Start services: `docker compose up -d`
4. Promote model in MLflow UI (Models tab → Production)
5. Test API: POST to `/predict` (see README for sample payload)
6. Use dashboard: open `http://localhost:8501` and upload a CSV

## Notes

- Ensure `requirements.txt` is installed in your virtual environment.
- For constrained networks, increase pip timeout and retries as needed.
