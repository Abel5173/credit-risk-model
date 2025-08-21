# Changelog

All notable changes to this project will be documented in this file.

## 2025-08-21

### Added
- Robust EDA notebook updates:
  - Auto-detect `target` (`is_high_risk`/`FraudResult`) and `time` columns with graceful fallbacks.
  - Safer plots (skip when cols missing) and new RFM sanity checks + leakage guard section.
- Explainability and stakeholder tools:
  - `src/explainability.py` for SHAP summaries; Pipeline-aware and MLflow artifact logging.
  - `src/dashboard.py` Streamlit app to upload CSVs and view risk/score/loan + SHAP summary.
- Modeling & utilities:
  - End-to-end sklearn Pipelines in `src/train.py` (features + estimator) with GridSearchCV (ROC-AUC).
  - Business metrics logging (credit score, optimal loan suggestions).
  - `src/utils.py` (timer decorator, cProfile context, batch generator).
  - `src/model_utils.py` (probability→credit score; loan suggestions; cached and timed).
- API & serving:
  - FastAPI loads sklearn Pipeline from MLflow Model Registry; robust probability handling.
  - Relaxed request schema in `src/api/pydantic_models.py` to ease demo inputs.
- Data processing & labels:
  - Streaming RFM proxy pipeline in `src/rfm_proxy.py` with safer I/O.
  - WOE encoding activated with a safe no-op when no columns; added `DropColumns` to remove IDs/timestamps.
- Infrastructure & CI:
  - `docker-compose.yml` with MLflow server (`--serve-artifacts`) and dashboard service.
  - GitHub Actions workflow `.github/workflows/ci-cd.yml` (tests + lint + docker build).
  - Expanded tests: `tests/test_model_utils.py`, `tests/test_integration.py`, and updates to `tests/test_data_processing.py`.

### Changed
- `src/data_processing.py`: stricter feature pipeline (aggregate → datetime → optional WOE → drop ID/time → ColumnTransformer with DataFrame-friendly output).
- `src/train.py`: safer model registration (try/except), consistent logging to MLflow.
- `src/rfm_proxy.py`: refactored to streaming/batched RFM calculation.
- `README.md`: portfolio-grade overhaul with clearer quickstart and structure.
- `requirements.txt`: added/pinned packages (scikit-learn, xverse, mlflow, shap, streamlit, etc.).

### Fixed
- Reduced leakage by dropping identifiers and timestamps before modeling.
- WOE transformer now handles empty/absent columns gracefully (no-op behavior).

### Notes
- Promote the trained model to "Production" in MLflow UI so API and dashboard load it by default.
- `docker-compose.yml` sets up MLflow with SQLite backend and serves artifacts for cross-service access.
