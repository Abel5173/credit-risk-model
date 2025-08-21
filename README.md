# B5W5: Credit Risk Probability Model (BNPL) using Alternative eCommerce Data

An end-to-end, production-ready credit risk project for a Buy-Now-Pay-Later (BNPL) use case built on the Xente transactions dataset. It emphasizes reliability, interpretability, and impact for financial stakeholders.

Key highlights:

- Proxy risk labeling from behavioral RFM metrics (when defaults are not available)
- Interpretable feature engineering with Weight of Evidence (WOE)
- Logged and versioned models with MLflow Model Registry
- FastAPI service for real-time inference
- Streamlit dashboard for stakeholders with SHAP explanations
- Dockerized services and CI-ready test suite

See also:

- CHANGELOG: ./CHANGELOG.md
- Implementation Guide: ./IMPLEMENTATION.md

## 1) Problem

Decide who to approve, how much to lend, and for how long, using transaction behavior. Reduce default risk while preserving approvals and revenue. Provide interpretable and auditable models.

## 2) Data & Labeling

- Source: Xente eCommerce transactions (Kaggle)
- No true default labels → create `is_high_risk` via RFM:
  - Recency (days since last transaction)
  - Frequency (count)
  - Monetary (total spend)
- KMeans clustering to pick the high-risk segment (high recency, low frequency, low monetary)

Run proxy labeling:

- python src/rfm_proxy.py
- Output: data/processed/processed_data.csv (adds `is_high_risk`)

## 3) Features & Modeling

- Custom transformers: per-customer aggregates and datetime parts
- WOE encoding for selected categorical features (monotonic binning, missing-safe)
- End-to-end sklearn Pipelines (preprocessing + model) for Logistic Regression and Random Forest
- GridSearchCV (ROC-AUC), metrics logged to MLflow
- Business metrics: map probabilities to credit scores (300–850) and suggest loan terms (amount, duration)

## 4) Serving & Explainability

- FastAPI `/predict` endpoint loads the Production model from MLflow Registry
- Streamlit dashboard to upload CSVs, preview risk/score/loan, and view SHAP explanations

## 5) Project Structure

```
.
├─ src/
│  ├─ api/               # FastAPI service
│  ├─ data_processing.py  # Transformers + pipeline (WOE activated)
│  ├─ rfm_proxy.py        # RFM proxy label generator
│  ├─ train.py            # Train & register models, log metrics
│  ├─ model_utils.py      # Credit score + loan suggestion helpers
│  ├─ explainability.py   # SHAP utilities (timed & profiled)
│  ├─ dashboard.py        # Streamlit stakeholder UI
│  └─ utils.py            # timer, profiler, batch generator
├─ tests/
├─ notebooks/1.0-eda.ipynb
├─ Dockerfile
├─ docker-compose.yml     # API + MLflow + Dashboard
├─ requirements.txt
└─ mlruns/                # local MLflow tracking (artifacts)
```

## 6) Quickstart

Prereqs: Python 3.12+ and Docker.

1. Install dependencies (for local dev):

   - pip install -r requirements.txt

2. Prepare data:

   - Place raw files under data/raw/ (e.g., raw_data.csv or raw_data.xlsx)
   - python src/rfm_proxy.py

3. Train & register models:

   - python src/train.py
   - Open MLflow at http://localhost:5000 (after compose up) to view runs and models

4. Promote a model to Production:

   - In MLflow UI: Models → CreditRiskModel_LogisticRegression (or \_RandomForest) → Transition latest version to Production

5. Bring up services with Docker:
   - docker compose up --build
   - API: http://localhost:8000 (health: /health, predict: /predict)
   - MLflow: http://localhost:5000
   - Dashboard: http://localhost:8501

### Sample predict request (JSON)

```
{
  "TransactionId": "T123",
  "BatchId": "B001",
  "AccountId": "A001",
  "SubscriptionId": "",
  "CurrencyCode": "UGX",
  "CountryCode": 256,
  "ProviderId": "MTN",
  "ProductId": "P100",
  "ProductCategory": "Electronics",
  "ChannelId": "App",
  "Amount": 120000,
  "Value": 120000,
  "TransactionStartTime": "2024-06-01 12:34:56",
  "PricingStrategy": 0,
  "FraudResult": 0
}
```

## 7) Testing & CI

- Run tests: pytest -v
- Lint: flake8
- GitHub Actions workflow included to run tests and lint on push/PR

## 8) Explainability

- SHAP summary plots are generated and can be logged to MLflow
- Dashboard displays SHAP to help stakeholders interpret model behavior

## 9) Notes for Reviewers (Employers/Clients)

- Reliable engineering: versioned models, tracked runs, pipeline encapsulation, and tests
- Interpretability: WOE features and SHAP visuals
- Ready to deploy: Dockerized API and dashboard; can be hosted on a single VM

## 10) License

MIT (update as needed)
