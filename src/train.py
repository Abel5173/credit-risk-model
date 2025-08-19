from src.model_utils import calculate_credit_score, predict_optimal_loan
from src.data_processing import build_feature_engineering_pipeline
from sklearn.pipeline import Pipeline as SkPipeline
import mlflow.sklearn
import mlflow
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")))


NON_FEATURE_COLS = [
    "TransactionId",
    "BatchId",
    "AccountId",
    "SubscriptionId",
    "CustomerId",
    "TransactionStartTime",
    "FraudResult",
    "Unnamed: 16",
    "Unnamed: 17",
]


def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series):
    y_pred = model.predict(X_test)
    y_proba = None
    # For sklearn pipelines with classifiers, predict_proba should exist if classifier supports it
    if hasattr(model, "predict_proba"):
        try:
            y_proba = model.predict_proba(X_test)[:, 1]
        except Exception:
            y_proba = None
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba) if y_proba is not None else None,
    }
    return metrics, y_proba


def train_and_log_model(model, param_grid, X_train, y_train, X_test, y_test, model_name):
    grid = GridSearchCV(model, param_grid, scoring="roc_auc", cv=5)
    grid.fit(X_train, y_train)
    best_model = grid.best_estimator_
    metrics, y_proba = evaluate_model(best_model, X_test, y_test)
    with mlflow.start_run(run_name=model_name):
        mlflow.sklearn.log_model(best_model, "model")
        mlflow.log_params(grid.best_params_)
        mlflow.log_metrics(metrics)
        mlflow.set_tag("model_type", model_name)
        if y_proba is not None:
            scores = calculate_credit_score(y_proba)
            mlflow.log_metric("avg_credit_score", float(np.mean(scores)))
            loans = list(
                map(lambda p: predict_optimal_loan(float(p)), y_proba))
            avg_amount = float(np.mean([amt for amt, _ in loans]))
            avg_duration = float(np.mean([dur for _, dur in loans]))
            mlflow.log_metric("avg_optimal_loan_amount", avg_amount)
            mlflow.log_metric("avg_optimal_loan_duration", avg_duration)
        # Register model (best-effort)
        try:
            mlflow.register_model(
                f"runs:/{mlflow.active_run().info.run_id}/model",
                f"CreditRiskModel_{model_name}",
            )
        except Exception as e:
            print(f"Model registry not available, skipping registration: {e}")
    return best_model, metrics


if __name__ == "__main__":
    data_path = os.path.join("data", "processed", "processed_data.csv")
    df = load_data(data_path)

    y = df["is_high_risk"].astype(int)
    X = df.drop(columns=["is_high_risk"], errors="ignore")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, random_state=42)

    # We build a feature engineering pipeline using training data schema, WOE enabled
    fe_pipeline = build_feature_engineering_pipeline(
        pd.concat([X_train, y_train.rename("is_high_risk")], axis=1), scaler="standard", use_woe_iv=True
    )

    # Logistic Regression: end-to-end pipeline
    lr_pipe = SkPipeline([
        ("features", fe_pipeline),
        ("clf", LogisticRegression(max_iter=1000)),
    ])
    lr_param_grid = {"clf__C": [0.01, 0.1, 1, 10]}
    best_lr, lr_metrics = train_and_log_model(
        lr_pipe, lr_param_grid, X_train, y_train, X_test, y_test, "LogisticRegression")

    # Random Forest: end-to-end pipeline
    rf_pipe = SkPipeline([
        ("features", fe_pipeline),
        ("clf", RandomForestClassifier(random_state=42)),
    ])
    rf_param_grid = {"clf__n_estimators": [
        50, 100], "clf__max_depth": [3, 5, 10]}
    best_rf, rf_metrics = train_and_log_model(
        rf_pipe, rf_param_grid, X_train, y_train, X_test, y_test, "RandomForest")

    print("Logistic Regression metrics:", lr_metrics)
    print("Random Forest metrics:", rf_metrics)
