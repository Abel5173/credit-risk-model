import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
import mlflow
import mlflow.sklearn
from src.data_processing import build_feature_engineering_pipeline

# List of columns to drop after feature engineering (IDs, timestamps, etc.)
NON_FEATURE_COLS = [
    "TransactionId",
    "BatchId",
    "AccountId",
    "SubscriptionId",
    "CustomerId",
    "TransactionStartTime",
    "is_high_risk",
    "FraudResult",
    "Unnamed: 16",
    "Unnamed: 17",
]


def load_data(path):
    return pd.read_csv(path)


def get_X_y(df):
    X = df.drop(
        [col for col in NON_FEATURE_COLS if col in df.columns], axis=1, errors="ignore"
    )
    y = df["is_high_risk"]
    # Keep only numeric columns for model training
    X = X.select_dtypes(include=["float64", "int64"])
    return X, y


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_proba = (
        model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
    )
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba) if y_proba is not None else None,
    }
    return metrics


def train_and_log_model(
    model, param_grid, X_train, y_train, X_test, y_test, model_name
):
    grid = GridSearchCV(model, param_grid, scoring="roc_auc", cv=5)
    grid.fit(X_train, y_train)
    best_model = grid.best_estimator_
    metrics = evaluate_model(best_model, X_test, y_test)
    with mlflow.start_run(run_name=model_name):
        mlflow.sklearn.log_model(best_model, "model")
        mlflow.log_params(grid.best_params_)
        mlflow.log_metrics(metrics)
        mlflow.set_tag("model_type", model_name)
        # Register model
        mlflow.register_model(
            f"runs:/{mlflow.active_run().info.run_id}/model",
            f"CreditRiskModel_{model_name}",
        )
    return best_model, metrics


if __name__ == "__main__":
    data_path = os.path.join("data", "processed", "processed_data.csv")
    df = load_data(data_path)
    # Apply feature engineering pipeline
    pipeline = build_feature_engineering_pipeline(
        df, scaler="standard", use_woe_iv=False
    )
    df_fe = pipeline.fit_transform(df, df["is_high_risk"])
    # Convert to DataFrame if needed
    if not isinstance(df_fe, pd.DataFrame):
        df_fe = pd.DataFrame(df_fe)
    # Add target back if missing
    if "is_high_risk" not in df_fe.columns and "is_high_risk" in df.columns:
        df_fe["is_high_risk"] = df["is_high_risk"].values
    X, y = get_X_y(df_fe)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, random_state=42
    )
    # Logistic Regression
    lr = LogisticRegression(max_iter=1000)
    lr_param_grid = {"C": [0.01, 0.1, 1, 10]}
    best_lr, lr_metrics = train_and_log_model(
        lr, lr_param_grid, X_train, y_train, X_test, y_test, "LogisticRegression"
    )
    # Random Forest
    rf = RandomForestClassifier(random_state=42)
    rf_param_grid = {"n_estimators": [50, 100], "max_depth": [3, 5, 10]}
    best_rf, rf_metrics = train_and_log_model(
        rf, rf_param_grid, X_train, y_train, X_test, y_test, "RandomForest"
    )
    print("Logistic Regression metrics:", lr_metrics)
    print("Random Forest metrics:", rf_metrics)
