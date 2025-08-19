import os
from typing import Optional
import numpy as np
import pandas as pd
import shap
import mlflow
import matplotlib.pyplot as plt
from src.utils import timer_decorator, profile_context


def _get_clf_and_features(model):
    """If model is a sklearn Pipeline, return (clf, features_step). Otherwise (model, None)."""
    clf = model
    features_step = None
    if hasattr(model, 'named_steps'):
        features_step = model.named_steps.get('features')
        clf = model.named_steps.get('clf', model)
    return clf, features_step


def _pick_explainer(clf, X_background):
    # Try model-specific explainers first
    try:
        from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
        from xgboost import XGBClassifier  # optional
    except Exception:
        RandomForestClassifier = GradientBoostingClassifier = XGBClassifier = tuple()

    if hasattr(clf, 'predict_proba'):
        # Tree-based
        if clf.__class__.__name__.lower().find('forest') >= 0 or clf.__class__.__name__.lower().find('boost') >= 0:
            try:
                return shap.TreeExplainer(clf)
            except Exception:
                pass
        # Linear-based
        if clf.__class__.__name__.lower().find('logistic') >= 0:
            try:
                return shap.LinearExplainer(clf, X_background)
            except Exception:
                pass
    # Fallback to KernelExplainer
    return shap.KernelExplainer(lambda X: clf.predict_proba(X)[:, 1], X_background)


@timer_decorator
def generate_shap_explanations(model, X_background: pd.DataFrame, X_instance: pd.DataFrame,
                               out_dir: str = 'artifacts', filename: str = 'shap_summary.png') -> str:
    """Generate SHAP summary plot for a model and save to disk, logging to MLflow if active.

    For Pipelines, transforms features before computing SHAP values and explains the classifier.
    Returns the path to the saved image.
    """
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, filename)

    clf, features_step = _get_clf_and_features(model)

    # Transform data if we have a features step
    if features_step is not None:
        X_bg = features_step.transform(X_background)
        X_ins = features_step.transform(X_instance)
        # Ensure arrays for SHAP if DataFrames are not handled by explainer
        if hasattr(X_bg, 'to_numpy'):
            X_bg_arr = X_bg.to_numpy()
            X_ins_arr = X_ins.to_numpy()
        else:
            X_bg_arr = np.asarray(X_bg)
            X_ins_arr = np.asarray(X_ins)
    else:
        X_bg_arr = X_background.to_numpy() if hasattr(
            X_background, 'to_numpy') else np.asarray(X_background)
        X_ins_arr = X_instance.to_numpy() if hasattr(
            X_instance, 'to_numpy') else np.asarray(X_instance)

    with profile_context('SHAP'):
        explainer = _pick_explainer(clf, X_bg_arr)
        shap_values = explainer.shap_values(X_ins_arr)
        # Some explainers return list for multiclass; handle binary
        if isinstance(shap_values, list):
            # Choose positive class index 1 if available
            shap_vals = shap_values[min(1, len(shap_values)-1)]
        else:
            shap_vals = shap_values
        plt.figure(figsize=(8, 5))
        try:
            shap.summary_plot(shap_vals, X_ins_arr, show=False)
        except Exception:
            # Fallback: force_plot for single instance
            plt.clf()
            shap.force_plot(explainer.expected_value if not isinstance(explainer.expected_value, (list, np.ndarray)) else explainer.expected_value[0],
                            shap_vals[0] if isinstance(shap_vals, np.ndarray) else shap_vals, X_ins_arr[0], matplotlib=True, show=False)
        plt.tight_layout()
        plt.savefig(out_path, dpi=150)
        plt.close()

    # Log as MLflow artifact if in an active run
    try:
        active = mlflow.active_run()
        if active is not None:
            mlflow.log_artifact(out_path)
    except Exception:
        pass

    return out_path
