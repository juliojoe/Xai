"""
explainability.py
------------------
SHAP-based global and local explanations for the best-performing
supervised model (XGBoost), plus a reconstruction-error-based
"explanation" for the Autoencoder.
"""

import numpy as np
import pandas as pd
import shap


def get_tree_explainer(model) -> shap.TreeExplainer:
    """Build a SHAP explainer for a tree-based model (Random Forest / XGBoost)."""
    return shap.TreeExplainer(model)


def global_shap_values(explainer: shap.TreeExplainer, X_sample: pd.DataFrame):
    """
    SHAP values for a sample of the test set -- feed the result into
    shap.summary_plot(shap_values, X_sample) for the global explanation
    (which features generally influence fraud predictions).
    """
    return explainer.shap_values(X_sample)


def local_explanation(explainer: shap.TreeExplainer, instance: pd.DataFrame):
    """
    SHAP values for a single transaction -- feed into
    shap.force_plot(explainer.expected_value, shap_values[0], instance)
    for the local explanation (why THIS transaction was flagged).
    """
    return explainer.shap_values(instance)


def top_shap_features(explainer: shap.TreeExplainer, instance: pd.DataFrame, top_n: int = 5) -> pd.Series:
    """
    Convenience helper for the dashboard: returns the top_n features by
    absolute SHAP contribution for one transaction, signed so positive
    values push toward fraud and negative values push toward legitimate.
    """
    values = explainer.shap_values(instance)[0]
    series = pd.Series(values, index=instance.columns)
    return series.reindex(series.abs().sort_values(ascending=False).index).head(top_n)


def autoencoder_feature_errors(instance_row: np.ndarray, reconstructed_row: np.ndarray,
                                columns) -> pd.Series:
    """
    The Autoencoder's natural "explanation": per-feature squared
    reconstruction error, sorted descending, for one transaction.
    """
    error = (instance_row - reconstructed_row) ** 2
    return pd.Series(error, index=columns).sort_values(ascending=False)
