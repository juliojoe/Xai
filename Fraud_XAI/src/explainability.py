"""
Explainable AI (XAI) module for fraud detection.

This module provides TWO complementary explanation techniques:

1. SHAP (SHapley Additive exPlanations)
   - Game-theory based feature attributions
   - Consistent global + local explanations
   - Fast for tree models (TreeExplainer)

2. LIME (Local Interpretable Model-agnostic Explanations)
   - Approximates the model locally with a simple linear model
   - Works on ANY black-box model (model-agnostic)
   - Very intuitive "why this prediction?" explanations

Both methods answer the same core question for a single transaction:
"Which features pushed the model toward FRAUD vs LEGITIMATE?"

Positive contribution  → increases fraud risk
Negative contribution → decreases fraud risk
"""

from __future__ import annotations

from typing import Any, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

# LIME is imported lazily so the rest of the project still works
# even if lime is not yet installed.
try:
    from lime.lime_tabular import LimeTabularExplainer
    LIME_AVAILABLE = True
except ImportError:
    LIME_AVAILABLE = False
    LimeTabularExplainer = None  # type: ignore


# ---------------------------------------------------------------------------
# SHAP helpers
# ---------------------------------------------------------------------------

def make_shap_explainer(
    model: Any,
    X_background: np.ndarray,
    feature_names: List[str],
    model_type: str = "tree",
) -> shap.Explainer:
    """
    Create a SHAP explainer.

    Parameters
    ----------
    model : fitted classifier (must support predict / predict_proba)
    X_background : small sample of training data used as background
                   (only needed for LinearExplainer / KernelExplainer)
    feature_names : list of column names (for nicer plots)
    model_type : "tree"   → TreeExplainer (RF, XGBoost, LightGBM …)
                 "linear" → LinearExplainer (LogisticRegression)
                 "kernel" → KernelExplainer (slow but model-agnostic)

    Returns
    -------
    A ready-to-use shap.Explainer instance.
    """
    if model_type == "tree":
        # Fastest path for tree ensembles
        explainer = shap.TreeExplainer(model)
    elif model_type == "linear":
        explainer = shap.LinearExplainer(model, X_background)
    else:
        # KernelExplainer is slow – subsample background
        bg = shap.sample(X_background, min(100, len(X_background)))
        explainer = shap.KernelExplainer(model.predict_proba, bg)
    return explainer


def shap_global_summary(
    explainer: shap.Explainer,
    X: np.ndarray,
    feature_names: List[str],
    max_display: int = 15,
    save_path: Optional[str] = None,
) -> None:
    """
    Global SHAP summary (beeswarm) plot.

    Shows which features are most important across many transactions
    and how high/low feature values push the prediction.
    """
    shap_values = explainer.shap_values(X)

    # Binary classifiers often return [class0, class1]
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    plt.figure()
    shap.summary_plot(
        shap_values,
        X,
        feature_names=feature_names,
        max_display=max_display,
        show=False,
    )
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close()


def shap_local_explanation(
    explainer: shap.Explainer,
    x_instance: np.ndarray,
    feature_names: List[str],
) -> pd.DataFrame:
    """
    Local SHAP explanation for ONE transaction.

    Returns a DataFrame sorted by absolute contribution:
        feature | value | shap | abs_shap
    """
    shap_vals = explainer.shap_values(x_instance.reshape(1, -1))
    if isinstance(shap_vals, list):
        shap_vals = shap_vals[1]
    shap_vals = np.asarray(shap_vals).flatten()

    df = pd.DataFrame(
        {
            "feature": feature_names,
            "value": x_instance.flatten(),
            "shap": shap_vals,
        }
    )
    df["abs_shap"] = df["shap"].abs()
    df = df.sort_values("abs_shap", ascending=False).reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# LIME helpers
# ---------------------------------------------------------------------------

def make_lime_explainer(
    X_train: np.ndarray,
    feature_names: List[str],
    class_names: Optional[List[str]] = None,
    mode: str = "classification",
) -> "LimeTabularExplainer":
    """
    Create a LIME tabular explainer.

    LIME learns a simple linear model in the neighbourhood of the
    instance you want to explain.  It is completely model-agnostic.

    Parameters
    ----------
    X_train : training feature matrix (used to compute feature statistics)
    feature_names : column names
    class_names : e.g. ["Legitimate", "Fraud"]
    mode : "classification" (default) or "regression"

    Returns
    -------
    LimeTabularExplainer ready for .explain_instance(...)
    """
    if not LIME_AVAILABLE:
        raise ImportError(
            "lime is not installed.  Run:  pip install lime\n"
            "Then restart your kernel / process."
        )

    if class_names is None:
        class_names = ["Legitimate", "Fraud"]

    explainer = LimeTabularExplainer(
        training_data=np.asarray(X_train, dtype=np.float64),
        feature_names=feature_names,
        class_names=class_names,
        mode=mode,
        discretize_continuous=True,   # makes explanations more readable
        random_state=42,
    )
    return explainer


def lime_local_explanation(
    lime_explainer: "LimeTabularExplainer",
    model: Any,
    x_instance: np.ndarray,
    feature_names: List[str],
    num_features: int = 10,
    predict_fn: Optional[Any] = None,
) -> pd.DataFrame:
    """
    Local LIME explanation for ONE transaction.

    Parameters
    ----------
    lime_explainer : object returned by make_lime_explainer
    model : any fitted model that has predict_proba
    x_instance : 1-D array of feature values
    feature_names : list of names (same order as x_instance)
    num_features : how many top features to return
    predict_fn : optional custom prediction function;
                 defaults to model.predict_proba

    Returns
    -------
    DataFrame with columns:
        feature | value | lime_weight | abs_weight
    Positive lime_weight → pushes toward Fraud
    Negative lime_weight → pushes toward Legitimate
    """
    if not LIME_AVAILABLE:
        raise ImportError("lime is not installed.  Run: pip install lime")

    if predict_fn is None:
        predict_fn = model.predict_proba

    # LIME expects a 1-D array
    instance = np.asarray(x_instance, dtype=np.float64).flatten()

    explanation = lime_explainer.explain_instance(
        data_row=instance,
        predict_fn=predict_fn,
        num_features=num_features,
        top_labels=1,           # we care about the positive (fraud) class
    )

    # explanation.as_list() returns (feature_description, weight)
    # For the fraud class (label 1)
    label = 1 if 1 in explanation.available_labels() else explanation.available_labels()[0]
    pairs = explanation.as_list(label=label)

    # Build a clean DataFrame.  LIME sometimes returns "feature=value"
    # strings; we also keep the original numeric value.
    rows = []
    for feat_desc, weight in pairs:
        # Try to recover the original feature name
        feat_name = feat_desc.split("=")[0].split("<")[0].split(">")[0].strip()
        # Fallback: keep the whole description if parsing fails
        if feat_name not in feature_names:
            feat_name = feat_desc

        # Get the numeric value from the instance when possible
        try:
            idx = feature_names.index(feat_name)
            val = float(instance[idx])
        except (ValueError, IndexError):
            val = np.nan

        rows.append(
            {
                "feature": feat_name,
                "value": val,
                "lime_weight": weight,
                "abs_weight": abs(weight),
            }
        )

    df = pd.DataFrame(rows)
    df = df.sort_values("abs_weight", ascending=False).reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Unified human-readable text
# ---------------------------------------------------------------------------

def format_local_text(
    local_df: pd.DataFrame,
    fraud_prob: float,
    method: str = "SHAP",
    top_k: int = 5,
) -> str:
    """
    Turn a local explanation DataFrame into a short plain-English summary.

    Works for both SHAP (column 'shap') and LIME (column 'lime_weight').
    """
    # Detect which column holds the contribution
    if "shap" in local_df.columns:
        contrib_col = "shap"
    elif "lime_weight" in local_df.columns:
        contrib_col = "lime_weight"
    else:
        raise ValueError("DataFrame must contain 'shap' or 'lime_weight'")

    risk = (
        "HIGH RISK" if fraud_prob >= 0.7
        else "MEDIUM RISK" if fraud_prob >= 0.3
        else "LOW RISK"
    )

    lines = [
        f"Fraud Risk Score : {fraud_prob * 100:.1f}%",
        f"Classification   : {risk}",
        f"Explanation method: {method}",
        "",
        "Top factors INCREASING fraud risk:",
    ]

    increasing = local_df[local_df[contrib_col] > 0].head(top_k)
    if increasing.empty:
        lines.append("  (none)")
    else:
        for _, row in increasing.iterrows():
            lines.append(
                f"  + {row['feature']:12s}  "
                f"({method} = {row[contrib_col]:+.4f}, value = {row['value']:.4f})"
            )

    lines.append("")
    lines.append("Top factors DECREASING fraud risk:")
    decreasing = local_df[local_df[contrib_col] < 0].head(top_k)
    if decreasing.empty:
        lines.append("  (none)")
    else:
        for _, row in decreasing.iterrows():
            lines.append(
                f"  - {row['feature']:12s}  "
                f"({method} = {row[contrib_col]:+.4f}, value = {row['value']:.4f})"
            )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Convenience wrappers that keep old call signatures working
# ---------------------------------------------------------------------------

# Backwards-compatible aliases used by the existing dashboard
make_explainer = make_shap_explainer
global_summary = shap_global_summary
local_explanation = shap_local_explanation
