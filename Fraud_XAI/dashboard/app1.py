"""
Streamlit prototype: Transaction → Fraud Risk Score → SHAP & LIME explanations

XAI is intentionally self-explanatory:
- SHAP shows game-theoretic feature contributions
- LIME shows a local linear approximation of the model
Both answer "Why did the model give this score?"
"""

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import shap
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing import load_processed, get_feature_matrix
from src.explainability import (
    shap_local_explanation,
    lime_local_explanation,
    make_lime_explainer,
    format_local_text,
    LIME_AVAILABLE,
)
from src.evaluation import risk_label

st.set_page_config(page_title="Fraud XAI Dashboard", layout="wide")
st.title("Explainable Fraud Detection – Risk Scoring Prototype")
st.caption(
    "This dashboard uses two complementary XAI methods: "
    "**SHAP** (global consistency) and **LIME** (local model-agnostic approximation)."
)


@st.cache_resource
def load_assets():
    """Load model, test data and build explainers once."""
    train, val, test, scaler = load_processed()
    X_train, _, feat_names = get_feature_matrix(train)
    model = joblib.load(PROJECT_ROOT / "models" / "xgboost_none.joblib")

    shap_explainer = shap.TreeExplainer(model)

    lime_explainer = None
    if LIME_AVAILABLE:
        bg = X_train[:2000] if len(X_train) > 2000 else X_train
        lime_explainer = make_lime_explainer(
            X_train=bg,
            feature_names=feat_names,
            class_names=["Legitimate", "Fraud"],
        )

    return model, shap_explainer, lime_explainer, feat_names, test, X_train


try:
    model, shap_explainer, lime_explainer, feat_names, test_df, X_train = load_assets()
except FileNotFoundError as e:
    st.error(
        "Processed data or model not found.\n\n"
        f"Details: {e}"
    )
    st.stop()


# ------------------------------------------------------------------
# Sidebar – transaction selection (test set only)
# ------------------------------------------------------------------
st.sidebar.header("Transaction Input")
idx = st.sidebar.number_input("Test-set index", 0, len(test_df) - 1, 0)
row = test_df.iloc[int(idx)]
# Keep only the exact feature columns the model was trained on
X = row[feat_names].values.reshape(1, -1)
true_label = int(row["Class"])
st.sidebar.write(f"True label: {'FRAUD' if true_label else 'Legitimate'}")

xai_method = st.sidebar.multiselect(
    "Explanation method(s)",
    options=["SHAP", "LIME"],
    default=["SHAP", "LIME"] if LIME_AVAILABLE else ["SHAP"],
)

if not LIME_AVAILABLE and "LIME" in xai_method:
    st.sidebar.warning("LIME package not installed.  Run: `pip install lime`")


# ------------------------------------------------------------------
# Main analysis
# ------------------------------------------------------------------
if st.sidebar.button("Analyze Transaction", type="primary"):
    prob = float(model.predict_proba(X)[0, 1])
    pred = int(prob >= 0.5)
    label = risk_label(prob)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Fraud Probability", f"{prob * 100:.1f}%")
    col2.metric("Risk Band", label)
    col3.metric("Predicted Class", "FRAUD" if pred else "Legitimate")
    col4.metric("Actual Class", "FRAUD" if true_label else "Legitimate")

    st.markdown("---")

    if "SHAP" in xai_method:
        st.subheader("SHAP Local Explanation")
        st.markdown(
            "*SHAP attributes the prediction to each feature using Shapley values "
            "from cooperative game theory.  Positive values push toward fraud.*"
        )
        local_shap = shap_local_explanation(
            shap_explainer, X.flatten(), feat_names
        )
        st.text(format_local_text(local_shap, prob, method="SHAP", top_k=5))

        fig, ax = plt.subplots(figsize=(9, 4.5))
        top = local_shap.head(12)
        colors = ["#e74c3c" if v > 0 else "#2ecc71" for v in top["shap"]]
        ax.barh(top["feature"][::-1], top["shap"][::-1], color=colors[::-1])
        ax.axvline(0, color="gray", lw=0.8)
        ax.set_xlabel("SHAP value  (→ fraud)")
        ax.set_title("SHAP feature contributions for this transaction")
        st.pyplot(fig)
        plt.close(fig)

    if "LIME" in xai_method:
        if not LIME_AVAILABLE or lime_explainer is None:
            st.warning("LIME is not available.  Install with: `pip install lime`")
        else:
            st.subheader("LIME Local Explanation")
            st.markdown(
                "*LIME builds a simple linear model around this single transaction.  "
                "Positive weight → increases fraud probability.*"
            )
            with st.spinner("Computing LIME explanation (a few seconds)…"):
                local_lime = lime_local_explanation(
                    lime_explainer,
                    model,
                    X.flatten(),
                    feat_names,
                    num_features=12,
                )
            st.text(format_local_text(local_lime, prob, method="LIME", top_k=5))

            fig, ax = plt.subplots(figsize=(9, 4.5))
            top = local_lime.head(12)
            colors = ["#e74c3c" if v > 0 else "#2ecc71" for v in top["lime_weight"]]
            ax.barh(top["feature"][::-1], top["lime_weight"][::-1], color=colors[::-1])
            ax.axvline(0, color="gray", lw=0.8)
            ax.set_xlabel("LIME weight  (→ fraud)")
            ax.set_title("LIME feature contributions for this transaction")
            st.pyplot(fig)
            plt.close(fig)

    if "SHAP" in xai_method and "LIME" in xai_method and LIME_AVAILABLE:
        st.subheader("Quick comparison")
        st.info(
            "SHAP and LIME often agree on the most important features, "
            "but the exact magnitudes differ because they use different assumptions.  "
            "Agreement on the top drivers increases trust in the explanation."
        )

else:
    st.info("Choose a test-set index in the sidebar and click **Analyze Transaction**.")