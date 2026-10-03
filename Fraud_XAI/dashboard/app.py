"""
Streamlit dashboard -- fraud-risk scoring system with an explanation panel.

Three sections, matching the methodology's prototype spec:
  1. Transaction input (pick one from the saved test sample)
  2. Result (fraud risk score + HIGH/LOW RISK classification)
  3. Explanation (SHAP force plot: why this score)

Run notebooks 02, 03 and 05 first (they save the model, scaler and a
transaction sample this app loads), then:

    streamlit run dashboard/app.py
"""

import sys
import os

sys.path.insert(0, os.path.abspath(".."))

import joblib
import pandas as pd
import shap
import streamlit as st
import matplotlib.pyplot as plt

from src import explainability

st.set_page_config(page_title="Fraud Risk Scoring Dashboard", layout="wide")

st.title("Fraud-Risk Scoring System")
st.caption(
    "Group 3 capstone prototype — pick a transaction, see its fraud-risk "
    "score and the explanation behind that score."
)


@st.cache_resource
def load_artifacts():
    model = joblib.load("../models/xgboost_smote.joblib")
    sample = pd.read_csv("../data/processed/sample_test_transactions.csv")
    explainer = explainability.get_tree_explainer(model)
    return model, sample, explainer


try:
    model, sample_transactions, explainer = load_artifacts()
except FileNotFoundError:
    st.error(
        "Model artefacts not found. Run notebooks 02_preprocessing, "
        "03_model_training and 05_shap_analysis first — they save "
        "models/xgboost_smote.joblib and "
        "data/processed/sample_test_transactions.csv, which this app needs."
    )
    st.stop()

# ---------------------------------------------------------------------------
# Section 1 — Transaction input
# ---------------------------------------------------------------------------
st.header("1. Transaction Input")
row_index = st.selectbox(
    "Choose a transaction (row index from the held-out test sample):",
    options=list(sample_transactions.index),
)
transaction = sample_transactions.loc[[row_index]]
st.dataframe(transaction)

# ---------------------------------------------------------------------------
# Section 2 — Result
# ---------------------------------------------------------------------------
st.header("2. Result")
risk_score = float(model.predict_proba(transaction)[0, 1])

col1, col2 = st.columns([1, 3])
with col1:
    st.metric("Fraud Risk Score", f"{risk_score:.0%}")
with col2:
    if risk_score >= 0.5:
        st.error("Classification: HIGH RISK")
    else:
        st.success("Classification: LOW RISK")

st.caption(
    "This score reflects the model's learned pattern on this dataset — it "
    "has not been calibrated or validated as a real-world probability of "
    "fraud, and should not be presented to a bank or customer as one."
)

# ---------------------------------------------------------------------------
# Section 3 — Explanation
# ---------------------------------------------------------------------------
st.header("3. Why was this transaction flagged?")

top_features = explainability.top_shap_features(explainer, transaction, top_n=5)
for feature, value in top_features.items():
    direction = "increased" if value > 0 else "decreased"
    st.write(f"- **{feature}** {direction} the fraud score (SHAP value: {value:+.3f})")

shap_values = explainability.local_explanation(explainer, transaction)
fig = plt.figure(figsize=(8, 4))
shap.force_plot(
    explainer.expected_value, shap_values[0], transaction,
    matplotlib=True, show=False,
)
st.pyplot(plt.gcf())
plt.close()

st.caption(
    "Red features push the score toward fraud; blue features push it "
    "toward legitimate. Bar width shows the size of each feature's effect "
    "on this specific prediction."
)
