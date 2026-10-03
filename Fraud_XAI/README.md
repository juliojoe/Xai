# Fraud_XAI

**An Explainable Machine Learning Framework for Detecting and Characterizing Financial Transaction Fraud**
Group 3 (S/NO 13-18) — MSc Computer Science group capstone.

Detects, scores, explains and compares financial transaction fraud using
three supervised models (Logistic Regression, Random Forest, XGBoost) and
two unsupervised models (Isolation Forest, Autoencoder), under three
imbalance-handling experiments (none, class weighting, SMOTE), with SHAP
explainability and a Streamlit prototype dashboard.

## Project structure

```
Fraud_XAI/
├── data/
│   ├── raw/            <- place creditcard.csv here (see "Dataset" below)
│   └── processed/       <- created by notebook 02 (train/test splits, sample)
├── notebooks/
│   ├── 01_data_analysis.ipynb      EDA
│   ├── 02_preprocessing.ipynb      cleaning, feature engineering, split
│   ├── 03_model_training.ipynb     5 models x 3 imbalance strategies
│   ├── 04_model_comparison.ipynb   metrics, CV, Wilcoxon/McNemar/DeLong, bootstrap CI
│   └── 05_shap_analysis.ipynb      global + local SHAP, exports dashboard artefacts
├── src/
│   ├── preprocessing.py    loading, cleaning, splitting
│   ├── features.py         feature engineering, scaling
│   ├── models.py           5 model builders + imbalance strategies
│   ├── evaluation.py       metrics + statistical tests (Wilcoxon, McNemar, DeLong, bootstrap CI)
│   └── explainability.py   SHAP global/local helpers
├── models/               <- created by notebook 03 (trained model artefacts)
├── dashboard/
│   └── app.py            Streamlit prototype: input -> risk score -> SHAP explanation
├── requirements.txt
└── README.md
```

## Dataset

Download the Kaggle **"Credit Card Fraud Detection"** dataset (ULB) and
place it at `data/raw/creditcard.csv`. If that file is missing, every
notebook and `src/preprocessing.py` automatically fall back to a small
synthetic stand-in dataset so the pipeline can still be exercised
end-to-end for testing — **replace it with the real file before recording
any results for your report; the synthetic numbers are not meaningful.**

## Setup

```bash
python -m venv venv
source venv/bin/activate        # venv\Scripts\activate on Windows
pip install -r requirements.txt
```

## Running the pipeline

Run the notebooks in order — each one saves artefacts the next one loads:

```
01_data_analysis.ipynb      (read-only, no artefacts saved)
02_preprocessing.ipynb      -> data/processed/*.csv, models/amount_time_scaler.joblib
03_model_training.ipynb     -> models/*.joblib (all 5 models x imbalance strategies)
04_model_comparison.ipynb   -> printed metrics tables + statistical test results
05_shap_analysis.ipynb      -> SHAP plots + data/processed/sample_test_transactions.csv
```

Then launch the dashboard:

```bash
cd dashboard
streamlit run app.py
```

## Reproducibility

Every model builder and split in `src/` fixes `RANDOM_STATE = 42`. Package
versions actually used to build and test this repository are pinned in
`requirements.txt`.

## Research questions this project answers

1. How effectively can machine-learning algorithms detect fraudulent
   financial transactions? — `notebooks/04_model_comparison.ipynb`
2. How does class-imbalance handling affect fraud-detection performance? —
   Experiment A/B/C comparison in `04_model_comparison.ipynb`
3. How do supervised algorithms compare with unsupervised anomaly
   detection? — same notebook, full results table
4. Which transaction features contribute most to fraud predictions? —
   `05_shap_analysis.ipynb`, global SHAP summary plot
5. How can XAI techniques explain individual fraud predictions? —
   `05_shap_analysis.ipynb` local explanation, and `dashboard/app.py`

## Limitations

See the "Error Analysis & Limitations" discussion produced at the end of
`04_model_comparison.ipynb`, and note in particular that `V1`-`V28` are
PCA-anonymized, which caps how business-interpretable any SHAP explanation
of them can be, and that the reference dataset is from 2013 and may not
reflect current fraud patterns.
