"""
features.py
-----------
Feature engineering for the fraud detection project.

V1-V28 in the ULB dataset are already PCA components (anonymized by the
dataset publishers), so there is limited room for manual engineering
there. We focus on the two raw features (Amount, Time) and derive an
hour-of-day feature, a common signal in fraud data since fraud often
clusters at unusual hours. Every feature added here has a stated reason
tied to fraud behaviour, rather than being added just to pad the list.
"""

import pandas as pd
from sklearn.preprocessing import StandardScaler


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add an hour-of-day feature derived from Time (seconds since first transaction)."""
    out = df.copy()
    out["Hour"] = (out["Time"] // 3600) % 24
    return out


def get_feature_columns(df: pd.DataFrame) -> list:
    """V1-V28 plus the engineered/raw features used by the models."""
    v_cols = [c for c in df.columns if c.startswith("V")]
    return v_cols + ["Amount_scaled", "Time_scaled", "Hour"]


def fit_scaler(X_train: pd.DataFrame) -> StandardScaler:
    """Fit a scaler for Amount/Time on the TRAINING split only, to avoid leakage."""
    scaler = StandardScaler()
    scaler.fit(X_train[["Amount", "Time"]])
    return scaler


def apply_scaler(df: pd.DataFrame, scaler: StandardScaler) -> pd.DataFrame:
    """Apply a previously-fit scaler to Amount/Time and return a df with the scaled columns added."""
    out = df.copy()
    out[["Amount_scaled", "Time_scaled"]] = scaler.transform(out[["Amount", "Time"]])
    return out
