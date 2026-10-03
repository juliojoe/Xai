"""
preprocessing.py
-----------------
Data loading, integrity checks, and train/test splitting for the fraud
detection project. Kept deliberately separate from feature engineering
(features.py) so each module has one job.
"""

from pathlib import Path
from typing import Tuple
import joblib
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

RANDOM_STATE = 42


def load_data(path: str = "data/raw/creditcard.csv") -> pd.DataFrame:
    """
    Load the credit-card fraud dataset.

    If the real file isn't present, generate a small synthetic stand-in
    with the same column names and a similar (heavy) class imbalance, so
    the rest of the pipeline can still be exercised end-to-end for
    testing/demo purposes.

    IMPORTANT: replace with the real Kaggle "Credit Card Fraud Detection"
    (ULB) dataset before reporting any results -- synthetic numbers are
    not meaningful for your report.
    """
    if os.path.exists(path):
        df = pd.read_csv(path)
        print(f"Loaded real dataset from {path}: {df.shape[0]:,} rows, {df.shape[1]} columns")
        return df

    print(f"[preprocessing] '{path}' not found -- generating a small synthetic "
          "stand-in dataset for testing purposes only.")
    n_legit, n_fraud = 5000, 20
    n_total = n_legit + n_fraud
    rng = np.random.RandomState(RANDOM_STATE)

    v_cols = {f"V{i}": rng.normal(0, 1, n_total) for i in range(1, 29)}
    df = pd.DataFrame(v_cols)
    df["Time"] = np.sort(rng.uniform(0, 172792, n_total))
    df["Amount"] = np.abs(rng.normal(88, 250, n_total))
    df["Class"] = np.array([0] * n_legit + [1] * n_fraud)

    fraud_idx = df.index[df["Class"] == 1]
    df.loc[fraud_idx, "V14"] -= 4
    df.loc[fraud_idx, "V4"] += 3
    df = df.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
    return df


def data_quality_report(df: pd.DataFrame) -> dict:
    """Return a small dict summarising missing values, duplicates and class balance."""
    class_counts = df["Class"].value_counts().to_dict()
    fraud_rate = class_counts.get(1, 0) / len(df) * 100
    return {
        "n_rows": len(df),
        "n_missing_cells": int(df.isnull().sum().sum()),
        "n_duplicate_rows": int(df.duplicated().sum()),
        "class_counts": class_counts,
        "fraud_rate_pct": fraud_rate,
    }


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Drop exact duplicate rows and any rows missing the target label."""
    cleaned = df.drop_duplicates().copy()
    cleaned = cleaned.dropna(subset=["Class"])
    return cleaned.reset_index(drop=True)


def split_data(X: pd.DataFrame, y: pd.Series, test_size: float = 0.2):
    """Stratified train/test split so the tiny fraud class stays proportionally represented."""
    return train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=RANDOM_STATE
    )

#new 


def load_processed(
    out_dir: Path = PROCESSED_DIR,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, object]:
    """
    Load processed data.

    Supports two layouts:
    A) Combined files: train.csv, val.csv, test.csv  (features + Class)
    B) Split files:    X_train.csv, y_train.csv, X_test.csv, y_test.csv
    """
    out_dir = Path(out_dir)
    scaler_path = out_dir / "scaler.joblib"
    scaler = joblib.load(scaler_path) if scaler_path.exists() else None

    # Layout A: combined CSVs
    if (out_dir / "train.csv").exists() and (out_dir / "test.csv").exists():
        train = pd.read_csv(out_dir / "train.csv")
        test = pd.read_csv(out_dir / "test.csv")
        val = pd.read_csv(out_dir / "val.csv") if (out_dir / "val.csv").exists() else test.copy()
        return train, val, test, scaler

    # Layout B: separate X_ / y_ files (your current format)
    if (out_dir / "X_train.csv").exists() and (out_dir / "y_train.csv").exists():
        X_train = pd.read_csv(out_dir / "X_train.csv")
        y_train = pd.read_csv(out_dir / "y_train.csv")
        if y_train.shape[1] == 1:
            y_train.columns = ["Class"]
        train = X_train.copy()
        train["Class"] = y_train.iloc[:, 0].values

        X_test = pd.read_csv(out_dir / "X_test.csv")
        y_test = pd.read_csv(out_dir / "y_test.csv")
        if y_test.shape[1] == 1:
            y_test.columns = ["Class"]
        test = X_test.copy()
        test["Class"] = y_test.iloc[:, 0].values

        if (out_dir / "X_val.csv").exists() and (out_dir / "y_val.csv").exists():
            X_val = pd.read_csv(out_dir / "X_val.csv")
            y_val = pd.read_csv(out_dir / "y_val.csv")
            if y_val.shape[1] == 1:
                y_val.columns = ["Class"]
            val = X_val.copy()
            val["Class"] = y_val.iloc[:, 0].values
        else:
            val = test.copy()

        return train, val, test, scaler

    raise FileNotFoundError(
        f"No processed data found in {out_dir}.\n"
        "Expected either:\n"
        "  - train.csv, test.csv\n"
        "  - X_train.csv, y_train.csv, X_test.csv, y_test.csv"
    )
def get_feature_matrix(
    df: pd.DataFrame, drop_cols: list[str] | None = None
) -> Tuple[np.ndarray, np.ndarray, list[str]]:
    """Return X, y and feature names, dropping Class and any extra columns."""
    if drop_cols is None:
        drop_cols = ["Class"]
    feature_cols = [c for c in df.columns if c not in drop_cols]
    X = df[feature_cols].values
    y = df["Class"].values if "Class" in df.columns else None
    return X, y, feature_cols