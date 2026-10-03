#!/usr/bin/env python3
"""
05_xai_shap_lime.py
===================
Self-explanatory demonstration of SHAP + LIME on the fraud model.

What this script does
---------------------
1. Loads the pre-trained XGBoost model and a small sample of test data.
2. Computes a GLOBAL SHAP summary (which features matter most overall).
3. Picks one high-risk transaction and explains it with BOTH:
   - SHAP  (game-theoretic attributions)
   - LIME  (local linear approximation)
4. Saves plots to reports/figures/ and prints a plain-English summary.

You do NOT need to re-train anything.  Just make sure the processed CSVs
and the model file exist (see instructions at the bottom).
"""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing import load_processed, get_feature_matrix
from src.explainability import (
    make_shap_explainer,
    shap_global_summary,
    shap_local_explanation,
    make_lime_explainer,
    lime_local_explanation,
    format_local_text,
    LIME_AVAILABLE,
)

FIG_DIR = PROJECT_ROOT / "reports" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def main():
    print("=" * 60)
    print("XAI DEMO – SHAP + LIME for Fraud Detection")
    print("=" * 60)

    # ------------------------------------------------------------------
    # 1. Load assets
    # ------------------------------------------------------------------
    print("\n[1/5] Loading processed data and model …")
    train, val, test, scaler = load_processed()
    X_train, y_train, feat_names = get_feature_matrix(train)
    X_test, y_test, _ = get_feature_matrix(test)

    model = joblib.load(PROJECT_ROOT / "models" / "xgboost_none.joblib")
    print(f"      Model loaded.  Features: {len(feat_names)}")
    print(f"      Test set size: {len(X_test):,}  (fraud rate ≈ {y_test.mean()*100:.3f}%)")

    # ------------------------------------------------------------------
    # 2. Global SHAP summary
    # ------------------------------------------------------------------
    print("\n[2/5] Computing global SHAP summary (this may take a minute) …")
    # Use a modest sample for speed
    sample_idx = np.random.RandomState(42).choice(len(X_test), size=min(500, len(X_test)), replace=False)
    X_sample = X_test[sample_idx]

    shap_explainer = make_shap_explainer(model, X_train[:200], feat_names, model_type="tree")
    shap_global_summary(
        shap_explainer,
        X_sample,
        feat_names,
        max_display=15,
        save_path=str(FIG_DIR / "05_shap_summary.png"),
    )
    print(f"      Saved → {FIG_DIR / '05_shap_summary.png'}")

    # ------------------------------------------------------------------
    # 3. Pick an interesting local example (high predicted fraud probability)
    # ------------------------------------------------------------------
    print("\n[3/5] Selecting a high-risk transaction for local explanation …")
    probs = model.predict_proba(X_test)[:, 1]
    # Prefer a true fraud that the model also scores high
    fraud_mask = y_test == 1
    if fraud_mask.any():
        fraud_probs = probs.copy()
        fraud_probs[~fraud_mask] = -1
        best_idx = int(np.argmax(fraud_probs))
    else:
        best_idx = int(np.argmax(probs))

    x_inst = X_test[best_idx]
    true_y = int(y_test[best_idx])
    pred_prob = float(probs[best_idx])
    print(f"      Index in test set : {best_idx}")
    print(f"      True label        : {'FRAUD' if true_y else 'Legitimate'}")
    print(f"      Model probability : {pred_prob*100:.1f}%")

    # ------------------------------------------------------------------
    # 4. Local SHAP
    # ------------------------------------------------------------------
    print("\n[4/5] Local SHAP explanation …")
    local_shap = shap_local_explanation(shap_explainer, x_inst, feat_names)
    print(format_local_text(local_shap, pred_prob, method="SHAP", top_k=5))

    # Bar plot
    fig, ax = plt.subplots(figsize=(9, 4.5))
    top = local_shap.head(12)
    colors = ["#e74c3c" if v > 0 else "#2ecc71" for v in top["shap"]]
    ax.barh(top["feature"][::-1], top["shap"][::-1], color=colors[::-1])
    ax.axvline(0, color="gray", lw=0.8)
    ax.set_xlabel("SHAP value (→ fraud)")
    ax.set_title(f"SHAP local explanation  |  P(fraud)={pred_prob*100:.1f}%")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "05_shap_local.png", dpi=130)
    plt.close(fig)
    print(f"      Saved → {FIG_DIR / '05_shap_local.png'}")

    # ------------------------------------------------------------------
    # 5. Local LIME
    # ------------------------------------------------------------------
    print("\n[5/5] Local LIME explanation …")
    if not LIME_AVAILABLE:
        print("      ⚠  lime package not installed.  Skipping LIME.")
        print("         Install with:  pip install lime")
    else:
        # Background sample for LIME (feature statistics)
        bg = X_train[:2000] if len(X_train) > 2000 else X_train
        lime_exp = make_lime_explainer(
            X_train=bg,
            feature_names=feat_names,
            class_names=["Legitimate", "Fraud"],
        )
        local_lime = lime_local_explanation(
            lime_exp, model, x_inst, feat_names, num_features=12
        )
        print(format_local_text(local_lime, pred_prob, method="LIME", top_k=5))

        fig, ax = plt.subplots(figsize=(9, 4.5))
        top = local_lime.head(12)
        colors = ["#e74c3c" if v > 0 else "#2ecc71" for v in top["lime_weight"]]
        ax.barh(top["feature"][::-1], top["lime_weight"][::-1], color=colors[::-1])
        ax.axvline(0, color="gray", lw=0.8)
        ax.set_xlabel("LIME weight (→ fraud)")
        ax.set_title(f"LIME local explanation  |  P(fraud)={pred_prob*100:.1f}%")
        fig.tight_layout()
        fig.savefig(FIG_DIR / "05_lime_local.png", dpi=130)
        plt.close(fig)
        print(f"      Saved → {FIG_DIR / '05_lime_local.png'}")

    print("\n" + "=" * 60)
    print("Done.  Open the PNG files in reports/figures/ to inspect the plots.")
    print("=" * 60)


if __name__ == "__main__":
    main()
