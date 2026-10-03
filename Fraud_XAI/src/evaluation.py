"""
evaluation.py
-------------
Metrics and statistical-comparison utilities.

Accuracy is deliberately NOT the headline metric anywhere in this module --
at a ~0.17% fraud rate a model that always predicts "legitimate" would
still look ~99.8% "accurate", so it is excluded from `compute_metrics`.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    roc_auc_score, precision_recall_curve, auc, confusion_matrix,
)
from sklearn.model_selection import StratifiedKFold
from scipy.stats import wilcoxon
from statsmodels.stats.contingency_tables import mcnemar

RANDOM_STATE = 42


# ---------------------------------------------------------------------------
# Section 8: core metrics
# ---------------------------------------------------------------------------
def compute_metrics(y_true, y_pred, y_score, name: str = "") -> dict:
    """Precision, Recall, F1, ROC-AUC, PR-AUC and the confusion matrix for one model."""
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    try:
        roc_auc = roc_auc_score(y_true, y_score)
    except ValueError:
        roc_auc = float("nan")  # only one class present, can happen on tiny demo data

    p_curve, r_curve, _ = precision_recall_curve(y_true, y_score)
    pr_auc = auc(r_curve, p_curve)

    cm = confusion_matrix(y_true, y_pred)

    return {
        "model": name, "precision": precision, "recall": recall, "f1": f1,
        "roc_auc": roc_auc, "pr_auc": pr_auc, "confusion_matrix": cm,
    }


def metrics_table(metric_dicts: list) -> pd.DataFrame:
    """Turn a list of compute_metrics() outputs into a tidy comparison table (drops the raw matrix)."""
    rows = [{k: v for k, v in d.items() if k != "confusion_matrix"} for d in metric_dicts]
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Section 9: cross-validation
# ---------------------------------------------------------------------------
def stratified_cv_f1(model_builder, imbalance_fn, X, y, n_splits: int = 5) -> list:
    """
    Stratified k-fold CV. `model_builder()` returns a fresh, untrained model.
    `imbalance_fn(X_tr, y_tr)` applies the chosen imbalance strategy to the
    TRAINING fold only (never the validation fold) -- this both avoids
    leakage and matches the "split first, then SMOTE" rule in the brief.
    Returns the list of per-fold F1 scores.
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    scores = []
    for train_idx, val_idx in skf.split(X, y):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

        X_tr_bal, y_tr_bal = imbalance_fn(X_tr, y_tr)

        model = model_builder()
        model.fit(X_tr_bal, y_tr_bal)
        preds = model.predict(X_val)
        scores.append(f1_score(y_val, preds, zero_division=0))
    return scores


# ---------------------------------------------------------------------------
# Section 9: paired statistical tests
# ---------------------------------------------------------------------------
def wilcoxon_test(scores_a: list, scores_b: list):
    """
    Wilcoxon signed-rank test on paired per-fold scores (e.g. CV F1 scores
    for two models). H0: no real difference between the two models --
    a small p-value (< 0.05 conventionally) lets you reject H0.
    """
    return wilcoxon(scores_a, scores_b)


def mcnemar_test(y_true, pred_a, pred_b):
    """
    McNemar's test on paired classification predictions from two models on
    the SAME test set. Builds the 2x2 table of (A correct/incorrect) x
    (B correct/incorrect) discordant pairs and tests whether the two
    models disagree symmetrically (H0) or one is systematically better.
    """
    a_correct = (np.asarray(pred_a) == np.asarray(y_true))
    b_correct = (np.asarray(pred_b) == np.asarray(y_true))

    both_correct = int(np.sum(a_correct & b_correct))
    a_only = int(np.sum(a_correct & ~b_correct))
    b_only = int(np.sum(~a_correct & b_correct))
    both_wrong = int(np.sum(~a_correct & ~b_correct))

    table = [[both_correct, a_only], [b_only, both_wrong]]
    result = mcnemar(table, exact=(a_only + b_only) < 25, correction=True)
    return result, table


def bootstrap_ci(y_true, y_score, metric_fn=roc_auc_score, n_boot: int = 1000, alpha: float = 0.05):
    """
    Bootstrap confidence interval for a scoring metric (default ROC-AUC).
    Resamples (y_true, y_score) pairs with replacement `n_boot` times and
    returns (point_estimate, lower_ci, upper_ci).
    """
    rng = np.random.RandomState(RANDOM_STATE)
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    n = len(y_true)

    boot_scores = []
    for _ in range(n_boot):
        idx = rng.randint(0, n, n)
        y_t, y_s = y_true[idx], y_score[idx]
        if len(np.unique(y_t)) < 2:
            continue  # skip resamples with only one class present
        boot_scores.append(metric_fn(y_t, y_s))

    boot_scores = np.array(boot_scores)
    point_estimate = metric_fn(y_true, y_score)
    lower = np.percentile(boot_scores, 100 * alpha / 2)
    upper = np.percentile(boot_scores, 100 * (1 - alpha / 2))
    return point_estimate, lower, upper


# ---------------------------------------------------------------------------
# Section 9: DeLong test for comparing two ROC-AUCs on the same test set
# ---------------------------------------------------------------------------
# Standard DeLong (1988) implementation via the structural components /
# Mann-Whitney U relationship (fast O(n log n) version, following the
# widely-used formulation of Sun & Xu, 2014).

def _compute_midrank(x):
    J = np.argsort(x)
    Z = x[J]
    N = len(x)
    T = np.zeros(N, dtype=float)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    T2 = np.empty(N, dtype=float)
    T2[J] = T
    return T2


def _fast_delong(predictions_sorted_transposed, label_1_count):
    m = label_1_count
    n = predictions_sorted_transposed.shape[1] - m
    positive_examples = predictions_sorted_transposed[:, :m]
    negative_examples = predictions_sorted_transposed[:, m:]
    k = predictions_sorted_transposed.shape[0]

    tx = np.empty([k, m], dtype=float)
    ty = np.empty([k, n], dtype=float)
    tz = np.empty([k, m + n], dtype=float)
    for r in range(k):
        tx[r, :] = _compute_midrank(positive_examples[r, :])
        ty[r, :] = _compute_midrank(negative_examples[r, :])
        tz[r, :] = _compute_midrank(predictions_sorted_transposed[r, :])

    aucs = tz[:, :m].sum(axis=1) / m / n - float(m + 1.0) / (2.0 * n)
    v01 = (tz[:, :m] - tx[:, :]) / n
    v10 = 1.0 - (tz[:, m:] - ty[:, :]) / m
    sx = np.cov(v01)
    sy = np.cov(v10)
    delongcov = sx / m + sy / n
    return aucs, delongcov


def delong_roc_test(y_true, score_a, score_b):
    """
    DeLong's test for two correlated ROC-AUCs computed on the SAME test set
    (e.g. XGBoost's predicted probabilities vs. Random Forest's, on the
    same y_test). Returns (auc_a, auc_b, z_statistic, p_value).
    """
    y_true = np.asarray(y_true)
    order = np.argsort(-y_true, kind="mergesort")  # positives (1) first
    y_sorted = y_true[order]
    label_1_count = int(np.sum(y_sorted))

    preds = np.vstack([np.asarray(score_a)[order], np.asarray(score_b)[order]])
    aucs, cov = _fast_delong(preds, label_1_count)

    auc_diff = aucs[0] - aucs[1]
    var = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    var = max(var, 1e-12)  # guard against tiny negative values from floating-point noise
    z = auc_diff / np.sqrt(var)
    from scipy.stats import norm
    p_value = 2 * (1 - norm.cdf(abs(z)))
    return aucs[0], aucs[1], z, p_value


def risk_label(prob: float) -> str:
    """Map probability / anomaly score to a human-readable risk band."""
    if prob >= 0.7:
        return "HIGH RISK"
    if prob >= 0.3:
        return "MEDIUM RISK"
    return "LOW RISK"