"""
models.py
---------
Model builders for the five competing models, plus the three
imbalance-handling experiments (A: none, B: class weighting, C: SMOTE)
described in the methodology.

IMPORTANT (data leakage): SMOTE must only ever be fit on the TRAINING
split, never on the full dataset before splitting. `apply_imbalance_strategy`
below enforces this by only ever touching X_train/y_train.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.neural_network import MLPRegressor
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

RANDOM_STATE = 42

IMBALANCE_STRATEGIES = ("none", "class_weight", "smote")


def apply_imbalance_strategy(X_train, y_train, strategy: str):
    """
    Experiment A/B/C from the methodology:
      - 'none'         : no treatment, train on the raw imbalanced data
      - 'class_weight' : keep raw data, but the model builders below apply
                         class_weight='balanced' / scale_pos_weight
      - 'smote'         : oversample the minority (fraud) class synthetically

    Returns (X_resampled, y_resampled). For 'none' and 'class_weight' this
    is just the original training data unchanged -- the weighting itself
    is applied inside the model builders, not to the data.
    """
    if strategy not in IMBALANCE_STRATEGIES:
        raise ValueError(f"Unknown strategy '{strategy}', expected one of {IMBALANCE_STRATEGIES}")

    if strategy == "smote":
        sm = SMOTE(random_state=RANDOM_STATE)
        return sm.fit_resample(X_train, y_train)

    return X_train, y_train


def build_logistic_regression(strategy: str) -> LogisticRegression:
    class_weight = "balanced" if strategy == "class_weight" else None
    return LogisticRegression(max_iter=1000, class_weight=class_weight, random_state=RANDOM_STATE)


def build_random_forest(strategy: str) -> RandomForestClassifier:
    class_weight = "balanced" if strategy == "class_weight" else None
    return RandomForestClassifier(
        n_estimators=200, class_weight=class_weight, random_state=RANDOM_STATE, n_jobs=-1
    )


def build_xgboost(strategy: str, y_train=None) -> XGBClassifier:
    scale_pos_weight = 1.0
    if strategy == "class_weight" and y_train is not None:
        pos = max((y_train == 1).sum(), 1)
        neg = (y_train == 0).sum()
        scale_pos_weight = neg / pos
    return XGBClassifier(
        n_estimators=300, max_depth=5, learning_rate=0.1, eval_metric="logloss",
        scale_pos_weight=scale_pos_weight, random_state=RANDOM_STATE, n_jobs=-1,
    )


def build_isolation_forest(contamination: float) -> IsolationForest:
    """Unsupervised -- trained without labels, so imbalance strategy does not apply."""
    return IsolationForest(
        n_estimators=200, contamination=contamination, random_state=RANDOM_STATE, n_jobs=-1
    )


def build_autoencoder() -> MLPRegressor:
    """
    Lightweight Autoencoder approximation using scikit-learn's MLPRegressor
    set up as an encoder-decoder (narrow bottleneck hidden layer). Trained
    to reconstruct its own input; used only on legitimate transactions so
    it learns "normal", and fraud shows up as high reconstruction error.
    Swap for a Keras/TensorFlow model for a true deep Autoencoder if your
    group has GPU access -- the rest of the pipeline is unaffected.
    """
    return MLPRegressor(
        hidden_layer_sizes=(16, 8, 16), activation="relu", max_iter=200, random_state=RANDOM_STATE
    )


def train_supervised_models(X_train, y_train, strategy: str) -> dict:
    """Train Logistic Regression, Random Forest and XGBoost under one imbalance strategy."""
    X_res, y_res = apply_imbalance_strategy(X_train, y_train, strategy)

    log_reg = build_logistic_regression(strategy)
    log_reg.fit(X_res, y_res)

    rf = build_random_forest(strategy)
    rf.fit(X_res, y_res)

    xgb = build_xgboost(strategy, y_train=y_res)
    xgb.fit(X_res, y_res)

    return {"logistic_regression": log_reg, "random_forest": rf, "xgboost": xgb}


def train_isolation_forest(X_train, y_train) -> IsolationForest:
    model = build_isolation_forest(contamination=float(np.mean(y_train)))
    model.fit(X_train)
    return model


def isolation_forest_scores(model: IsolationForest, X):
    """Return (predicted_label, fraud_score) with the convention 1=fraud, higher score = more fraud-like."""
    pred = np.where(model.predict(X) == -1, 1, 0)
    score = -model.decision_function(X)  # flip sign: higher = more anomalous
    return pred, score


def train_autoencoder(X_train, y_train) -> MLPRegressor:
    model = build_autoencoder()
    X_legit = X_train[y_train == 0]
    model.fit(X_legit, X_legit)
    return model


def autoencoder_scores(model: MLPRegressor, X, fraud_rate: float):
    """Reconstruction error as the fraud score; threshold set at the (1 - fraud_rate) quantile."""
    reconstructed = model.predict(X)
    error = np.mean((X.values - reconstructed) ** 2, axis=1)
    threshold = np.quantile(error, 1 - fraud_rate)
    pred = (error > threshold).astype(int)
    return pred, error
