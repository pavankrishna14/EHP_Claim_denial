"""Candidate models and their hyperparameter grids."""

from __future__ import annotations

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from .config import SEED
from .preprocessing import build_preprocessor


def make_pipeline(estimator) -> Pipeline:
    """Wrap a classifier with a fresh preprocessing step."""
    return Pipeline([
        ("prep", build_preprocessor()),
        ("clf", estimator),
    ])


def build_candidates(pos_weight: float) -> dict:
    """Return the baseline (untuned) candidate pipelines.

    `pos_weight = n_negative / n_positive` handles class imbalance for XGBoost;
    LogReg and RandomForest use `class_weight="balanced"` for the same purpose.
    """
    return {
        "logreg": make_pipeline(
            LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                random_state=SEED,
            )
        ),
        "random_forest": make_pipeline(
            RandomForestClassifier(
                n_estimators=300,
                max_depth=8,
                class_weight="balanced",
                random_state=SEED,
                n_jobs=-1,
            )
        ),
        "xgboost": make_pipeline(
            XGBClassifier(
                n_estimators=300,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.9,
                colsample_bytree=0.9,
                reg_lambda=5.0,
                scale_pos_weight=pos_weight,
                eval_metric="logloss",
                random_state=SEED,
                n_jobs=-1,
            )
        ),
    }


def build_param_grids(pos_weight: float) -> dict:
    """Return (pipeline, grid) pairs for GridSearchCV. `clf__` targets the classifier step."""
    return {
        "logreg": (
            make_pipeline(
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    random_state=SEED,
                )
            ),
            {
                "clf__C": [0.01, 0.1, 0.5, 1.0, 5.0],
                "clf__penalty": ["l2"],
                "clf__solver": ["lbfgs", "liblinear"],
            },
        ),
        "random_forest": (
            make_pipeline(
                RandomForestClassifier(
                    class_weight="balanced",
                    random_state=SEED,
                    n_jobs=-1,
                )
            ),
            {
                "clf__n_estimators": [200, 400],
                "clf__max_depth": [4, 6, 8, None],
                "clf__min_samples_leaf": [1, 5, 10],
            },
        ),
        "xgboost": (
            make_pipeline(
                XGBClassifier(
                    eval_metric="logloss",
                    random_state=SEED,
                    n_jobs=-1,
                )
            ),
            {
                "clf__n_estimators": [100, 300],
                "clf__max_depth": [2, 3, 4],
                "clf__learning_rate": [0.03, 0.05, 0.1],
                "clf__subsample": [0.7, 0.9],
                "clf__colsample_bytree": [0.7, 0.9],
                "clf__reg_lambda": [1.0, 5.0],
                "clf__scale_pos_weight": [1.0, pos_weight],
            },
        ),
    }