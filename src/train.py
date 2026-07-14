"""Train and select the best claim-denial model.

Example:

    python src/train.py --data_path claims_history.csv --model all --seed 42 \
        --tune --model_path outputs/model.pkl
"""

from __future__ import annotations

import argparse
import os

import joblib
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import average_precision_score, make_scorer

from .config import REVIEW_FRACTION, SEED
from .data import get_split, load_csv
from .evaluate import (
    evaluate_across_splits,
    metrics_at_topk,
    plot_evaluation,
    plot_feature_importance,
)
from .features import engineer_features
from .models import build_candidates, build_param_grids


def parse_args():
    p = argparse.ArgumentParser(
        description="Train claim-denial classifier(s)."
    )

    p.add_argument(
        "--data_path",
        default="claims_history.csv",
        help="Path to claims_history.csv",
    )

    p.add_argument(
        "--model",
        default="all",
        choices=["all", "logreg", "random_forest", "xgboost"],
        help="Which candidate(s) to train.",
    )

    p.add_argument(
        "--tune",
        action="store_true",
        help="Run GridSearchCV hyperparameter tuning.",
    )

    p.add_argument(
        "--seed",
        type=int,
        default=SEED,
    )

    p.add_argument(
        "--model_path",
        default="outputs/model.pkl",
        help="Where to save the selected model + threshold.",
    )

    p.add_argument(
        "--plots",
        action="store_true",
        help="Save evaluation/importance plots.",
    )

    return p.parse_args()

def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.model_path) or ".", exist_ok=True)

    df = engineer_features(load_csv(args.data_path))
    X_train, y_train = get_split(df, "train")
    X_val, y_val = get_split(df, "validation")
    X_test, y_test = get_split(df, "test")

    pos_weight = (y_train == 0).sum() / max(1, (y_train == 1).sum())

    candidates = build_candidates(pos_weight)
    if args.model != "all":
        candidates = {args.model: candidates[args.model]}

    if args.tune:
        scoring = {
            "pr_auc": make_scorer(
                average_precision_score,
                response_method="predict_proba",
            ),
            "roc_auc": "roc_auc",
            "accuracy": "accuracy",
        }

        grids = build_param_grids(pos_weight)
        if args.model != "all":
            grids = {args.model: grids[args.model]}

        tuned = {}
        for name, (pipe, grid) in grids.items():
            gs = GridSearchCV(
                pipe,
                grid,
                scoring=scoring,
                refit="pr_auc",
                cv=5,
                n_jobs=-1,
            )
            gs.fit(X_train, y_train)
            tuned[name] = gs.best_estimator_
            print(
                f"{name}: best CV PR-AUC={gs.best_score_:.3f} "
                f"params={gs.best_params_}"
            )
        candidates = tuned
    else:
        for name, model in candidates.items():
            model.fit(X_train, y_train)
            print(f"Trained: {name}")

    # Compare on validation and select the best by PR-AUC.
    splits = {
        "train": (X_train, y_train),
        "validation": (X_val, y_val),
        "test": (X_test, y_test),
    }

    report = evaluate_across_splits(candidates, splits)
    print("\n=== Metrics (train / validation / test) ===")
    print(report.round(3))

    val_scores = report.xs("validation", level="split")["pr_auc"].sort_values(
        ascending=False
    )
    best_name = val_scores.index[0]
    best_model = candidates[best_name]
    print(f"\nSelected model (best validation PR-AUC): {best_name}")

    # Freeze the top-25% operating threshold on validation (no leakage).
    val_prob = best_model.predict_proba(X_val)[:, 1]
    threshold = metrics_at_topk(
        y_val,
        val_prob,
        REVIEW_FRACTION,
    )["threshold@topk"]

    print(
        f"Chosen top-{int(REVIEW_FRACTION * 100)}% threshold = {threshold:.3f}"
    )

    if args.plots:
        os.makedirs("outputs", exist_ok=True)
        plot_evaluation(
            best_model,
            X_val,
            y_val,
            threshold,
            "outputs/evaluation.png",
        )
        plot_feature_importance(
            best_model,
            "outputs/feature_importance.png",
        )
        print("Saved plots to outputs/")

    joblib.dump(
        {
            "model": best_model,
            "threshold": threshold,
            "name": best_name,
        },
        args.model_path,
    )
    print(f"Saved model bundle to {args.model_path}")


if __name__ == "__main__":
    main()