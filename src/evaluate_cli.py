"""Evaluate a saved model on a labeled dataset.

Example:
    python src/evaluate.py --model_path outputs/model.pkl --data_path claims_history.csv
"""

from __future__ import annotations

import argparse
import joblib

from .data import get_split, load_csv
from .evaluate import evaluate_across_splits
from .features import engineer_features


def parse_args():
    p = argparse.ArgumentParser(
        description="Evaluate a saved claim-denial model."
    )
    p.add_argument("--model_path", default="outputs/model.pkl")
    p.add_argument("--data_path", default="claims_history.csv")
    return p.parse_args()


def main():
    args = parse_args()

    bundle = joblib.load(args.model_path)
    model = bundle["model"]

    print(
        f"Loaded model: {bundle.get('name')} "
        f"(threshold={bundle.get('threshold'):.3f})"
    )

    df = engineer_features(load_csv(args.data_path))

    splits = {}
    for split_name in ["train", "validation", "test"]:
        X, y = get_split(df, split_name)
        if len(X):
            splits[split_name] = (X, y)

    report = evaluate_across_splits(
        {bundle.get("name", "model"): model},
        splits,
    )

    print(report.round(3))


if __name__ == "__main__":
    main()