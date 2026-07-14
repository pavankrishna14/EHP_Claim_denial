"""Score current_claims.csv, assign risk tiers, generate explanations, export CSV.

Example:
    python src/predict.py --model_path outputs/model.pkl \
        --data_path current_claims.csv --output predictions_current_claims.csv
"""

from __future__ import annotations

import argparse
import joblib

from .config import FEATURE_COLS, MEDIUM_TIER_MULTIPLIER
from .data import load_csv
from .explain import explain_row, top_risk_factors
from .features import engineer_features


def parse_args():
    p = argparse.ArgumentParser(
        description="Score current claims and generate explanations."
    )
    p.add_argument("--model_path", default="outputs/model.pkl")
    p.add_argument("--data_path", default="current_claims.csv")
    p.add_argument("--output", default="predictions_current_claims.csv")
    p.add_argument(
        "--top_n",
        type=int,
        default=10,
        help="How many top claims get an explanation.",
    )
    return p.parse_args()


def assign_tier(p: float, threshold: float) -> str:
    if p >= threshold:
        return "High"
    if p >= MEDIUM_TIER_MULTIPLIER * threshold:
        return "Medium"
    return "Low"


def main():
    args = parse_args()

    bundle = joblib.load(args.model_path)
    model, threshold = bundle["model"], bundle["threshold"]

    df = engineer_features(load_csv(args.data_path))
    prob = model.predict_proba(df[FEATURE_COLS])[:, 1]

    scored = df.copy()
    scored["denial_probability"] = prob
    scored["predicted_denial"] = (prob >= threshold).astype(int)
    scored["risk_tier"] = [
        assign_tier(p, threshold) for p in prob
    ]
    scored["top_risk_factors"] = [
        top_risk_factors(scored.iloc[i])
        for i in range(len(scored))
    ]

    scored = (
        scored.sort_values(
            "denial_probability",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    # Explanations only for the top-N highest-risk claims.
    top = scored.head(args.top_n)
    exp_map = {
        row["claim_id"]: explain_row(row)
        for _, row in top.iterrows()
    }

    scored["explanation"] = (
        scored["claim_id"]
        .map(exp_map)
        .fillna("")
    )

    out = scored[
        [
            "claim_id",
            "denial_probability",
            "predicted_denial",
            "risk_tier",
            "top_risk_factors",
            "explanation",
        ]
    ].copy()

    out["top_risk_factors"] = out["top_risk_factors"].apply(
        lambda lst: " | ".join(lst)
    )
    out["denial_probability"] = out["denial_probability"].round(4)

    out.to_csv(args.output, index=False)

    print(f"Wrote {args.output} with {len(out)} rows.")
    print(f"Risk tier distribution:\n{out['risk_tier'].value_counts()}")


if __name__ == "__main__":
    main()