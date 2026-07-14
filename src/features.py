"""Domain-specific feature engineering.

Denials are frequently driven by *gaps* between what a payer requires and what
is actually on file. We encode those gaps explicitly so both linear and tree
models can use them directly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add domain gap/ratio features. Safe to call on history and current data."""
    df = df.copy()

    df["auth_gap"] = (
        ((df["prior_auth_required"] == 1) & (df["has_prior_auth"] == 0))
        .astype(int)
    )

    df["referral_gap"] = (
        ((df["referral_required"] == 1) & (df["referral_present"] == 0))
        .astype(int)
    )

    df["out_of_network"] = (
        (1 - df["is_in_network"]).astype(int)
    )

    df["eligibility_unverified"] = (
        (1 - df["eligibility_verified"]).astype(int)
    )

    df["billed_to_expected_ratio"] = (
        df["total_billed"]
        / df["expected_payment"].replace(0, np.nan)
    ).fillna(1.0)

    return df