"""Data loading and split helpers."""

from __future__ import annotations

import pandas as pd

from .config import FEATURE_COLS, TARGET


def load_csv(path: str) -> pd.DataFrame:
    """Load a claims CSV into a DataFrame."""
    return pd.read_csv(path)


def get_split(df: pd.DataFrame, split_name: str):
    """Return (X, y) for a given split using the provided 'split' column."""
    sub = df[df["split"] == split_name]
    return sub[FEATURE_COLS], sub[TARGET]