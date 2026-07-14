"""Review-constraint-aware evaluation metrics and plots."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from .config import REVIEW_FRACTION


def metrics_at_topk(
    y_true,
    y_prob,
    fraction: float = REVIEW_FRACTION,
) -> dict:
    """Precision/recall when we flag the top `fraction` of ranked claims.

    Mirrors the review team's workflow: they can only inspect the top X% by risk.
    """
    y_true = np.asarray(y_true)
    n_flag = max(1, int(np.ceil(len(y_true) * fraction)))

    order = np.argsort(y_prob)[::-1]
    pred = np.zeros_like(y_true)
    pred[order[:n_flag]] = 1

    return {
        "precision@topk": precision_score(
            y_true, pred, zero_division=0
        ),
        "recall@topk": recall_score(
            y_true, pred, zero_division=0
        ),
        "threshold@topk": float(
            np.sort(y_prob)[::-1][n_flag - 1]
        ),
    }


def evaluate(model, X, y) -> dict:
    """Return ROC-AUC, PR-AUC and top-k precision/recall for a fitted model."""
    prob = model.predict_proba(X)[:, 1]

    out = {
        "roc_auc": roc_auc_score(y, prob),
        "pr_auc": average_precision_score(y, prob),
    }

    out.update(metrics_at_topk(y, prob))
    return out


def evaluate_across_splits(
    models: dict,
    splits: dict,
) -> pd.DataFrame:
    """Tidy metrics table for {name: model} over {split_name: (X, y)}."""
    rows = []

    for name, model in models.items():
        for split_name, (X, y) in splits.items():
            m = evaluate(model, X, y)
            m.update({"model": name, "split": split_name})
            rows.append(m)
    return pd.DataFrame(rows).set_index(["model", "split"])


def plot_evaluation(
    model,
    X,
    y,
    threshold: float,
    out_path: str | None = None,
):
    """ROC curve, PR curve, and confusion matrix at the top-k threshold."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    prob = model.predict_proba(X)[:, 1]
    pred = (prob >= threshold).astype(int)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    fpr, tpr, _ = roc_curve(y, prob)
    axes[0].plot(
        fpr,
        tpr,
        label=f"AUC={roc_auc_score(y, prob):.3f}",
    )
    axes[0].plot([0, 1], [0, 1], "k--", lw=0.8)
    axes[0].set(
        title="ROC Curve",
        xlabel="FPR",
        ylabel="TPR",
    )
    axes[0].legend()

    prec, rec, _ = precision_recall_curve(y, prob)
    axes[1].plot(
        rec,
        prec,
        label=f"PR-AUC={average_precision_score(y, prob):.3f}",
    )
    axes[1].axhline(
        np.mean(y),
        color="grey",
        ls="--",
        label="baseline",
    )
    axes[1].set(
        title="Precision-Recall Curve",
        xlabel="Recall",
        ylabel="Precision",
    )
    axes[1].legend()

    cm = confusion_matrix(y, pred)
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        ax=axes[2],
        xticklabels=["pay", "deny"],
        yticklabels=["pay", "deny"],
    )
    axes[2].set(
        title=f"Confusion Matrix @ threshold={threshold:.2f}",
        xlabel="Predicted",
        ylabel="Actual",
    )

    plt.tight_layout()

    if out_path:
        fig.savefig(
            out_path,
            dpi=120,
            bbox_inches="tight",
        )

    return fig


def plot_feature_importance(
    model,
    out_path: str | None = None,
    top_n: int = 15,
):
    """Bar chart of coefficients (LogReg) or feature importances (trees)."""
    import matplotlib.pyplot as plt

    names = list(
        model.named_steps["prep"].get_feature_names_out()
    )
    clf = model.named_steps["clf"]

    vals = (
        clf.feature_importances_
        if hasattr(clf, "feature_importances_")
        else clf.coef_[0]
    )

    imp = (
        pd.Series(vals, index=names)
        .sort_values(key=np.abs, ascending=False)
    )

    fig = plt.figure(figsize=(8, 6))
    imp.head(top_n)[::-1].plot(
        kind="barh",
        color="steelblue",
    )
    plt.title(f"Top {top_n} feature importances")
    plt.tight_layout()

    if out_path:
        fig.savefig(
            out_path,
            dpi=120,
            bbox_inches="tight",
        )

    return imp