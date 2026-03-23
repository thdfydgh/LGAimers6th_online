"""Evaluation utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score


def evaluate_classifier(
    y_true: pd.Series,
    y_pred: np.ndarray,
    y_prob: np.ndarray | None = None,
) -> Dict[str, float]:
    confusion = confusion_matrix(y_true, y_pred)
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
    recall = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
    f1 = f1_score(y_true, y_pred, pos_label=1, zero_division=0)
    auc = roc_auc_score(y_true, y_prob) if y_prob is not None else float("nan")

    print("오차 행렬:\n", confusion)
    print(f"정확도: {accuracy:.4f}")
    print(f"정밀도: {precision:.4f}")
    print(f"재현율: {recall:.4f}")
    print(f"F1: {f1:.4f}")
    if y_prob is not None:
        print(f"ROC AUC: {auc:.4f}")

    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "auc": float(auc),
    }


def print_cv_summary(metrics: Dict[str, Iterable[float]]) -> None:
    print("\n📌 Final Cross-Validation Results")
    for name, values in metrics.items():
        arr = np.array(list(values), dtype=float)
        print(f" - {name}: {arr.mean():.4f} ± {arr.std():.4f}")


def summarize_feature_importance(
    feature_names: list[str],
    fold_importances: list[np.ndarray],
    output_csv: Path,
    output_png: Path,
    top_n: int = 50,
) -> Tuple[pd.DataFrame, float]:
    mean_importance = np.mean(np.vstack(fold_importances), axis=0)
    ranking = (
        pd.DataFrame({"feature": feature_names, "importance": mean_importance})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
    ranking.to_csv(output_csv, index=False)

    top = ranking.head(top_n).sort_values("importance", ascending=True)
    plt.figure(figsize=(10, 12))
    plt.barh(top["feature"], top["importance"])
    plt.xlabel("Mean Importance")
    plt.ylabel("Feature")
    plt.title(f"Top {top_n} Feature Importances")
    plt.tight_layout()
    plt.savefig(output_png, dpi=200)
    plt.close()

    return ranking, float(np.mean(mean_importance))

