"""Unified regression and classification metrics.

Every number the agents report comes from here, so the LLM never has an
opportunity to invent a metric.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)

#: The metric the loop optimises and reports first, per task.
PRIMARY_METRIC = {"regression": "r2", "classification": "roc_auc"}

#: sklearn scorer names used for cross-validation, per task.
CV_SCORER = {"regression": "r2", "classification": "roc_auc"}


def primary_metric_name(task: str) -> str:
    try:
        return PRIMARY_METRIC[task]
    except KeyError:
        raise ValueError(f"Unknown task: {task!r}") from None


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "rmse": rmse,
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "n_samples": float(len(y_true)),
    }


def classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: np.ndarray | None = None,
) -> dict[str, float]:
    """Imbalance-aware classification metrics.

    ``roc_auc`` is computed from ``y_score`` when available; for multiclass
    problems it uses the one-vs-rest macro average.
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    classes = np.unique(y_true)
    binary = len(classes) <= 2
    average = "binary" if binary else "macro"

    metrics: dict[str, float] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, average=average, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, average=average, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average=average, zero_division=0)),
        "n_samples": float(len(y_true)),
    }

    if binary and len(classes) == 2:
        positive = classes[-1]
        true_positive = float(np.sum((y_true == positive) & (y_pred == positive)))
        true_negative = float(np.sum((y_true != positive) & (y_pred != positive)))
        n_positive = float(np.sum(y_true == positive))
        n_negative = float(np.sum(y_true != positive))
        metrics["sensitivity"] = true_positive / n_positive if n_positive else float("nan")
        metrics["specificity"] = true_negative / n_negative if n_negative else float("nan")

    metrics["roc_auc"] = _roc_auc(y_true, y_score, classes)
    return metrics


def _roc_auc(y_true: np.ndarray, y_score: np.ndarray | None, classes: np.ndarray) -> float:
    if y_score is None or len(classes) < 2:
        return float("nan")
    score = np.asarray(y_score, dtype=float)
    try:
        if len(classes) == 2:
            if score.ndim == 2:
                score = score[:, -1]
            return float(roc_auc_score(y_true, score))
        return float(roc_auc_score(y_true, score, multi_class="ovr", average="macro"))
    except ValueError:
        # A fold or split containing a single class cannot yield an AUC.
        return float("nan")


def compute_metrics(
    task: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: np.ndarray | None = None,
) -> dict[str, float]:
    if task == "regression":
        return regression_metrics(y_true, y_pred)
    if task == "classification":
        return classification_metrics(y_true, y_pred, y_score)
    raise ValueError(f"Unknown task: {task!r}")
