"""Decide whether an activity column is a regression or classification endpoint."""

from __future__ import annotations

import numpy as np
import pandas as pd

from qsar_agent.schemas.agentic import TaskDetection

#: Integer columns with at most this many distinct values are treated as classes.
MAX_CLASS_LEVELS = 10


def detect_task(values: pd.Series | np.ndarray | list, forced: str | None = None) -> TaskDetection:
    """Infer the modelling task from the activity values.

    Binary or small-cardinality integer endpoints become classification;
    anything continuous becomes regression. ``forced`` overrides the inference
    while still reporting the class statistics.
    """
    series = pd.Series(values).dropna()
    if series.empty:
        raise ValueError("Activity column contains no usable values.")

    numeric = pd.to_numeric(series, errors="coerce")
    non_numeric_count = int(numeric.isna().sum())
    unique_raw = series.unique()
    n_unique = int(len(unique_raw))

    if non_numeric_count > 0:
        task = "classification"
        reason = (
            f"{non_numeric_count} non-numeric activity values found; treating the endpoint "
            f"as categorical with {n_unique} levels."
        )
        labels = series.astype(str)
    else:
        numeric = numeric.dropna()
        all_integral = bool(np.all(np.isclose(numeric.values, np.round(numeric.values))))
        if n_unique == 2:
            task = "classification"
            reason = f"Exactly 2 distinct activity values {sorted(unique_raw.tolist())}: binary classification."
        elif all_integral and n_unique <= MAX_CLASS_LEVELS:
            task = "classification"
            reason = (
                f"{n_unique} distinct integer activity values (<= {MAX_CLASS_LEVELS}): "
                "treating as discrete classes."
            )
        else:
            task = "regression"
            reason = (
                f"{n_unique} distinct continuous activity values: regression endpoint "
                f"(range {float(numeric.min()):.3g} to {float(numeric.max()):.3g})."
            )
        labels = numeric.astype(str) if task == "classification" else numeric.astype(str)

    if forced and forced != task:
        reason = f"Task forced to '{forced}' by the caller; automatic inference said '{task}' ({reason})"
        task = forced

    class_counts: dict[str, int] = {}
    positive_fraction = None
    imbalance_ratio = None
    n_classes = None
    if task == "classification":
        counts = labels.value_counts()
        class_counts = {str(k): int(v) for k, v in counts.items()}
        n_classes = len(class_counts)
        ordered = sorted(class_counts.items(), key=lambda kv: kv[0])
        if n_classes == 2:
            positive_fraction = float(ordered[-1][1]) / float(sum(class_counts.values()))
        smallest = min(class_counts.values())
        largest = max(class_counts.values())
        imbalance_ratio = float(largest) / float(smallest) if smallest else float("inf")

    return TaskDetection(
        task=task,
        reason=reason,
        n_unique_values=n_unique,
        n_classes=n_classes,
        class_counts=class_counts,
        positive_fraction=positive_fraction,
        imbalance_ratio=imbalance_ratio,
    )
