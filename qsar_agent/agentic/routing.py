"""Explicit specialist-routing rules from observed failure signals."""

from __future__ import annotations

from typing import Any

SPECIALIST_DATA_QUALITY = "data_quality"
SPECIALIST_DESCRIPTOR_FEATURE = "descriptor_feature"
SPECIALIST_MODELING = "modeling"
SPECIALIST_VALIDATION = "validation"

# Ordered priority: first matches win, then capped by max_specialist_calls.
_ROUTING_RULES: list[tuple[str, str]] = [
    ("external_leakage_risk", SPECIALIST_VALIDATION),
    ("methodology_dispute", SPECIALIST_VALIDATION),
    ("acceptance_dispute", SPECIALIST_VALIDATION),
    ("dataset_too_small", SPECIALIST_DATA_QUALITY),
    ("duplicate_conflicts", SPECIALIST_DATA_QUALITY),
    ("activity_distribution", SPECIALIST_DATA_QUALITY),
    ("invalid_structures", SPECIALIST_DATA_QUALITY),
    ("samples_per_feature_low", SPECIALIST_DESCRIPTOR_FEATURE),
    ("feature_count_issue", SPECIALIST_DESCRIPTOR_FEATURE),
    ("sfs_ga_unproductive", SPECIALIST_DESCRIPTOR_FEATURE),
    ("descriptor_removal_heavy", SPECIALIST_DESCRIPTOR_FEATURE),
    ("severe_overfit", SPECIALIST_MODELING),
    ("overfit", SPECIALIST_MODELING),
    ("underfit", SPECIALIST_MODELING),
    ("unstable_cv", SPECIALIST_MODELING),
    ("poor_performance", SPECIALIST_MODELING),
    ("unproductive_hpo", SPECIALIST_MODELING),
    ("estimator_mismatch", SPECIALIST_MODELING),
]


def infer_failure_signals(internal_metrics: dict[str, Any], summary: dict[str, Any] | None = None) -> list[str]:
    """Derive routing signals from structured metrics (deterministic)."""
    signals: list[str] = []
    summary = summary or {}
    status = str(internal_metrics.get("overfitting_status") or summary.get("overfitting_status") or "")
    gap = internal_metrics.get("train_cv_gap", summary.get("train_cv_gap"))
    cv_std = internal_metrics.get("cv_r2_std", summary.get("cv_r2_std"))
    mean_cv = internal_metrics.get("mean_cv_r2", summary.get("mean_cv_r2"))
    ratio = internal_metrics.get("samples_per_feature_ratio", summary.get("samples_per_feature_ratio"))
    n_dev = internal_metrics.get("agent_dev_size", summary.get("agent_dev_size"))
    hpo_rounds = internal_metrics.get("hpo_rounds", summary.get("hpo_rounds"))

    if status == "unstable" or (cv_std is not None and float(cv_std) > 0.15):
        signals.append("unstable_cv")
    if status == "underfit":
        signals.append("underfit")
    if status in ("overfit",) or (gap is not None and float(gap) > 0.15):
        signals.append("overfit")
    if status == "severe_overfit" or (gap is not None and float(gap) > 0.25):
        signals.append("severe_overfit")
    if status == "poor_performance" or (mean_cv is not None and float(mean_cv) < 0.50):
        signals.append("poor_performance")
    if ratio is not None and float(ratio) < 5.0:
        signals.append("samples_per_feature_low")
        signals.append("feature_count_issue")
    if n_dev is not None and int(n_dev) < 30:
        signals.append("dataset_too_small")
    if hpo_rounds is not None and int(hpo_rounds) >= 3 and mean_cv is not None and float(mean_cv) < 0.55:
        signals.append("unproductive_hpo")
    if summary.get("descriptor_removal_fraction", 0) and float(summary.get("descriptor_removal_fraction", 0)) > 0.5:
        signals.append("descriptor_removal_heavy")

    # Always include modeling if nothing else matched and model is unacceptable
    if not signals:
        signals.append("poor_performance")
    return signals


def select_specialists(
    failure_signals: list[str],
    *,
    max_specialists: int = 2,
    force_validation: bool = False,
) -> list[str]:
    """Select up to max_specialists using explicit routing rules."""
    selected: list[str] = []
    signal_set = set(failure_signals)
    for signal, specialist in _ROUTING_RULES:
        if signal in signal_set and specialist not in selected:
            selected.append(specialist)
        if len(selected) >= max_specialists:
            break
    if force_validation and SPECIALIST_VALIDATION not in selected:
        if len(selected) >= max_specialists:
            selected[-1] = SPECIALIST_VALIDATION
        else:
            selected.append(SPECIALIST_VALIDATION)
    return selected[:max_specialists]
