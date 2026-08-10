"""Deterministic internal model-acceptance evaluation."""

from __future__ import annotations

from typing import Any

from qsar_agent.schemas.agentic import (
    AcceptanceResult,
    AgenticAcceptanceCriteria,
    MetricEvidence,
)
from qsar_agent.schemas.hyperparameter_optimization import OverfittingThresholds
from qsar_agent.tools.overfitting_assessment import assess_overfitting


def evaluate_acceptance(
    internal_metrics: dict[str, Any],
    criteria: AgenticAcceptanceCriteria | None = None,
    *,
    validation_approved: bool | None = None,
    overfit_thresholds: OverfittingThresholds | None = None,
) -> AcceptanceResult:
    """Accept only when deterministic thresholds pass. LLM cannot override."""
    criteria = criteria or AgenticAcceptanceCriteria()
    evidence: list[MetricEvidence] = []
    failed: list[str] = []

    mean_cv = _f(internal_metrics.get("mean_cv_r2"))
    mean_train = _f(internal_metrics.get("mean_train_r2"))
    gap = _f(internal_metrics.get("train_cv_gap"))
    cv_std = _f(internal_metrics.get("cv_r2_std"))
    agent_val_r2 = _f(internal_metrics.get("agent_val_r2"))

    if mean_cv is not None:
        evidence.append(
            MetricEvidence(
                name="mean_cv_r2",
                value=mean_cv,
                source_artifact="internal_metrics",
                source_field="mean_cv_r2",
            )
        )
        if mean_cv < criteria.minimum_mean_cv_r2:
            failed.append(
                f"mean_cv_r2 {mean_cv:.4f} < minimum_mean_cv_r2 {criteria.minimum_mean_cv_r2:.4f}"
            )
    else:
        failed.append("mean_cv_r2 missing")

    if gap is not None:
        evidence.append(
            MetricEvidence(
                name="train_cv_gap",
                value=gap,
                source_artifact="internal_metrics",
                source_field="train_cv_gap",
            )
        )
        if gap > criteria.maximum_train_cv_gap:
            failed.append(
                f"train_cv_gap {gap:.4f} > maximum_train_cv_gap {criteria.maximum_train_cv_gap:.4f}"
            )
    else:
        failed.append("train_cv_gap missing")

    if cv_std is not None:
        evidence.append(
            MetricEvidence(
                name="cv_r2_std",
                value=cv_std,
                source_artifact="internal_metrics",
                source_field="cv_r2_std",
            )
        )
        if cv_std > criteria.maximum_cv_r2_std:
            failed.append(
                f"cv_r2_std {cv_std:.4f} > maximum_cv_r2_std {criteria.maximum_cv_r2_std:.4f}"
            )
    else:
        failed.append("cv_r2_std missing")

    if criteria.minimum_mean_train_r2 is not None:
        if mean_train is None or mean_train < criteria.minimum_mean_train_r2:
            failed.append("minimum_mean_train_r2 not met")

    overfit_status = None
    if criteria.require_non_overfit_status and mean_cv is not None and mean_train is not None and gap is not None and cv_std is not None:
        assessment = assess_overfitting(
            {
                "mean_train_r2": mean_train,
                "mean_cv_r2": mean_cv,
                "std_cv_r2": cv_std,
                "mean_train_rmse": float(internal_metrics.get("mean_train_rmse", 0.0)),
                "mean_cv_rmse": float(internal_metrics.get("mean_cv_rmse", 0.0)),
                "mean_train_mae": float(internal_metrics.get("mean_train_mae", 0.0)),
                "mean_cv_mae": float(internal_metrics.get("mean_cv_mae", 0.0)),
                "train_cv_r2_gap": gap,
                "n_folds": int(internal_metrics.get("n_folds", 5)),
            },
            thresholds=overfit_thresholds,
        )
        overfit_status = assessment.status
        evidence.append(
            MetricEvidence(
                name="overfitting_status",
                value=assessment.status,
                source_artifact="overfitting_assessment",
                source_field="status",
            )
        )
        if not assessment.is_acceptable:
            failed.append(f"overfitting status is '{assessment.status}', require status 'good'")

    if criteria.minimum_agent_val_r2 is not None and agent_val_r2 is not None:
        evidence.append(
            MetricEvidence(
                name="agent_val_r2",
                value=agent_val_r2,
                source_artifact="internal_metrics",
                source_field="agent_val_r2",
            )
        )
        if agent_val_r2 < criteria.minimum_agent_val_r2:
            failed.append(
                f"agent_val_r2 {agent_val_r2:.4f} < minimum_agent_val_r2 {criteria.minimum_agent_val_r2:.4f}"
            )

    if (
        criteria.maximum_cv_agent_val_gap is not None
        and agent_val_r2 is not None
        and mean_cv is not None
    ):
        cv_av_gap = abs(mean_cv - agent_val_r2)
        evidence.append(
            MetricEvidence(
                name="cv_agent_val_gap",
                value=cv_av_gap,
                source_artifact="internal_metrics",
                source_field="cv_agent_val_gap",
            )
        )
        if cv_av_gap > criteria.maximum_cv_agent_val_gap:
            failed.append(
                f"cv_agent_val_gap {cv_av_gap:.4f} > maximum_cv_agent_val_gap "
                f"{criteria.maximum_cv_agent_val_gap:.4f}"
            )

    if criteria.require_validation_agent_approval:
        if validation_approved is False:
            failed.append("validation agent approval required but not granted")
        elif validation_approved is None:
            # Caller has not yet run validation; do not fail solely on missing review
            pass

    accepted = len(failed) == 0
    if accepted:
        explanation = "All configured internal acceptance criteria were satisfied."
    else:
        explanation = "Acceptance failed: " + "; ".join(failed)

    return AcceptanceResult(
        accepted=accepted,
        evidence=evidence,
        failed_criteria=failed,
        explanation=explanation,
        overfit_status=overfit_status,
    )


def _f(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
