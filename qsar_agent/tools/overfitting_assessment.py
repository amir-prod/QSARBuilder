"""Deterministic overfitting assessment from training CV metrics."""

from __future__ import annotations

from qsar_agent.schemas.hyperparameter_optimization import (
    CVSummary,
    OverfittingAssessment,
    OverfittingStatus,
    OverfittingThresholds,
)


def assess_overfitting(
    summary: CVSummary | dict,
    thresholds: OverfittingThresholds | None = None,
) -> OverfittingAssessment:
    """Classify model quality using training CV metrics only."""
    th = thresholds or OverfittingThresholds()
    if isinstance(summary, dict):
        summary = CVSummary(**summary)

    gap = summary.train_cv_r2_gap
    mean_train = summary.mean_train_r2
    mean_cv = summary.mean_cv_r2
    cv_std = summary.std_cv_r2

    warnings: list[str] = []
    is_severe_overfit = gap > th.severe_overfit_gap_threshold
    if is_severe_overfit:
        warnings.append(
            f"Severe overfitting: train-CV R² gap ({gap:.3f}) exceeds "
            f"{th.severe_overfit_gap_threshold:.2f}."
        )

    is_unstable = cv_std > th.cv_std_threshold
    is_overfit = gap > th.overfit_gap_threshold and mean_train > th.minimum_train_r2
    is_underfit = mean_train < th.minimum_train_r2 and mean_cv < th.minimum_cv_r2
    poor_performance = mean_cv < th.minimum_cv_r2

    if is_unstable:
        status: OverfittingStatus = "unstable"
        explanation = (
            f"CV R² variability is high (std={cv_std:.3f} > {th.cv_std_threshold:.2f}). "
            "The model may be sensitive to training fold composition."
        )
    elif is_underfit:
        status = "underfit"
        explanation = (
            f"Both training R² ({mean_train:.3f}) and CV R² ({mean_cv:.3f}) are low. "
            "The model lacks capacity or informative descriptors."
        )
    elif is_overfit:
        status = "overfit"
        explanation = (
            f"Training R² ({mean_train:.3f}) is much higher than CV R² ({mean_cv:.3f}); "
            f"gap={gap:.3f} exceeds {th.overfit_gap_threshold:.2f}."
        )
    elif poor_performance:
        status = "poor_performance"
        explanation = (
            f"CV R² ({mean_cv:.3f}) is below the minimum acceptable threshold "
            f"({th.minimum_cv_r2:.2f})."
        )
    else:
        status = "good"
        explanation = (
            f"CV R² ({mean_cv:.3f}) is acceptable, train-CV gap ({gap:.3f}) is within "
            f"limits, and CV variability (std={cv_std:.3f}) is acceptable."
        )

    if is_unstable:
        warnings.append(f"High CV R² standard deviation: {cv_std:.3f}.")
    if is_overfit and not is_severe_overfit:
        warnings.append(f"Train-CV R² gap ({gap:.3f}) suggests overfitting.")
    if poor_performance and status != "poor_performance":
        warnings.append(f"CV R² ({mean_cv:.3f}) is below minimum threshold.")

    is_acceptable = status == "good"

    return OverfittingAssessment(
        status=status,
        is_acceptable=is_acceptable,
        is_overfit=is_overfit,
        is_underfit=is_underfit,
        is_unstable=is_unstable,
        is_severe_overfit=is_severe_overfit,
        mean_train_r2=mean_train,
        mean_cv_r2=mean_cv,
        train_cv_r2_gap=gap,
        cv_r2_std=cv_std,
        warnings=warnings,
        explanation=explanation,
    )
