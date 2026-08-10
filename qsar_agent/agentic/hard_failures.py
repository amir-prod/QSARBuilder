"""Deterministic hard-failure registry for Validation Agent vetoes."""

from __future__ import annotations

from typing import Any

from qsar_agent.schemas.agentic import HardFailureCondition, MetricEvidence, ValidationReview


REGISTERED_HARD_FAILURES: frozenset[HardFailureCondition] = frozenset(
    {
        "external_test_access_attempted",
        "external_eval_before_lock",
        "agentic_after_external_access",
        "preprocessing_fit_on_non_train",
        "feature_selection_used_protected_targets",
        "hpo_used_protected_targets",
        "acceptance_criteria_failed",
        "invalid_action",
        "incompatible_estimator",
        "duplicate_experiment",
        "budget_exhausted",
        "missing_lock_prerequisites",
    }
)


def confirm_hard_failures(
    claimed: list[HardFailureCondition] | list[str],
    *,
    evidence: list[MetricEvidence] | None = None,
    deterministic_flags: dict[str, bool] | None = None,
) -> list[HardFailureCondition]:
    """Return only hard failures confirmed by deterministic flags.

    Qualitative LLM claims without a matching True flag are ignored.
    """
    flags = deterministic_flags or {}
    confirmed: list[HardFailureCondition] = []
    for item in claimed:
        key = str(item)
        if key not in REGISTERED_HARD_FAILURES:
            continue
        if flags.get(key, False):
            confirmed.append(key)  # type: ignore[arg-type]
    return confirmed


def build_validation_review(
    *,
    acceptance_passed: bool,
    deterministic_flags: dict[str, bool],
    soft_warnings: list[str] | None = None,
    soft_rejection_recommended: bool = False,
    additional_validation_proposals: list[str] | None = None,
    evidence: list[MetricEvidence] | None = None,
    llm_summary: str = "",
) -> ValidationReview:
    """Combine deterministic hard checks with optional soft LLM concerns."""
    claimed = [k for k, v in deterministic_flags.items() if v]
    confirmed = confirm_hard_failures(claimed, deterministic_flags=deterministic_flags)
    hard_veto = bool(confirmed)
    approved = acceptance_passed and not hard_veto
    warnings = list(soft_warnings or [])
    if soft_rejection_recommended and not hard_veto:
        warnings.append(
            "Validation Agent recommended rejection, but no deterministic hard-failure "
            "condition was confirmed; treated as warning only."
        )
    summary_parts = []
    if hard_veto:
        summary_parts.append(f"Hard veto: {', '.join(confirmed)}.")
    if llm_summary:
        summary_parts.append(llm_summary)
    if not summary_parts:
        summary_parts.append(
            "Accepted." if approved else "Rejected by deterministic acceptance criteria."
        )
    return ValidationReview(
        approved=approved,
        hard_veto=hard_veto,
        hard_failure_conditions=confirmed,
        soft_rejection_recommended=soft_rejection_recommended and not hard_veto,
        warnings=warnings,
        additional_validation_proposals=list(additional_validation_proposals or []),
        evidence=list(evidence or []),
        summary=" ".join(summary_parts),
        decision_source="deterministic_code",
    )


def flags_from_runtime(
    *,
    external_test_access_attempted: bool = False,
    external_eval_before_lock: bool = False,
    agentic_after_external_access: bool = False,
    preprocessing_fit_on_non_train: bool = False,
    feature_selection_used_protected_targets: bool = False,
    hpo_used_protected_targets: bool = False,
    acceptance_criteria_failed: bool = False,
    invalid_action: bool = False,
    incompatible_estimator: bool = False,
    duplicate_experiment: bool = False,
    budget_exhausted: bool = False,
    missing_lock_prerequisites: bool = False,
) -> dict[str, bool]:
    return {
        "external_test_access_attempted": external_test_access_attempted,
        "external_eval_before_lock": external_eval_before_lock,
        "agentic_after_external_access": agentic_after_external_access,
        "preprocessing_fit_on_non_train": preprocessing_fit_on_non_train,
        "feature_selection_used_protected_targets": feature_selection_used_protected_targets,
        "hpo_used_protected_targets": hpo_used_protected_targets,
        "acceptance_criteria_failed": acceptance_criteria_failed,
        "invalid_action": invalid_action,
        "incompatible_estimator": incompatible_estimator,
        "duplicate_experiment": duplicate_experiment,
        "budget_exhausted": budget_exhausted,
        "missing_lock_prerequisites": missing_lock_prerequisites,
    }
