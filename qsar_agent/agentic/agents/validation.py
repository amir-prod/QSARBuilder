"""Validation Agent — soft recommendations vs deterministic hard veto."""

from __future__ import annotations

from typing import Any

from qsar_agent.agentic.agents.base import call_agent_structured
from qsar_agent.agentic.hard_failures import build_validation_review, flags_from_runtime
from qsar_agent.agentic.prompts import VALIDATION_SYSTEM_PROMPT
from qsar_agent.agentic.provider import AgentProvider
from qsar_agent.schemas.agentic import MetricEvidence, ValidationReview


class _LLMValidationOpinion(ValidationReview):
    """Schema for optional LLM soft opinion; hard_veto ignored unless confirmed."""


def run_validation_agent(
    *,
    provider: AgentProvider | None,
    payload: dict[str, Any],
    acceptance_passed: bool,
    deterministic_flags: dict[str, bool] | None = None,
) -> tuple[ValidationReview, str]:
    flags = deterministic_flags or flags_from_runtime(
        acceptance_criteria_failed=not acceptance_passed
    )

    soft_warnings: list[str] = []
    soft_reject = False
    proposals: list[str] = []
    llm_summary = ""
    source = "deterministic_code"

    def fallback() -> ValidationReview:
        return build_validation_review(
            acceptance_passed=acceptance_passed,
            deterministic_flags=flags,
            soft_warnings=["Deterministic validation review (no LLM)."],
            evidence=[
                MetricEvidence(
                    name="acceptance_passed",
                    value=acceptance_passed,
                    source_artifact="acceptance",
                    source_field="accepted",
                )
            ],
            llm_summary="Deterministic Validation Agent review.",
        )

    if provider is not None:
        try:
            opinion, source = call_agent_structured(
                provider,
                agent_name="validation",
                system_prompt=VALIDATION_SYSTEM_PROMPT,
                payload={**payload, "deterministic_hard_failure_flags": flags},
                response_model=_LLMValidationOpinion,
                deterministic_fallback=fallback,
            )
            soft_warnings = list(opinion.warnings)
            soft_reject = bool(opinion.soft_rejection_recommended or (not opinion.approved and not opinion.hard_veto))
            proposals = list(opinion.additional_validation_proposals)
            llm_summary = opinion.summary
            # Ignore LLM hard_veto unless flags confirm
            if opinion.hard_veto and not any(flags.values()):
                soft_warnings.append(
                    "LLM requested hard veto without deterministic hard-failure confirmation; ignored."
                )
                soft_reject = True
        except Exception as exc:
            soft_warnings.append(f"Validation LLM call failed: {exc}")
            source = "deterministic_fallback"

    review = build_validation_review(
        acceptance_passed=acceptance_passed,
        deterministic_flags=flags,
        soft_warnings=soft_warnings,
        soft_rejection_recommended=soft_reject,
        additional_validation_proposals=proposals
        or [
            "Consider Y-randomization in a future release.",
            "Consider repeated CV / feature-selection stability analysis.",
        ],
        evidence=[
            MetricEvidence(
                name="acceptance_passed",
                value=acceptance_passed,
                source_artifact="acceptance",
                source_field="accepted",
            )
        ],
        llm_summary=llm_summary,
    )
    return review, source
