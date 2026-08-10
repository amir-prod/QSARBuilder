"""Supervisor Agent — sole authority to select the next experiment."""

from __future__ import annotations

from typing import Any

from qsar_agent.agentic.agents.base import call_agent_structured
from qsar_agent.agentic.duplicate import ensure_proposal_key, find_duplicate, near_duplicate_warnings
from qsar_agent.agentic.prompts import SUPERVISOR_SYSTEM_PROMPT
from qsar_agent.agentic.provider import AgentProvider
from qsar_agent.agentic.routing import infer_failure_signals, select_specialists
from qsar_agent.schemas.agentic import (
    ExperimentProposal,
    ExperimentRecord,
    MetricEvidence,
    SupervisorDecision,
    normalize_action,
)


def run_supervisor(
    *,
    provider: AgentProvider | None,
    cycle_index: int,
    payload: dict[str, Any],
    proposals: list[ExperimentProposal],
    ledger: list[ExperimentRecord],
    acceptance_passed: bool,
    budget_exhausted: bool,
    specialists_consulted: list[str],
) -> tuple[SupervisorDecision, ExperimentProposal | None, str]:
    # Filter duplicates / invalid
    usable: list[ExperimentProposal] = []
    rejected: list[dict[str, Any]] = []
    for prop in proposals:
        prop = ensure_proposal_key(prop)
        try:
            action = normalize_action(prop.action)
        except Exception:
            rejected.append({"proposal_id": prop.proposal_id, "reason": "invalid_action"})
            continue
        dup = find_duplicate(ledger, action, prop.configuration_changes)
        if dup is not None:
            rejected.append(
                {
                    "proposal_id": prop.proposal_id,
                    "reason": "duplicate_experiment",
                    "prior_experiment_id": dup.experiment_id,
                }
            )
            continue
        for w in near_duplicate_warnings(ledger, action, prop.configuration_changes):
            rejected.append({"proposal_id": prop.proposal_id, "reason": "near_duplicate_warning", "detail": w})
        usable.append(prop)

    def fallback() -> SupervisorDecision:
        if acceptance_passed:
            return SupervisorDecision(
                cycle_index=cycle_index,
                selected_proposal_id=None,
                action="accept_model",
                rationale="Deterministic acceptance criteria satisfied.",
                evidence_considered=_evidence_from_payload(payload),
                rejected_proposals=rejected,
                stopping_reason="accepted",
                specialists_consulted=specialists_consulted,
                decision_source="deterministic_fallback",
            )
        if budget_exhausted:
            return SupervisorDecision(
                cycle_index=cycle_index,
                selected_proposal_id=None,
                action="stop_budget_exhausted",
                rationale="Experiment or cycle budget exhausted.",
                evidence_considered=_evidence_from_payload(payload),
                rejected_proposals=rejected,
                stopping_reason="budget_exhausted",
                specialists_consulted=specialists_consulted,
                decision_source="deterministic_fallback",
            )
        if not usable:
            return SupervisorDecision(
                cycle_index=cycle_index,
                selected_proposal_id=None,
                action="stop_no_viable_model",
                rationale="No scientifically defensible non-duplicate action remains.",
                evidence_considered=_evidence_from_payload(payload),
                rejected_proposals=rejected,
                stopping_reason="no_viable_action",
                specialists_consulted=specialists_consulted,
                decision_source="deterministic_fallback",
            )
        # Prefer controlled modeling actions over multi-component
        ranked = sorted(
            usable,
            key=lambda p: (
                0 if p.action == "compare_registered_estimators" else 1,
                0 if p.action == "try_registered_estimator" else 1,
                0 if p.action == "refine_hyperparameters" else 1,
                0 if not p.multi_component else 1,
                p.estimated_cost != "low",
            ),
        )
        chosen = ranked[0]
        return SupervisorDecision(
            cycle_index=cycle_index,
            selected_proposal_id=chosen.proposal_id,
            action=normalize_action(chosen.action),
            rationale=f"Selected {chosen.action}: {chosen.scientific_rationale or chosen.hypothesis}",
            evidence_considered=_evidence_from_payload(payload),
            rejected_proposals=rejected
            + [
                {"proposal_id": p.proposal_id, "reason": "not_selected"}
                for p in ranked[1:]
            ],
            stopping_reason=None,
            specialists_consulted=specialists_consulted,
            decision_source="deterministic_fallback",
        )

    decision, source = call_agent_structured(
        provider,
        agent_name="supervisor",
        system_prompt=SUPERVISOR_SYSTEM_PROMPT,
        payload={
            **payload,
            "proposals": [p.model_dump() for p in usable],
            "rejected_proposals": rejected,
            "acceptance_passed": acceptance_passed,
            "budget_exhausted": budget_exhausted,
            "specialists_consulted": specialists_consulted,
        },
        response_model=SupervisorDecision,
        deterministic_fallback=fallback,
    )
    decision = decision.model_copy(
        update={
            "decision_source": source,
            "cycle_index": cycle_index,
            "specialists_consulted": specialists_consulted,
        }
    )

    selected: ExperimentProposal | None = None
    if decision.selected_proposal_id:
        for p in usable:
            if p.proposal_id == decision.selected_proposal_id:
                selected = p
                break
    if selected is None and decision.action not in (
        "accept_model",
        "stop_budget_exhausted",
        "stop_no_viable_model",
        "request_user_approval",
        "request_user_input",
        "recommend_unregistered_estimator",
        "request_model_dependency_approval",
    ):
        # Fall back to deterministic choice if LLM picked unknown id
        decision = fallback()
        source = "deterministic_fallback"
        if decision.selected_proposal_id:
            selected = next(
                (p for p in usable if p.proposal_id == decision.selected_proposal_id),
                None,
            )

    return decision, selected, source


def plan_specialists_for_cycle(
    internal_metrics: dict[str, Any],
    summary: dict[str, Any],
    *,
    max_specialists: int,
    force_validation: bool = False,
) -> list[str]:
    signals = infer_failure_signals(internal_metrics, summary)
    return select_specialists(
        signals, max_specialists=max_specialists, force_validation=force_validation
    )


def _evidence_from_payload(payload: dict[str, Any]) -> list[MetricEvidence]:
    summary = payload.get("summary") or {}
    out: list[MetricEvidence] = []
    for name in ("mean_cv_r2", "train_cv_gap", "cv_r2_std", "overfitting_status"):
        if name in summary:
            out.append(
                MetricEvidence(
                    name=name,
                    value=summary.get(name),
                    source_artifact="agent_visible_summary",
                    source_field=name,
                )
            )
    return out
