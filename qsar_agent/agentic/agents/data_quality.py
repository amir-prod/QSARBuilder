"""Data Quality Agent — diagnostic-only in v1."""

from __future__ import annotations

from typing import Any

from qsar_agent.agentic.agents.base import call_agent_structured
from qsar_agent.agentic.prompts import DATA_QUALITY_SYSTEM_PROMPT
from qsar_agent.agentic.provider import AgentProvider
from qsar_agent.schemas.agentic import AgentDiagnosis, ExperimentProposal, MetricEvidence


def run_data_quality_agent(
    *,
    provider: AgentProvider | None,
    experiment_id: str,
    payload: dict[str, Any],
) -> tuple[AgentDiagnosis, list[ExperimentProposal], str]:
    def fallback() -> AgentDiagnosis:
        summary = payload.get("summary") or {}
        n = summary.get("agent_dev_size") or summary.get("development_split_size") or 0
        evidence = [
            MetricEvidence(
                name="agent_dev_size",
                value=n,
                source_artifact="agent_visible_summary",
                source_field="agent_dev_size",
            )
        ]
        if int(n or 0) < 30:
            return AgentDiagnosis(
                agent_name="data_quality",
                experiment_id=experiment_id,
                failure_category="dataset_too_small",
                summary=(
                    "Development set appears small for stable QSAR modeling. "
                    "No dataset mutations are executable in v1."
                ),
                evidence=evidence,
                hypotheses=["Additional compounds may be required."],
                confidence=0.7,
                warnings=["Data Quality Agent is diagnostic-only in v1."],
                decision_source="deterministic_fallback",
                recommended_actions=["stop_no_viable_model", "request_user_input"],
            )
        return AgentDiagnosis(
            agent_name="data_quality",
            experiment_id=experiment_id,
            failure_category="data_quality_ok",
            summary="No blocking data-quality failure identified from available summaries.",
            evidence=evidence,
            hypotheses=[],
            confidence=0.4,
            warnings=["Data Quality Agent is diagnostic-only in v1."],
            decision_source="deterministic_fallback",
            recommended_actions=["request_user_input"],
        )

    diagnosis, source = call_agent_structured(
        provider,
        agent_name="data_quality",
        system_prompt=DATA_QUALITY_SYSTEM_PROMPT,
        payload=payload,
        response_model=AgentDiagnosis,
        deterministic_fallback=fallback,
    )
    diagnosis = diagnosis.model_copy(update={"decision_source": source, "agent_name": "data_quality"})

    proposals: list[ExperimentProposal] = []
    for i, action in enumerate(diagnosis.recommended_actions):
        if action not in ("stop_no_viable_model", "request_user_input"):
            continue
        proposals.append(
            ExperimentProposal(
                proposal_id=f"dq_{experiment_id}_{i}",
                parent_experiment_id=experiment_id,
                proposed_by="data_quality",
                hypothesis=diagnosis.summary,
                action=action,  # type: ignore[arg-type]
                configuration_changes={"reason": diagnosis.failure_category},
                scientific_rationale=diagnosis.summary,
                estimated_cost="low",
                requires_user_approval=False,
                duplicate_check_key="",
                experiment_kind="diagnostic_only",
                decision_source=source,  # type: ignore[arg-type]
            )
        )
    return diagnosis, proposals, source
