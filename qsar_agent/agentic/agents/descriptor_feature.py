"""Descriptor and Feature Agent."""

from __future__ import annotations

from typing import Any

from qsar_agent.agentic.agents.base import call_agent_structured
from qsar_agent.agentic.prompts import DESCRIPTOR_FEATURE_SYSTEM_PROMPT
from qsar_agent.agentic.provider import AgentProvider
from qsar_agent.schemas.agentic import AgentDiagnosis, ExperimentProposal, MetricEvidence


def run_descriptor_feature_agent(
    *,
    provider: AgentProvider | None,
    experiment_id: str,
    payload: dict[str, Any],
) -> tuple[AgentDiagnosis, list[ExperimentProposal], str]:
    def fallback() -> AgentDiagnosis:
        summary = payload.get("summary") or {}
        ratio = summary.get("samples_per_feature_ratio")
        fc = summary.get("feature_count") or 0
        evidence = [
            MetricEvidence(
                name="samples_per_feature_ratio",
                value=ratio,
                source_artifact="agent_visible_summary",
                source_field="samples_per_feature_ratio",
            ),
            MetricEvidence(
                name="feature_count",
                value=fc,
                source_artifact="agent_visible_summary",
                source_field="feature_count",
            ),
        ]
        if ratio is not None and float(ratio) < 5 and int(fc) > 2:
            return AgentDiagnosis(
                agent_name="descriptor_feature",
                experiment_id=experiment_id,
                failure_category="samples_per_feature_low",
                summary="Samples-per-feature ratio is low; reducing feature count may help.",
                evidence=evidence,
                hypotheses=["Model may overfit due to excess descriptors."],
                confidence=0.65,
                decision_source="deterministic_fallback",
                recommended_actions=["reduce_feature_count"],
            )
        return AgentDiagnosis(
            agent_name="descriptor_feature",
            experiment_id=experiment_id,
            failure_category="feature_selection",
            summary="Consider SFS-fixed GA expansion if features underfit.",
            evidence=evidence,
            hypotheses=["Additional complementary features may improve CV."],
            confidence=0.4,
            decision_source="deterministic_fallback",
            recommended_actions=["run_sfs_fixed_ga_expansion", "expand_feature_count"],
        )

    diagnosis, source = call_agent_structured(
        provider,
        agent_name="descriptor_feature",
        system_prompt=DESCRIPTOR_FEATURE_SYSTEM_PROMPT,
        payload=payload,
        response_model=AgentDiagnosis,
        deterministic_fallback=fallback,
    )
    diagnosis = diagnosis.model_copy(
        update={"decision_source": source, "agent_name": "descriptor_feature"}
    )

    summary = payload.get("summary") or {}
    fc = int(summary.get("feature_count") or 5)
    proposals: list[ExperimentProposal] = []
    for i, action in enumerate(diagnosis.recommended_actions):
        if action == "reduce_feature_count":
            changes = {"feature_count": max(1, fc - 2)}
            kind = "feature_count_change"
            multi = True
        elif action == "expand_feature_count":
            changes = {"feature_count": fc + 2}
            kind = "feature_count_change"
            multi = True
        elif action == "run_sfs_fixed_ga_expansion":
            changes = {"extra_features": 2}
            kind = "sfs_fixed_ga_expansion"
            multi = True
        else:
            continue
        proposals.append(
            ExperimentProposal(
                proposal_id=f"df_{experiment_id}_{i}",
                parent_experiment_id=experiment_id,
                proposed_by="descriptor_feature",
                hypothesis=diagnosis.summary,
                action=action,  # type: ignore[arg-type]
                configuration_changes=changes,
                scientific_rationale=diagnosis.summary,
                estimated_cost="medium",
                experiment_kind=kind,  # type: ignore[arg-type]
                multi_component=multi,
                component_list=["feature_selection", "genetic_algorithm", "hpo"] if multi else [],
                decision_source=source,  # type: ignore[arg-type]
            )
        )
    return diagnosis, proposals, source
