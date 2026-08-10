"""Modeling Agent — family-aware registered estimator recommendations."""

from __future__ import annotations

from typing import Any

from qsar_agent.agentic.agents.base import call_agent_structured
from qsar_agent.agentic.prompts import MODELING_SYSTEM_PROMPT
from qsar_agent.agentic.provider import AgentProvider
from qsar_agent.models.registry import list_available_estimators
from qsar_agent.schemas.agentic import AgentDiagnosis, ExperimentProposal, MetricEvidence


def run_modeling_agent(
    *,
    provider: AgentProvider | None,
    experiment_id: str,
    payload: dict[str, Any],
) -> tuple[AgentDiagnosis, list[ExperimentProposal], str]:
    def fallback() -> AgentDiagnosis:
        summary = payload.get("summary") or {}
        status = str(summary.get("overfitting_status") or "poor_performance")
        estimator = summary.get("estimator") or "RandomForestRegressor"
        evidence = [
            MetricEvidence(
                name="overfitting_status",
                value=status,
                source_artifact="agent_visible_summary",
                source_field="overfitting_status",
            ),
            MetricEvidence(
                name="mean_cv_r2",
                value=summary.get("mean_cv_r2"),
                source_artifact="agent_visible_summary",
                source_field="mean_cv_r2",
            ),
            MetricEvidence(
                name="train_cv_gap",
                value=summary.get("train_cv_gap"),
                source_artifact="agent_visible_summary",
                source_field="train_cv_gap",
            ),
        ]
        actions = []
        if status in ("overfit", "severe_overfit"):
            actions = ["refine_hyperparameters", "try_registered_estimator", "compare_registered_estimators"]
            hyp = [
                "Tree ensemble capacity may be too high; try regularization or linear/kernel models."
            ]
        elif status == "underfit":
            actions = ["try_registered_estimator", "compare_registered_estimators", "refine_hyperparameters"]
            hyp = ["Current family may lack capacity; screen boosting or kernel models."]
        elif status == "unstable":
            actions = ["refine_hyperparameters", "try_registered_estimator"]
            hyp = ["High CV variance; prefer more regularized or simpler families."]
        else:
            actions = ["compare_registered_estimators", "try_registered_estimator"]
            hyp = ["Screen alternate registered families with shared folds/features."]

        return AgentDiagnosis(
            agent_name="modeling",
            experiment_id=experiment_id,
            failure_category=status,
            summary=f"Modeling diagnosis for {estimator}: status={status}.",
            evidence=evidence,
            hypotheses=hyp,
            confidence=0.6,
            decision_source="deterministic_fallback",
            recommended_actions=actions,  # type: ignore[arg-type]
        )

    diagnosis, source = call_agent_structured(
        provider,
        agent_name="modeling",
        system_prompt=MODELING_SYSTEM_PROMPT,
        payload=payload,
        response_model=AgentDiagnosis,
        deterministic_fallback=fallback,
    )
    diagnosis = diagnosis.model_copy(update={"decision_source": source, "agent_name": "modeling"})

    summary = payload.get("summary") or {}
    current = summary.get("estimator") or "RandomForestRegressor"
    status = str(summary.get("overfitting_status") or "default")
    available = [e for e in list_available_estimators() if e != current]
    # Prefer diverse families for screening
    screen = available[:5]
    proposals: list[ExperimentProposal] = []
    for i, action in enumerate(diagnosis.recommended_actions):
        if action == "refine_hyperparameters":
            proposals.append(
                ExperimentProposal(
                    proposal_id=f"md_{experiment_id}_hpo_{i}",
                    parent_experiment_id=experiment_id,
                    proposed_by="modeling",
                    hypothesis=diagnosis.summary,
                    action="refine_hyperparameters",
                    configuration_changes={
                        "estimator": current,
                        "status_hint": status if status in ("overfit", "underfit", "unstable") else "default",
                        "max_candidates": 40,
                    },
                    scientific_rationale="Bounded HPO refinement on fixed features/folds.",
                    estimated_cost="medium",
                    experiment_kind="hyperparameter_refinement",
                    multi_component=False,
                    component_list=["hyperparameter_optimization"],
                    decision_source=source,  # type: ignore[arg-type]
                )
            )
        elif action == "compare_registered_estimators" and screen:
            proposals.append(
                ExperimentProposal(
                    proposal_id=f"md_{experiment_id}_cmp_{i}",
                    parent_experiment_id=experiment_id,
                    proposed_by="modeling",
                    hypothesis=diagnosis.summary,
                    action="compare_registered_estimators",
                    configuration_changes={
                        "estimators": screen[:5],
                        "optimize_top_k": 1,
                    },
                    scientific_rationale=(
                        "Controlled comparison: same data, features, preprocessing, and CV folds; "
                        "change only the estimator."
                    ),
                    estimated_cost="medium",
                    experiment_kind="controlled_estimator_comparison",
                    multi_component=False,
                    component_list=["estimator"],
                    decision_source=source,  # type: ignore[arg-type]
                )
            )
        elif action == "try_registered_estimator" and available:
            # Pick a contrasting family
            pick = _pick_contrast_estimator(current, status, available)
            proposals.append(
                ExperimentProposal(
                    proposal_id=f"md_{experiment_id}_try_{i}",
                    parent_experiment_id=experiment_id,
                    proposed_by="modeling",
                    hypothesis=diagnosis.summary,
                    action="try_registered_estimator",
                    configuration_changes={
                        "estimator": pick,
                        "mode": "controlled",
                        "run_hpo": False,
                    },
                    scientific_rationale=(
                        f"Controlled try of {pick}: same features/folds; estimator only."
                    ),
                    estimated_cost="low",
                    experiment_kind="controlled_estimator_comparison",
                    multi_component=False,
                    component_list=["estimator"],
                    decision_source=source,  # type: ignore[arg-type]
                )
            )
    return diagnosis, proposals, source


def _pick_contrast_estimator(current: str, status: str, available: list[str]) -> str:
    preferred_overfit = ["Ridge", "ElasticNet", "PLSRegression", "SVR", "KNeighborsRegressor"]
    preferred_underfit = [
        "HistGradientBoostingRegressor",
        "GradientBoostingRegressor",
        "ExtraTreesRegressor",
        "SVR",
    ]
    preferred = preferred_overfit if status in ("overfit", "severe_overfit", "unstable") else preferred_underfit
    for name in preferred:
        if name in available:
            return name
    return available[0]
