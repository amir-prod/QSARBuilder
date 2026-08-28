"""LLM decision schema validation for the modeling-improvement agent."""

from __future__ import annotations

import json
from typing import Any, Callable

from pydantic import ValidationError

from qsar_agent.agentic.ids import make_experiment_id
from qsar_agent.agentic.sealing import SealedTestAccessError, assert_no_test_paths
from qsar_agent.config import WorkflowConfig, get_openai_api_key, get_openai_model
from qsar_agent.models.registry import SUPPORTED_ESTIMATORS, sanitize_param_grid
from qsar_agent.schemas.agentic import (
    APPROVED_TOOL_NAMES,
    AgentDecision,
    ModelingAgentState,
)


class DecisionRejected(ValueError):
    """Raised when an agent decision fails structured validation."""


def _parse_agent_json(content: str) -> dict[str, Any]:
    text = content.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        lines = [ln for ln in lines if not ln.strip().startswith("```")]
        text = "\n".join(lines)
    return json.loads(text)


def fallback_decision(state: ModelingAgentState) -> AgentDecision:
    """Deterministic action when OpenAI is unavailable (tests / --no-openai)."""
    failed = [item.get("name") for item in state.failed_requirements if isinstance(item, dict)]
    used = set(state.completed_experiment_ids)
    if "maximum_train_cv_gap" in failed or "minimum_cv_r2" in failed:
        action = {
            "tool_name": "run_feature_selection_search",
            "arguments": {"method": "genetic_algorithm", "n_features": 5},
        }
        diagnosis = "moderate_overfitting" if "maximum_train_cv_gap" in failed else "underfitting"
    elif "maximum_cv_r2_std" in failed:
        action = {"tool_name": "run_robustness_analysis", "arguments": {"repeats": 3}}
        diagnosis = "unstable_cv"
    else:
        action = {"tool_name": "run_model_search", "arguments": {"estimator": "RandomForestRegressor"}}
        diagnosis = "poor_performance"
    exp_id = make_experiment_id(
        tool_name=action["tool_name"],
        arguments=action["arguments"],
        dataset_hash=state.dataset_hash,
        development_split_hash=state.development_split_hash,
        parent_experiment_id=(state.current_best_candidate or {}).get("experiment_id"),
    )
    if exp_id in used:
        action = {
            "tool_name": "request_new_capability",
            "arguments": {
                "capability": "no remaining allowlisted experiment",
                "scientific_reason": "Deterministic fallback already exhausted equivalent experiments.",
                "why_existing_tools_are_insufficient": "Equivalent experiment ID already completed.",
                "existing_tools_considered": list(APPROVED_TOOL_NAMES),
            },
        }
        diagnosis = "search_exhausted"
    evidence = [
        {
            "observation": item.get("name", "requirement"),
            "value": item.get("observed"),
            "interpretation": f"required {item.get('required')}",
        }
        for item in state.failed_requirements
        if isinstance(item, dict)
    ] or [{"observation": "acceptance_status", "value": state.acceptance_status, "interpretation": "failed"}]
    return AgentDecision.model_validate(
        {
            "diagnosis": diagnosis,
            "evidence": evidence,
            "hypothesis": "A constrained allowlisted experiment may improve development metrics.",
            "action": action,
            "expected_effect": {"cv_r2": "maintain_or_improve", "train_cv_gap": "decrease"},
            "success_conditions": {
                "minimum_cv_r2": (state.requirements or {}).get("minimum_cv_r2", 0.5),
                "maximum_train_cv_gap": (state.requirements or {}).get("maximum_train_cv_gap", 0.15),
            },
            "reason_existing_results_are_insufficient": "Hard requirements are still failed.",
            "confidence": "low",
        }
    )


def propose_decision(
    state: ModelingAgentState,
    *,
    use_openai: bool = True,
    openai_model: str | None = None,
    decision_fn: Callable[[ModelingAgentState], AgentDecision] | None = None,
) -> AgentDecision:
    if decision_fn is not None:
        return decision_fn(state)
    if not use_openai or not get_openai_api_key():
        return fallback_decision(state)

    system = (
        "You are a QSAR modeling-improvement agent. Diagnose failed development requirements "
        "and propose ONE allowlisted deterministic tool. Do not invent metrics. "
        "Do not use external-test information. Do not request Python, shell, or arbitrary code. "
        "Return JSON matching the AgentDecision schema."
    )
    user = (
        "Development-only state (sealed test omitted):\n"
        f"{json.dumps(_prompt_state(state), indent=2, default=str)}\n\n"
        f"Approved tools: {list(APPROVED_TOOL_NAMES)}\n"
        "Respond with JSON only."
    )
    try:
        from openai import OpenAI

        client = OpenAI(api_key=get_openai_api_key())
        response = client.chat.completions.create(
            model=openai_model or state.openai_model or get_openai_model(),
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0,
        )
        raw = response.choices[0].message.content or ""
        return AgentDecision.model_validate(_parse_agent_json(raw))
    except (json.JSONDecodeError, ValidationError, Exception):
        return fallback_decision(state)


def _prompt_state(state: ModelingAgentState) -> dict[str, Any]:
    view = dict(state.development_view or {})
    view.pop("applicability_domain", None)
    experiments = []
    for exp in (view.get("experiments") or [])[:12]:
        experiments.append(
            {
                "run_id": exp.get("run_id"),
                "model": exp.get("model"),
                "feature_selection_method": exp.get("feature_selection_method"),
                "feature_count": exp.get("feature_count"),
                "metrics": exp.get("metrics"),
                "is_winner": exp.get("is_winner"),
            }
        )
    return {
        "failed_requirements": state.failed_requirements,
        "acceptance_status": state.acceptance_status,
        "completed_experiment_ids": state.completed_experiment_ids,
        "current_best": state.current_best_candidate,
        "experiments": experiments,
        "adaptive_experiments_used": state.adaptive_experiments_used,
        "stagnation_count": state.stagnation_count,
        "requirements": state.requirements,
    }


def validate_decision(
    decision: AgentDecision,
    state: ModelingAgentState,
    config: WorkflowConfig,
) -> str:
    """Return an empty string if valid, otherwise a rejection reason."""
    try:
        assert_no_test_paths(decision.model_dump(), label="decision")
    except SealedTestAccessError as exc:
        return str(exc)
    if not decision.evidence:
        return "Decision does not cite evidence from structured results."
    if not decision.hypothesis.strip():
        return "Decision is missing a testable hypothesis."
    success = decision.success_conditions
    if success.minimum_cv_r2 is None and success.maximum_train_cv_gap is None and not success.extra:
        return "Decision proposes an action without a measurable success condition."
    tool = decision.action.tool_name
    if tool not in APPROVED_TOOL_NAMES:
        return f"Unsupported tool {tool!r}."
    args = decision.action.arguments or {}
    try:
        assert_no_test_paths(args, label="action.arguments")
    except SealedTestAccessError as exc:
        return str(exc)
    if "acceptance_criteria" in args or "requirements" in args:
        return "Decisions must not modify protected acceptance criteria."
    estimator = args.get("estimator") or args.get("model")
    if estimator and estimator not in SUPPORTED_ESTIMATORS:
        return f"Unsupported model family {estimator!r}."
    if args.get("param_grid") and estimator:
        sanitization = sanitize_param_grid(
            str(estimator),
            args["param_grid"],
            max_candidates=config.hpo.max_candidates_per_round,
        )
        requested = set(args["param_grid"])
        kept = [key for key in requested if key in (sanitization.sanitized_grid or {})]
        if not sanitization.sanitized_grid or not kept:
            return "Requested hyperparameters are not in the allowlisted parameter space."
        removed = sanitization.removed_params or sanitization.removed_values
        if removed and not sanitization.sanitized_grid:
            return "Unsupported model parameters were requested."
    limits = config.agentic_improvement.limits
    if state.adaptive_experiments_used >= limits.maximum_adaptive_experiments:
        return "Adaptive experiment budget is exhausted."
    if state.agent_iteration >= limits.maximum_agent_iterations:
        return "Iteration budget is exhausted."
    if (
        args.get("method") in {"pca", "pls"}
        and not config.agentic_improvement.allow_latent_components
        and not config.agentic_improvement.requirements.allow_latent_components
    ):
        return "Latent-component representations are not permitted by interpretability requirements."
    exp_id = make_experiment_id(
        tool_name=tool,
        arguments=args,
        dataset_hash=state.dataset_hash,
        development_split_hash=state.development_split_hash,
        parent_experiment_id=(state.current_best_candidate or {}).get("experiment_id"),
    )
    if exp_id in set(state.completed_experiment_ids):
        return f"Equivalent completed experiment {exp_id} would be repeated."
    return ""
