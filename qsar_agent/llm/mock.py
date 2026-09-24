"""Scripted LLM test double.

For tests and offline demos **only**. It satisfies the same
:class:`~qsar_agent.llm.provider.LLMClient` contract as the production client, so
the workflow keeps its single execution path: the loop cannot tell that it is
talking to a stub rather than a model.

It is never selected implicitly — the caller must ask for provider ``mock``.
"""

from __future__ import annotations

import json
from typing import Any, Callable

from qsar_agent.llm.provider import (
    DEFAULT_MAX_TOOL_STEPS,
    LLMResponseError,
    LLMResult,
    ToolCallRecord,
    ToolSpec,
)
from qsar_agent.logging_utils import get_logger

logger = get_logger()

Responder = Callable[[dict[str, Any]], dict[str, Any]]


def _initial_plan(context: dict[str, Any]) -> dict[str, Any]:
    """A deliberately modest opening plan, so the loop has room to improve."""
    task = context.get("task", "regression")
    estimator = "Ridge" if task == "regression" else "LogisticRegression"
    return {
        "features": {
            "blocks": ["rdkit_descriptors"],
            "correlation_threshold": 0.95,
            "scale_features": True,
        },
        "model": {"estimator": estimator, "hyperparameters": {}},
        "rationale": (
            "Start from interpretable physicochemical descriptors with a linear baseline "
            "to establish whether the endpoint has a simple structure-activity trend."
        ),
    }


#: Ordered revisions the stub walks through, keyed by failure diagnosis. Each
#: entry mirrors an action a competent modeller would try for that diagnosis.
_REVISION_LADDER: dict[str, list[dict[str, Any]]] = {
    "underfit": [
        {
            "features": {"blocks": ["rdkit_descriptors", "morgan_fp"], "fingerprint_bits": 1024},
            "model": {"estimator": "RandomForest", "hyperparameters": {"n_estimators": 300}},
            "rationale": (
                "Linear model underfits, so add Morgan substructure bits for nonlinear "
                "structural signal and switch to a random forest."
            ),
        },
        {
            "features": {
                "blocks": ["rdkit_descriptors", "morgan_fp", "maccs_keys"],
                "fingerprint_bits": 2048,
            },
            "model": {
                "estimator": "GradientBoosting",
                "hyperparameters": {"n_estimators": 300, "learning_rate": 0.05, "max_depth": 3},
            },
            "rationale": "Still underfitting: widen the feature space and use boosted trees.",
        },
        {
            "features": {
                "blocks": ["rdkit_descriptors", "morgan_counts", "maccs_keys"],
                "fingerprint_bits": 2048,
            },
            "model": {"estimator": "ExtraTrees", "hyperparameters": {"n_estimators": 500}},
            "rationale": "Try count-based fingerprints with a higher-variance ensemble.",
        },
    ],
    "overfit": [
        {
            "features": {"blocks": ["rdkit_descriptors"], "correlation_threshold": 0.85,
                         "univariate_top_k": 40},
            "model": {
                "estimator": "RandomForest",
                "hyperparameters": {"n_estimators": 300, "max_depth": 6, "min_samples_leaf": 3},
            },
            "rationale": (
                "Cut the descriptor-to-sample ratio and constrain tree depth to close the "
                "train/CV gap."
            ),
        },
        {
            "features": {"blocks": ["rdkit_descriptors"], "correlation_threshold": 0.80,
                         "univariate_top_k": 20},
            "model": {"estimator": "Ridge", "hyperparameters": {"alpha": 10.0}},
            "rationale": "Fall back to a strongly regularised linear model on few features.",
        },
    ],
    "unstable_cv": [
        {
            "features": {"blocks": ["rdkit_descriptors"], "univariate_top_k": 30,
                         "correlation_threshold": 0.85},
            "model": {
                "estimator": "RandomForest",
                "hyperparameters": {"n_estimators": 500, "min_samples_leaf": 2},
            },
            "rationale": "Reduce feature noise and average more trees to stabilise CV scores.",
        },
    ],
    "chance_correlation": [
        {
            "features": {"blocks": ["rdkit_descriptors"], "univariate_top_k": 15,
                         "correlation_threshold": 0.80},
            "model": {"estimator": "Ridge", "hyperparameters": {"alpha": 1.0}},
            "rationale": (
                "Scrambled models score too well, which points at an inflated feature count; "
                "shrink it hard and use a simple estimator."
            ),
        },
    ],
    "poor_generalisation": [
        {
            "features": {"blocks": ["rdkit_descriptors", "morgan_fp"], "univariate_top_k": 60},
            "model": {
                "estimator": "RandomForest",
                "hyperparameters": {"n_estimators": 400, "max_depth": 8, "min_samples_leaf": 2},
            },
            "rationale": "CV is healthy but the hold-out is not; regularise and broaden features.",
        },
    ],
    "narrow_applicability_domain": [
        {
            "features": {
                "blocks": ["rdkit_descriptors"],
                "univariate_top_k": 25,
                "drop_ad_outliers": True,
            },
            "model": {"estimator": "RandomForest", "hyperparameters": {"n_estimators": 300}},
            "rationale": "Drop training outliers and lower dimensionality to widen the domain.",
        },
    ],
    "class_imbalance": [
        {
            "features": {"blocks": ["rdkit_descriptors", "morgan_fp"]},
            "model": {
                "estimator": "RandomForest",
                "hyperparameters": {"n_estimators": 300, "class_weight": "balanced"},
            },
            "rationale": "Weight classes inversely to frequency so the minority class counts.",
        },
    ],
}

_GENERIC_LADDER: list[dict[str, Any]] = [
    {
        "features": {"blocks": ["rdkit_descriptors", "morgan_fp"], "fingerprint_bits": 1024},
        "model": {"estimator": "RandomForest", "hyperparameters": {"n_estimators": 300}},
        "rationale": "Broaden features and use an ensemble as a general-purpose improvement.",
    },
    {
        "features": {"blocks": ["rdkit_descriptors", "morgan_fp", "maccs_keys"],
                     "fingerprint_bits": 2048},
        "model": {
            "estimator": "GradientBoosting",
            "hyperparameters": {"n_estimators": 250, "learning_rate": 0.05},
        },
        "rationale": "Escalate to boosted trees on a wider feature space.",
    },
]


def scripted_responder(context: dict[str, Any]) -> dict[str, Any]:
    """Produce strategist JSON for a request context, mimicking a capable planner."""
    request = context.get("request", "revision")
    if request == "initial_plan":
        return _initial_plan(context)

    diagnosis = str(context.get("diagnosis", {}).get("label", "")) or "unknown"
    tried = set(context.get("tried_signatures", []) or [])
    candidates = [*_REVISION_LADDER.get(diagnosis, []), *_GENERIC_LADDER]

    for candidate in candidates:
        revision = json.loads(json.dumps(candidate))
        revision["addresses_diagnosis"] = diagnosis
        signature = _candidate_signature(revision, context)
        if signature not in tried:
            return revision

    return {
        "stop": True,
        "stop_reason": (
            f"Exhausted the revisions I can justify for diagnosis '{diagnosis}' without "
            "repeating an already-tested configuration."
        ),
        "addresses_diagnosis": diagnosis,
    }


def _candidate_signature(revision: dict[str, Any], context: dict[str, Any]) -> str:
    """Recreate the loop's plan signature so the stub avoids repeating itself."""
    from qsar_agent.schemas.agentic import FeatureRecipe, IterationPlan, ModelPlan

    previous = context.get("current_plan") or {}
    fallback_features = {"blocks": ["rdkit_descriptors"]}
    features = revision.get("features") or previous.get("features") or fallback_features
    model = revision.get("model") or previous.get("model") or {"estimator": "RandomForest"}
    try:
        plan = IterationPlan(
            features=FeatureRecipe.model_validate(features),
            model=ModelPlan.model_validate(model),
        )
    except Exception:
        return json.dumps([features, model], sort_keys=True, default=str)
    return plan.signature()


class MockLLMClient:
    """Deterministic stand-in for a real model. Test and demo use only."""

    def __init__(
        self,
        responder: Responder | None = None,
        model: str = "mock-strategist",
        exercise_tools: bool = True,
    ):
        self.provider = "mock"
        self.model = model
        self._responder = responder or scripted_responder
        self._exercise_tools = exercise_tools
        self.calls: list[dict[str, Any]] = []
        logger.warning(
            "Using the mock LLM provider: replies are scripted, not model-generated. "
            "This mode exists for tests and offline demos only."
        )

    def complete_json(
        self,
        system: str,
        user: str,
        schema_hint: str,
        tools: list[ToolSpec] | None = None,
        max_tool_steps: int = DEFAULT_MAX_TOOL_STEPS,
    ) -> LLMResult:
        try:
            context = json.loads(user)
        except json.JSONDecodeError as exc:
            raise LLMResponseError(
                "MockLLMClient expects the user message to be a JSON context document."
            ) from exc
        self.calls.append(context)

        tool_calls: list[ToolCallRecord] = []
        if self._exercise_tools and tools:
            tool_calls = self._call_reference_tools(context, tools)

        data = self._responder(context)
        return LLMResult(
            data=data,
            raw_text=json.dumps(data),
            tool_calls=tool_calls,
            model=self.model,
            provider=self.provider,
        )

    def _call_reference_tools(
        self, context: dict[str, Any], tools: list[ToolSpec]
    ) -> list[ToolCallRecord]:
        """Exercise the tool path once per revision so the audit trail is realistic."""
        by_name = {spec.name: spec for spec in tools}
        records: list[ToolCallRecord] = []
        search = by_name.get("web_search")
        if search is None:
            return records
        diagnosis = context.get("diagnosis", {}).get("label", "model selection")
        task = context.get("task", "regression")
        query = f"QSAR {task} modelling how to address {diagnosis}"
        try:
            result = search.handler(query=query, max_results=3)
        except Exception as exc:
            result = {"error": f"{type(exc).__name__}: {exc}"}
        records.append(
            ToolCallRecord(
                name="web_search",
                arguments={"query": query, "max_results": 3},
                result_preview=json.dumps(result, default=str)[:600],
            )
        )
        return records


def mock_provider_factory(model: str | None = None, **_: Any) -> MockLLMClient:
    return MockLLMClient(model=model or "mock-strategist")
