"""Tool specifications the reasoning agents may call.

Each tool wraps a deterministic function. Nothing here computes a metric on the
model's behalf: the tools only expose facts the loop already established, plus
web search for outside knowledge.
"""

from __future__ import annotations

from typing import Any, Callable

from qsar_agent.llm.provider import ToolSpec
from qsar_agent.tools.model_zoo import describe_zoo
from qsar_agent.tools.rdkit_features import AVAILABLE_BLOCKS
from qsar_agent.tools.web_search import web_search


def make_web_search_tool(on_call: Callable[[], None] | None = None) -> ToolSpec:
    """Web search, with an optional counter hook for the run report."""

    def handler(query: str, max_results: int = 5) -> dict[str, Any]:
        if on_call is not None:
            on_call()
        return web_search(query, max_results=max_results).as_tool_output()

    return ToolSpec(
        name="web_search",
        description=(
            "Search the web for QSAR modelling guidance, descriptor recommendations, or "
            "published performance on similar endpoints. Returns titles, URLs and snippets. "
            "If the network is unavailable the result set is empty and a note explains why."
        ),
        parameters={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query."},
                "max_results": {
                    "type": "integer",
                    "description": "How many results to return (1-10).",
                    "default": 5,
                },
            },
            "required": ["query"],
        },
        handler=handler,
    )


def make_model_zoo_tool(task: str) -> ToolSpec:
    def handler() -> dict[str, Any]:
        return {"task": task, "estimators": describe_zoo(task)}

    return ToolSpec(
        name="list_model_zoo",
        description=(
            "List every estimator available for this task with its default and tunable "
            "hyperparameters. Only these estimators may be named in a plan."
        ),
        parameters={"type": "object", "properties": {}},
        handler=handler,
    )


def make_feature_blocks_tool() -> ToolSpec:
    def handler() -> dict[str, Any]:
        return {"feature_blocks": AVAILABLE_BLOCKS}

    return ToolSpec(
        name="list_feature_blocks",
        description=(
            "List every descriptor and fingerprint block that can appear in a feature recipe, "
            "with a short description of each."
        ),
        parameters={"type": "object", "properties": {}},
        handler=handler,
    )


def make_history_tool(history_provider: Callable[[], list[dict[str, Any]]]) -> ToolSpec:
    def handler() -> dict[str, Any]:
        return {"iterations": history_provider()}

    return ToolSpec(
        name="get_iteration_history",
        description=(
            "Return every iteration attempted so far: the feature recipe, estimator, "
            "hyperparameters, measured metrics, and failure diagnosis."
        ),
        parameters={"type": "object", "properties": {}},
        handler=handler,
    )


def make_dataset_summary_tool(summary_provider: Callable[[], dict[str, Any]]) -> ToolSpec:
    def handler() -> dict[str, Any]:
        return summary_provider()

    return ToolSpec(
        name="get_dataset_summary",
        description=(
            "Return dataset facts: compound count, split sizes, task type, activity "
            "distribution, and class balance for classification."
        ),
        parameters={"type": "object", "properties": {}},
        handler=handler,
    )


def make_criteria_tool(criteria_provider: Callable[[], dict[str, Any]]) -> ToolSpec:
    def handler() -> dict[str, Any]:
        return criteria_provider()

    return ToolSpec(
        name="get_acceptance_criteria",
        description=(
            "Return the acceptance criteria the user specified for this run. These are "
            "read-only: they cannot be changed from inside the loop."
        ),
        parameters={"type": "object", "properties": {}},
        handler=handler,
    )


def build_strategist_toolset(
    task: str,
    history_provider: Callable[[], list[dict[str, Any]]],
    summary_provider: Callable[[], dict[str, Any]],
    criteria_provider: Callable[[], dict[str, Any]],
    on_web_search: Callable[[], None] | None = None,
) -> list[ToolSpec]:
    """Assemble the full toolset handed to the Strategist."""
    return [
        make_web_search_tool(on_call=on_web_search),
        make_model_zoo_tool(task),
        make_feature_blocks_tool(),
        make_history_tool(history_provider),
        make_dataset_summary_tool(summary_provider),
        make_criteria_tool(criteria_provider),
    ]
