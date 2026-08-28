"""Deterministic stopping rules for the modeling-improvement graph."""

from __future__ import annotations

from typing import Any

from qsar_agent.config import AgentLimits, WorkflowConfig


STOP_REQUIREMENTS_MET = "requirements_satisfied"
STOP_PLATEAU = "performance_plateau"
STOP_BUDGET = "budget_exhausted"
STOP_NO_ACTION = "no_scientifically_justified_action"
STOP_CAPABILITY = "missing_capability"
STOP_INTEGRITY = "integrity_check_failed"
STOP_RECURSION = "recursion_limit"


def default_recursion_limit(limits: AgentLimits) -> int:
    nodes_per_iteration = 10
    return int(limits.maximum_agent_iterations) * nodes_per_iteration + 20


def check_stopping(
    *,
    acceptance_status: str,
    agent_iteration: int,
    adaptive_experiments_used: int,
    compute_budget_used_hours: float,
    stagnation_count: int,
    model_families_used: int,
    representation_changes_used: int,
    pending_capability: bool,
    integrity_ok: bool,
    limits: AgentLimits,
    action_error: str = "",
) -> str:
    """Return a stopping reason or empty string to continue."""
    if not integrity_ok:
        return STOP_INTEGRITY
    if pending_capability:
        return STOP_CAPABILITY
    if acceptance_status == "passed":
        return STOP_REQUIREMENTS_MET
    if agent_iteration >= limits.maximum_agent_iterations:
        return STOP_BUDGET
    if adaptive_experiments_used >= limits.maximum_adaptive_experiments:
        return STOP_BUDGET
    if compute_budget_used_hours >= limits.maximum_runtime_hours:
        return STOP_BUDGET
    if model_families_used > limits.maximum_new_model_families:
        return STOP_BUDGET
    if representation_changes_used > limits.maximum_representation_changes:
        return STOP_BUDGET
    if stagnation_count >= limits.stagnation_window:
        return STOP_PLATEAU
    if action_error in {"no_justified_action", "unsupported_tool"}:
        return STOP_NO_ACTION
    return ""


def limits_from_config(config: WorkflowConfig) -> AgentLimits:
    return config.agentic_improvement.limits


def meaningful_improvement(history: list[float], minimum: float) -> bool:
    if len(history) < 2:
        return True
    return (history[-1] - max(history[:-1])) >= minimum
