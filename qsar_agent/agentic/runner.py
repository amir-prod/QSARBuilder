"""Run or resume the modeling-improvement LangGraph after a deterministic handoff."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from qsar_agent.agentic.graph import GraphDeps, _DEPS, compile_modeling_graph, invoke_modeling_graph
from qsar_agent.agentic.stopping import default_recursion_limit
from qsar_agent.agentic.tools import agent_dir, load_workflow_config
from qsar_agent.schemas.agentic import ModelingAgentState


def run_modeling_agent(
    run_dir: str | Path,
    *,
    resume: bool = False,
    approval: dict[str, Any] | None = None,
    use_openai: bool = True,
    checkpointer: Any | None = None,
    decision_fn: Any | None = None,
    execute_fn: Any | None = None,
) -> ModelingAgentState:
    """Invoke the compiled graph. Failures here must not erase ``final_report/``."""
    run_dir = Path(run_dir)
    config = load_workflow_config(run_dir)
    agent_dir(run_dir)
    compiled = compile_modeling_graph(run_dir, checkpointer=checkpointer)
    limits = config.agentic_improvement.limits
    recursion = config.agentic_improvement.recursion_limit or default_recursion_limit(limits)
    token = _DEPS.set(GraphDeps(decision_fn=decision_fn, execute_fn=execute_fn, config=config))
    try:
        initial = ModelingAgentState(
            run_dir=str(run_dir),
            project_id=run_dir.name,
            use_openai=use_openai,
            openai_model=config.agentic_improvement.openai_model,
            phase="DEVELOPMENT",
        )
        return invoke_modeling_graph(
            compiled,
            initial,
            resume=resume or approval is not None,
            resume_value=approval,
            recursion_limit=recursion,
        )
    finally:
        _DEPS.reset(token)
