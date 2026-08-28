"""Recursion-limit handling must stop rather than loop forever."""

from __future__ import annotations

from langgraph.checkpoint.memory import InMemorySaver

from qsar_agent.agentic.runner import run_modeling_agent
from qsar_agent.config import AgenticImprovementSettings
from tests.agentic_harness import FAILING_METRICS, default_agent_config, make_decision, write_agent_run


def test_low_recursion_limit_sets_stopping_reason(tmp_path):
    config = default_agent_config()
    config.agentic_improvement = AgenticImprovementSettings(enabled=True, recursion_limit=8)
    run_dir = write_agent_run(tmp_path, passing=False, config=config)
    n = {"i": 0}
    calls: list[str] = []

    def decision_fn(_state):
        n["i"] += 1
        return make_decision("run_model_search", {"estimator": "RandomForestRegressor", "i": n["i"]})

    def execute_fn(tool, args, state):
        calls.append(tool)
        from qsar_agent.schemas.agentic import ToolResult

        return ToolResult(
            experiment_id=f"loop-{len(calls)}",
            tool_name=tool,
            arguments=args,
            metrics=FAILING_METRICS,
            selected_features=["a", "b", "c"],
        )

    state = run_modeling_agent(
        run_dir,
        use_openai=False,
        checkpointer=InMemorySaver(),
        decision_fn=decision_fn,
        execute_fn=execute_fn,
    )
    assert state.stopping_reason == "recursion_limit"
    assert state.report_path
    assert (run_dir / "agent_results" / "agent_final_report.md").is_file()
    assert "evaluate_sealed_test" not in calls
