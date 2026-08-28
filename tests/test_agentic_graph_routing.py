"""LangGraph conditional routing for the modeling-improvement agent."""

from __future__ import annotations

from langgraph.checkpoint.memory import InMemorySaver

from qsar_agent.agentic.graph import build_modeling_graph
from qsar_agent.agentic.runner import run_modeling_agent
from qsar_agent.agentic.stopping import STOP_BUDGET, STOP_CAPABILITY, STOP_INTEGRITY
from qsar_agent.config import AgentLimits, AgenticImprovementSettings
from qsar_agent.schemas.agentic import PipelinePhase
from tests.agentic_harness import (
    FAILING_METRICS,
    default_agent_config,
    make_decision,
    recording_execute,
    write_agent_run,
)


def _edges(compiled):
    return {(edge.source, edge.target) for edge in compiled.get_graph().edges}


def test_compiled_graph_has_specified_routes():
    compiled = build_modeling_graph(InMemorySaver())
    edges = _edges(compiled)
    assert ("__start__", "load_and_validate_handoff") in edges
    assert ("load_and_validate_handoff", "create_development_safe_view") in edges
    assert ("load_and_validate_handoff", "write_final_report") in edges
    assert ("create_development_safe_view", "evaluate_requirements") in edges
    assert ("evaluate_requirements", "freeze_and_refit_pipeline") in edges
    assert ("evaluate_requirements", "diagnose_failure") in edges
    assert ("validate_action", "execute_deterministic_tool") in edges
    assert ("validate_action", "request_capability") in edges
    assert ("validate_action", "request_outlier_approval") in edges
    assert ("validate_action", "check_stopping_conditions") in edges
    assert ("request_capability", "write_final_report") in edges
    assert ("freeze_and_refit_pipeline", "evaluate_sealed_test") in edges
    assert ("evaluate_sealed_test", "write_final_report") in edges
    assert ("write_final_report", "__end__") in edges


def test_invalid_handoff_routes_to_report(tmp_path):
    run_dir = tmp_path / "broken"
    run_dir.mkdir()
    (run_dir / "final_report").mkdir()
    state = run_modeling_agent(
        run_dir,
        use_openai=False,
        checkpointer=InMemorySaver(),
        execute_fn=recording_execute([]),
    )
    assert state.validation_passed is False
    assert state.stopping_reason == STOP_INTEGRITY
    assert state.report_path
    assert (run_dir / "agent_results" / "agent_final_report.md").is_file()
    assert state.sealed_test_result is None


def test_requirements_satisfied_freezes_then_sealed_test_once(tmp_path):
    calls: list[str] = []
    run_dir = write_agent_run(tmp_path, passing=True)
    state = run_modeling_agent(
        run_dir,
        use_openai=False,
        checkpointer=InMemorySaver(),
        execute_fn=recording_execute(calls),
    )
    assert calls.count("freeze_pipeline") == 1
    assert calls.count("evaluate_sealed_test") == 1
    assert calls.index("freeze_pipeline") < calls.index("evaluate_sealed_test")
    assert state.phase == PipelinePhase.EXTERNAL_EVALUATED
    assert state.sealed_test_result is not None
    assert state.report_path
    diagnose_calls = [c for c in calls if c == "run_model_search"]
    assert diagnose_calls == []


def test_valid_action_executes_then_can_freeze(tmp_path):
    calls: list[str] = []
    run_dir = write_agent_run(tmp_path, passing=False)

    def decision_fn(_state):
        return make_decision("run_model_search", {"estimator": "RandomForestRegressor"})

    state = run_modeling_agent(
        run_dir,
        use_openai=False,
        checkpointer=InMemorySaver(),
        decision_fn=decision_fn,
        execute_fn=recording_execute(calls),
    )
    assert "run_model_search" in calls
    assert "freeze_pipeline" in calls
    assert "evaluate_sealed_test" in calls
    assert state.phase == PipelinePhase.EXTERNAL_EVALUATED
    assert state.completed_experiment_ids


def test_missing_capability_requests_and_stops(tmp_path):
    calls: list[str] = []
    run_dir = write_agent_run(tmp_path, passing=False)

    def decision_fn(_state):
        return make_decision(
            "request_new_capability",
            {
                "capability": "xtb_rerun",
                "scientific_reason": "Need a new representation.",
                "why_existing_tools_are_insufficient": "RDKit/Mordred already used.",
                "existing_tools_considered": ["run_representation_experiment"],
            },
        )

    state = run_modeling_agent(
        run_dir,
        use_openai=False,
        checkpointer=InMemorySaver(),
        decision_fn=decision_fn,
        execute_fn=recording_execute(calls),
    )
    assert "request_new_capability" in calls
    assert "freeze_pipeline" not in calls
    assert "evaluate_sealed_test" not in calls
    assert state.stopping_reason == STOP_CAPABILITY
    assert state.pending_capability_request
    assert state.sealed_test_result is None


def test_budget_exhaustion_writes_report_without_sealed_eval(tmp_path):
    calls: list[str] = []
    config = default_agent_config()
    config.agentic_improvement = AgenticImprovementSettings(
        enabled=True,
        limits=AgentLimits(maximum_adaptive_experiments=1, maximum_agent_iterations=12),
    )
    run_dir = write_agent_run(tmp_path, passing=False, config=config)
    n = {"i": 0}

    def decision_fn(_state):
        n["i"] += 1
        return make_decision(
            "run_model_search",
            {"estimator": "RandomForestRegressor", "tag": n["i"]},
        )

    state = run_modeling_agent(
        run_dir,
        use_openai=False,
        checkpointer=InMemorySaver(),
        decision_fn=decision_fn,
        execute_fn=recording_execute(calls, metrics=FAILING_METRICS),
    )
    assert state.stopping_reason == STOP_BUDGET
    assert "evaluate_sealed_test" not in calls
    assert (run_dir / "agent_results" / "agent_final_report.md").is_file()


def test_agentic_improvement_defaults_off():
    from qsar_agent.config import WorkflowConfig

    assert WorkflowConfig().agentic_improvement.enabled is False
