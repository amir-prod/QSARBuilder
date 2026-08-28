"""Human interrupt for outlier exclusion approval."""

from __future__ import annotations

import json

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from qsar_agent.agentic.graph import compile_modeling_graph, thread_config
from qsar_agent.agentic.runner import GraphDeps, _DEPS, run_modeling_agent
from qsar_agent.config import AgentLimits
from qsar_agent.agentic.stopping import default_recursion_limit
from qsar_agent.schemas.agentic import ModelingAgentState, ToolResult
from tests.agentic_harness import (
    FAILING_METRICS,
    PASSING_METRICS,
    make_decision,
    write_agent_run,
    write_development_tables,
)


def _exclusion_decision():
    return make_decision(
        "run_exclusion_sensitivity_analysis",
        {"compound_id": "C000", "proposed_reason": "persistent OOF residual"},
        diagnosis="outlier",
    )


def test_outlier_proposal_interrupts_and_writes_audit(tmp_path):
    run_dir = write_agent_run(tmp_path, passing=False)
    original = write_development_tables(run_dir)
    train_path = run_dir / "preprocessed_train_descriptors.csv"
    before = train_path.read_text(encoding="utf-8")
    saver = InMemorySaver()
    compiled = compile_modeling_graph(run_dir, checkpointer=saver)
    token = _DEPS.set(
        GraphDeps(
            decision_fn=lambda _s: _exclusion_decision(),
            execute_fn=lambda tool, args, state: ToolResult(
                experiment_id="excl",
                tool_name=tool,
                metrics=FAILING_METRICS,
                selected_features=original[:3],
            ),
        )
    )
    try:
        config = thread_config(run_dir.name, default_recursion_limit(AgentLimits()))
        compiled.invoke(
            ModelingAgentState(
                run_dir=str(run_dir),
                project_id=run_dir.name,
                use_openai=False,
                phase="DEVELOPMENT",
            ).model_dump(mode="json"),
            config,
        )
        snapshot = compiled.get_state(config)
        assert "request_outlier_approval" in snapshot.next
        pending = run_dir / "agent_results" / "exclusion_proposals" / "pending.json"
        assert pending.is_file()
        payload = json.loads(pending.read_text(encoding="utf-8"))
        assert payload["compound_id"] == "C000"
        assert train_path.read_text(encoding="utf-8") == before
    finally:
        _DEPS.reset(token)


def test_approve_resume_runs_sensitivity_without_deleting_rows(tmp_path):
    run_dir = write_agent_run(tmp_path, passing=False)
    write_development_tables(run_dir)
    train_path = run_dir / "preprocessed_train_descriptors.csv"
    before = train_path.read_text(encoding="utf-8")
    saver = InMemorySaver()
    tools: list[str] = []

    def execute_fn(tool, args, state):
        tools.append(tool)
        metrics = PASSING_METRICS if tool != "evaluate_sealed_test" else {"test_r2": 0.4}
        return ToolResult(
            experiment_id=f"t-{len(tools)}",
            tool_name=tool,
            arguments=args,
            metrics=metrics,
            selected_features=["feat_0", "feat_1", "feat_2"],
            extra={"approved": args.get("approved")},
        )

    def decision_fn(_state):
        return _exclusion_decision()

    token = _DEPS.set(GraphDeps(decision_fn=decision_fn, execute_fn=execute_fn))
    try:
        compiled = compile_modeling_graph(run_dir, checkpointer=saver)
        config = thread_config(run_dir.name, default_recursion_limit(AgentLimits()))
        compiled.invoke(
            ModelingAgentState(
                run_dir=str(run_dir),
                project_id=run_dir.name,
                use_openai=False,
                phase="DEVELOPMENT",
            ).model_dump(mode="json"),
            config,
        )
        assert compiled.get_state(config).next
        assert "request_outlier_approval" in compiled.get_state(config).next
        compiled.invoke(Command(resume={"approved": True, "approver": "tester"}), config)
        snapshot = compiled.get_state(config)
        assert snapshot.next == ()
        assert "run_exclusion_sensitivity_analysis" in tools
        assert train_path.read_text(encoding="utf-8") == before
        values = snapshot.values
        phase = values.get("phase") if isinstance(values, dict) else getattr(values, "phase", None)
        assert str(getattr(phase, "value", phase)) == "EXTERNAL_EVALUATED"
    finally:
        _DEPS.reset(token)


def test_reject_resume_returns_to_diagnosis(tmp_path):
    run_dir = write_agent_run(tmp_path, passing=False)
    saver = InMemorySaver()
    n = {"i": 0}
    tools: list[str] = []

    def decision_fn(_state):
        n["i"] += 1
        if n["i"] == 1:
            return _exclusion_decision()
        return make_decision(
            "request_new_capability",
            {
                "capability": "none",
                "scientific_reason": "rejected exclusion",
                "why_existing_tools_are_insufficient": "test",
                "existing_tools_considered": ["run_exclusion_sensitivity_analysis"],
            },
        )

    def execute_fn(tool, args, state):
        tools.append(tool)
        return ToolResult(
            experiment_id=f"r-{len(tools)}",
            tool_name=tool,
            arguments=args,
            extra=dict(args),
        )

    token = _DEPS.set(GraphDeps(decision_fn=decision_fn, execute_fn=execute_fn))
    try:
        compiled = compile_modeling_graph(run_dir, checkpointer=saver)
        config = thread_config(run_dir.name, default_recursion_limit(AgentLimits()))
        compiled.invoke(
            ModelingAgentState(
                run_dir=str(run_dir),
                project_id=run_dir.name,
                use_openai=False,
                phase="DEVELOPMENT",
            ).model_dump(mode="json"),
            config,
        )
        compiled.invoke(Command(resume={"approved": False}), config)
        snapshot = compiled.get_state(config)
        assert snapshot.next == ()
        assert "run_exclusion_sensitivity_analysis" not in tools
        assert "request_new_capability" in tools
    finally:
        _DEPS.reset(token)


def test_cli_approval_resume_helper(tmp_path):
    """``run_modeling_agent(..., approval=)`` resumes with Command."""
    run_dir = write_agent_run(tmp_path, passing=False)
    write_development_tables(run_dir)
    saver = InMemorySaver()
    tools: list[str] = []

    def execute_fn(tool, args, state):
        tools.append(tool)
        metrics = PASSING_METRICS if tool != "evaluate_sealed_test" else {"test_r2": 0.2}
        return ToolResult(
            experiment_id=f"c-{len(tools)}",
            tool_name=tool,
            arguments=args,
            metrics=metrics,
            selected_features=["feat_0"],
        )

    run_modeling_agent(
        run_dir,
        use_openai=False,
        checkpointer=saver,
        decision_fn=lambda _s: _exclusion_decision(),
        execute_fn=execute_fn,
    )
    # First call hits the interrupt; runner still returns a state snapshot.
    pending = run_dir / "agent_results" / "exclusion_proposals" / "pending.json"
    assert pending.is_file()
    run_modeling_agent(
        run_dir,
        resume=True,
        approval={"approved": False},
        use_openai=False,
        checkpointer=saver,
        decision_fn=lambda _s: make_decision(
            "request_new_capability",
            {
                "capability": "none",
                "scientific_reason": "rejected",
                "why_existing_tools_are_insufficient": "test",
                "existing_tools_considered": ["run_exclusion_sensitivity_analysis"],
            },
        ),
        execute_fn=execute_fn,
    )
    assert "run_exclusion_sensitivity_analysis" not in tools
