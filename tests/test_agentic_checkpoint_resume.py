"""Sqlite / in-memory checkpoint resume and experiment-id idempotency."""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

from langgraph.checkpoint.memory import InMemorySaver

from qsar_agent.agentic.runner import run_modeling_agent
from tests.agentic_harness import FAILING_METRICS, make_decision, write_agent_run


def test_crash_resume_does_not_reexecute_completed_ids(tmp_path):
    run_dir = write_agent_run(tmp_path, passing=False)
    calls: list[str] = []
    flags = {"allow": False}

    def decision_fn(_state):
        n = sum(1 for item in calls if item.startswith("decision"))
        calls.append(f"decision-{n + 1}")
        if len([c for c in calls if c.startswith("decision")]) == 2 and not flags["allow"]:
            raise RuntimeError("simulated crash after first experiment")
        if flags["allow"] or n >= 1:
            return make_decision(
                "request_new_capability",
                {
                    "capability": "stop",
                    "scientific_reason": "resume path",
                    "why_existing_tools_are_insufficient": "test",
                    "existing_tools_considered": ["run_model_search"],
                },
            )
        return make_decision("run_model_search", {"estimator": "RandomForestRegressor"})

    def execute_fn(tool, args, state):
        calls.append(f"exec:{tool}")
        from qsar_agent.schemas.agentic import ToolResult

        metrics = dict(FAILING_METRICS)
        if tool == "evaluate_sealed_test":
            metrics = {"test_r2": 0.1}
        return ToolResult(
            experiment_id=f"id-{tool}",
            tool_name=tool,
            arguments=args,
            metrics=metrics,
            selected_features=["a", "b", "c"],
            extra=dict(args),
        )

    saver = InMemorySaver()
    try:
        run_modeling_agent(
            run_dir,
            use_openai=False,
            checkpointer=saver,
            decision_fn=decision_fn,
            execute_fn=execute_fn,
        )
        raise AssertionError("expected simulated crash")
    except RuntimeError as exc:
        assert "simulated crash" in str(exc)

    assert calls.count("exec:run_model_search") == 1
    flags["allow"] = True
    resumed = run_modeling_agent(
        run_dir,
        resume=True,
        use_openai=False,
        checkpointer=saver,
        decision_fn=decision_fn,
        execute_fn=execute_fn,
    )
    assert calls.count("exec:run_model_search") == 1
    assert "exec:request_new_capability" in calls
    assert resumed.report_path


def test_sqlite_checkpoint_file_is_created_and_resumable(tmp_path):
    """SqliteSaver is exercised in a child process so conda libstdc++ loads first."""
    run_dir = write_agent_run(tmp_path, passing=True)
    env = os.environ.copy()
    lib = Path(sys.prefix) / "lib"
    env["LD_LIBRARY_PATH"] = str(lib) + os.pathsep + env.get("LD_LIBRARY_PATH", "")
    project = str(Path(__file__).resolve().parents[1])
    env["PYTHONPATH"] = project + os.pathsep + env.get("PYTHONPATH", "")
    script = textwrap.dedent(
        f"""
        from pathlib import Path
        from qsar_agent.agentic.graph import compile_modeling_graph, thread_config
        from qsar_agent.agentic.runner import run_modeling_agent
        from qsar_agent.agentic.stopping import default_recursion_limit
        from qsar_agent.config import AgentLimits
        from qsar_agent.schemas.agentic import ToolResult

        run_dir = Path({str(run_dir)!r})

        def execute_fn(tool, args, state):
            metrics = {{"test_r2": 0.4}} if tool == "evaluate_sealed_test" else {{}}
            return ToolResult(
                experiment_id=f"sqlite-{{tool}}",
                tool_name=tool,
                arguments=args,
                metrics=metrics,
                selected_features=["a"],
            )

        run_modeling_agent(run_dir, use_openai=False, execute_fn=execute_fn)
        db = run_dir / "agent_results" / "langgraph_checkpoints.sqlite"
        assert db.is_file() and db.stat().st_size > 0, db
        compiled = compile_modeling_graph(run_dir)
        snapshot = compiled.get_state(
            thread_config(run_dir.name, default_recursion_limit(AgentLimits()))
        )
        assert snapshot.next == ()
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    db = run_dir / "agent_results" / "langgraph_checkpoints.sqlite"
    assert db.is_file()
    assert db.stat().st_size > 0
