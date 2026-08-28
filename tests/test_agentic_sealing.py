"""Sealed external-test isolation for development nodes and the graph."""

from __future__ import annotations

import pytest
from langgraph.checkpoint.memory import InMemorySaver

from qsar_agent.agentic.graph import (
    build_modeling_graph,
    compare_candidates,
    create_development_safe_view,
    diagnose_failure,
    evaluate_sealed_test_node,
    execute_deterministic_tool,
    form_hypothesis,
    ingest_result,
    load_and_validate_handoff,
    validate_action,
)
from qsar_agent.agentic.sealing import (
    SealedTestAccessError,
    assert_no_test_paths,
    development_view,
    development_view_dict,
)
from qsar_agent.schemas.agentic import ModelingAgentState, PipelinePhase
from qsar_agent.schemas.handoff import ExternalTestMetrics, LargestErrorCompound
from tests.agentic_harness import write_agent_run
from tests.test_handoff import _experiment, _package


def test_development_view_strips_test_outcomes():
    exp = _experiment("winner1")
    exp.external_test = ExternalTestMetrics(r2=0.99, rmse=0.01, mae=0.01, n=5)
    exp.artifacts.test_predictions = "plots/winner1_test_predictions.csv"
    package = _package(experiments=[exp])
    package.error_analysis.largest_error_compounds = [
        LargestErrorCompound(
            compound_id="t1",
            split="test",
            activity=1.0,
            predicted_activity=0.1,
            residual=0.9,
            abs_residual=0.9,
        ),
        LargestErrorCompound(
            compound_id="tr1",
            split="train",
            activity=1.0,
            predicted_activity=0.9,
            residual=0.1,
            abs_residual=0.1,
        ),
    ]
    package.applicability_domain.outliers_by_partition = {
        "test": {"structural": ["t1"]},
        "train": {"structural": []},
    }
    view = development_view(package)
    assert view.experiments[0].external_test.r2 is None
    assert view.experiments[0].artifacts.test_predictions is None
    assert view.experiments[0].artifacts.williams.status == "unavailable"
    assert view.experiments[0].artifacts.residuals.status == "unavailable"
    assert all(row.split != "test" for row in view.error_analysis.largest_error_compounds)
    assert "test" not in (view.applicability_domain.outliers_by_partition or {})
    dumped = development_view_dict(package)
    assert "external_test" not in dumped["experiments"][0]
    assert_no_test_paths(dumped, label="development_view")


def test_create_development_safe_view_omits_test_metrics(tmp_path):
    run_dir = write_agent_run(tmp_path, passing=True)
    state = ModelingAgentState(run_dir=str(run_dir), phase=PipelinePhase.DEVELOPMENT)
    loaded = load_and_validate_handoff(state)
    state = state.model_copy(update=loaded)
    out = create_development_safe_view(state)
    view = out["development_view"]
    assert "external_test" not in view["experiments"][0]
    metrics = view["experiments"][0]["metrics"]
    assert "test_r2" not in metrics
    assert metrics.get("cv_r2") is not None


@pytest.mark.parametrize(
    "node",
    [
        diagnose_failure,
        form_hypothesis,
        validate_action,
        execute_deterministic_tool,
        ingest_result,
        compare_candidates,
    ],
)
def test_development_nodes_refuse_sealed_test_state(node, tmp_path):
    run_dir = write_agent_run(tmp_path, passing=False)
    state = ModelingAgentState(
        run_dir=str(run_dir),
        phase=PipelinePhase.DEVELOPMENT,
        sealed_test_result={"test_r2": 0.91, "test_rmse": 0.1},
        last_decision={
            "diagnosis": "x",
            "evidence": [{"observation": "cv_r2", "value": 0.2, "interpretation": "low"}],
            "hypothesis": "try something",
            "action": {"tool_name": "run_model_search", "arguments": {}},
            "success_conditions": {"minimum_cv_r2": 0.5},
            "reason_existing_results_are_insufficient": "failed",
        },
        proposed_action={"tool_name": "run_model_search", "arguments": {}},
        last_tool_result={"experiment_id": "e1", "metrics": {}},
        development_view={"experiments": []},
    )
    with pytest.raises(SealedTestAccessError):
        node(state)


def test_development_nodes_refuse_frozen_phase(tmp_path):
    run_dir = write_agent_run(tmp_path, passing=False)
    state = ModelingAgentState(run_dir=str(run_dir), phase=PipelinePhase.FROZEN)
    with pytest.raises(SealedTestAccessError, match="phase"):
        diagnose_failure(state)


def test_evaluate_sealed_test_requires_frozen(tmp_path):
    run_dir = write_agent_run(tmp_path, passing=True)
    state = ModelingAgentState(run_dir=str(run_dir), phase=PipelinePhase.DEVELOPMENT)
    with pytest.raises(SealedTestAccessError, match="FROZEN"):
        evaluate_sealed_test_node(state)


def test_execute_rejects_test_artifact_paths(tmp_path):
    run_dir = write_agent_run(tmp_path, passing=False)
    state = ModelingAgentState(
        run_dir=str(run_dir),
        phase=PipelinePhase.DEVELOPMENT,
        proposed_action={
            "tool_name": "run_model_search",
            "arguments": {"test_path": str(run_dir / "preprocessed_test_descriptors.csv")},
        },
    )
    with pytest.raises(SealedTestAccessError):
        execute_deterministic_tool(state)


def test_sealed_eval_has_no_edge_back_to_diagnosis():
    compiled = build_modeling_graph(InMemorySaver())
    edges = {(edge.source, edge.target) for edge in compiled.get_graph().edges}
    assert ("evaluate_sealed_test", "write_final_report") in edges
    assert ("evaluate_sealed_test", "diagnose_failure") not in edges
    assert ("evaluate_sealed_test", "evaluate_requirements") not in edges
    assert ("evaluate_sealed_test", "form_hypothesis") not in edges
    sources_into_diagnose = {src for src, dst in edges if dst == "diagnose_failure"}
    assert "evaluate_sealed_test" not in sources_into_diagnose
