"""Tests for resuming agentic improvement from a prior deterministic run."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from qsar_agent.agentic.artifact_view import assert_no_external_test_fields
from qsar_agent.agentic.provider import MockAgentProvider
from qsar_agent.config import WorkflowConfig
from qsar_agent.schemas.agentic import (
    AgenticAcceptanceCriteria,
    AgenticImprovementConfig,
    AgentDiagnosis,
    SupervisorDecision,
)
from qsar_agent.schemas.hyperparameter_optimization import FinalModelSelection
from qsar_agent.services.resume_agentic import (
    InPlaceResumeForbiddenError,
    assert_not_inplace_on_tainted,
    detect_external_access,
    fork_run_for_agentic,
    list_resumable_runs,
    load_winner_from_run,
    run_agentic_only,
)
from qsar_agent.tools.overfitting_assessment import assess_overfitting
from tests.test_hyperparameter_optimization import _cv_summary


def _write_winner_run(run_dir: Path, *, with_external: bool = True) -> Path:
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(0)
    X = rng.randn(40, 5)
    y = X[:, 0] + rng.randn(40) * 0.1
    cols = [f"f{i}" for i in range(5)]
    train = pd.DataFrame(X, columns=cols)
    train["activity"] = y
    train["compound_id"] = [f"c{i}" for i in range(40)]
    train["canonical_smiles"] = ["CCO"] * 40
    train.to_csv(run_dir / "preprocessed_train_descriptors.csv", index=False)
    test = train.iloc[:10].copy()
    test.to_csv(run_dir / "preprocessed_test_descriptors.csv", index=False)

    summary = _cv_summary(0.75, 0.55, 0.08)
    assessment = assess_overfitting(summary)
    selection = FinalModelSelection(
        source="baseline",
        params={"alpha": 1.0},
        cv_summary=summary,
        assessment=assessment,
        selection_rationale="test",
    )
    comparison = {
        "winning_estimator": "Ridge (sfs_fixed_ga_plus2)",
        "selected_features": cols[:3],
        "final_model_config": {
            "estimator": "Ridge",
            "params": {"alpha": 1.0},
            "random_state": 0,
            "n_jobs": 1,
        },
        "final_selection": selection.model_dump(),
        "selection_rationale": "winner",
    }
    (run_dir / "model_comparison.json").write_text(
        json.dumps(comparison, indent=2), encoding="utf-8"
    )
    (run_dir / "run_manifest.json").write_text(
        json.dumps(
            {
                "workflow_config": WorkflowConfig(random_seed=0).model_dump(),
                "selected_features": cols[:3],
                "dataset_hash": "abc",
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    if with_external:
        (run_dir / "predictions.csv").write_text("compound_id,activity\nc0,1\n", encoding="utf-8")
        (run_dir / "model_metrics.json").write_text(
            json.dumps({"test": {"r2": 0.99}}), encoding="utf-8"
        )
    return run_dir


def test_load_winner_from_model_comparison(tmp_path):
    run_dir = _write_winner_run(tmp_path, with_external=False)
    winner = load_winner_from_run(run_dir)
    assert winner.estimator == "Ridge"
    assert winner.winning_estimator_label.startswith("Ridge")
    assert len(winner.selected_features) == 3
    assert winner.mean_cv_r2 == pytest.approx(0.55)


def test_detect_external_access(tmp_path):
    clean = _write_winner_run(tmp_path / "clean", with_external=False)
    tainted = _write_winner_run(tmp_path / "tainted", with_external=True)
    assert not detect_external_access(clean).external_previously_evaluated
    info = detect_external_access(tainted)
    assert info.external_previously_evaluated
    assert any("predictions.csv" in r or "model_metrics" in r for r in info.reasons)


def test_fork_does_not_copy_external_artifacts(tmp_path):
    source = _write_winner_run(tmp_path / "src", with_external=True)
    out = tmp_path / "outputs"
    fork_dir, meta = fork_run_for_agentic(source, output_root=out)
    assert meta["external_previously_evaluated"] is True
    assert (fork_dir / "preprocessed_train_descriptors.csv").exists()
    assert not (fork_dir / "predictions.csv").exists()
    assert not (fork_dir / "model_metrics.json").exists()
    assert (fork_dir / "source_external_reference" / "README.txt").exists()
    # Quarantine must not contain metric values for agents
    ref_metrics = fork_dir / "source_external_reference" / "model_metrics.json"
    assert not ref_metrics.exists()


def test_inplace_resume_on_tainted_raises(tmp_path):
    source = _write_winner_run(tmp_path, with_external=True)
    with pytest.raises(InPlaceResumeForbiddenError):
        assert_not_inplace_on_tainted(source, source)


def test_list_resumable_runs(tmp_path):
    _write_winner_run(tmp_path / "outputs" / "aaa", with_external=True)
    (tmp_path / "outputs" / "bbb").mkdir(parents=True)
    runs = list_resumable_runs(tmp_path / "outputs")
    assert any(r.run_id == "aaa" for r in runs)
    assert all(r.run_id != "bbb" for r in runs)


def test_run_agentic_only_mock_provider(tmp_path):
    source = _write_winner_run(tmp_path / "src", with_external=True)
    out = tmp_path / "outputs"

    def default_factory(agent_name, response_model, payload):
        assert_no_external_test_fields(payload)
        if agent_name == "supervisor":
            return SupervisorDecision(
                cycle_index=1,
                selected_proposal_id=None,
                action="stop_no_viable_model",
                rationale="stop",
            )
        return AgentDiagnosis(
            agent_name=agent_name,
            experiment_id="exp_001",
            failure_category="poor_performance",
            summary="mock",
            recommended_actions=["stop_no_viable_model"],
        )

    provider = MockAgentProvider(default_factory=default_factory)
    cfg = WorkflowConfig(
        random_seed=0,
        agentic=AgenticImprovementConfig(
            enabled=True,
            max_cycles=1,
            max_total_experiments=2,
            acceptance=AgenticAcceptanceCriteria(
                minimum_mean_cv_r2=0.99,
                require_validation_agent_approval=False,
                require_non_overfit_status=False,
            ),
        ),
    )
    result = run_agentic_only(
        source,
        workflow_config=cfg,
        output_root=out,
        evaluate_external_after_lock=False,
        provider=provider,
    )
    assert result.external_previously_evaluated is True
    assert result.forked_run_id != source.name
    assert result.forked_run_id.startswith(source.name + "_agentic_")
    assert Path(result.forked_run_dir).exists()
    assert not (Path(result.forked_run_dir) / "predictions.csv").exists()
    assert result.agentic_state.status == "model_locked"
    assert result.disclaimer is None  # no external eval requested
    for call in provider.calls:
        assert_no_external_test_fields(call["payload"])


def test_external_eval_sets_disclaimer_when_source_tainted(tmp_path):
    source = _write_winner_run(tmp_path / "src", with_external=True)
    out = tmp_path / "outputs"
    cfg = WorkflowConfig(
        random_seed=0,
        agentic=AgenticImprovementConfig(
            enabled=True,
            max_cycles=1,
            max_total_experiments=2,
            acceptance=AgenticAcceptanceCriteria(
                minimum_mean_cv_r2=0.99,
                require_validation_agent_approval=False,
                require_non_overfit_status=False,
            ),
        ),
    )

    def default_factory(agent_name, response_model, payload):
        if agent_name == "supervisor":
            return SupervisorDecision(
                cycle_index=1,
                action="stop_no_viable_model",
                rationale="stop",
            )
        return AgentDiagnosis(
            agent_name=agent_name,
            experiment_id="exp_001",
            failure_category="poor_performance",
            summary="mock",
            recommended_actions=["stop_no_viable_model"],
        )

    result = run_agentic_only(
        source,
        workflow_config=cfg,
        output_root=out,
        evaluate_external_after_lock=True,
        provider=MockAgentProvider(default_factory=default_factory),
    )
    assert result.evaluated_external is True
    assert result.disclaimer is not None
    disc = Path(result.forked_run_dir) / "locked_external" / "external_independence_disclaimer.json"
    assert disc.exists()
    payload = json.loads(disc.read_text(encoding="utf-8"))
    assert payload["external_previously_evaluated"] is True
