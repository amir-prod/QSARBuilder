"""Integration-style tests for agentic loop with mock provider."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from qsar_agent.agentic.artifact_view import assert_no_external_test_fields
from qsar_agent.agentic.loop import AgenticImprovementLoop
from qsar_agent.agentic.provider import MockAgentProvider
from qsar_agent.agentic.summary_builder import build_agent_visible_summary
from qsar_agent.config import WorkflowConfig
from qsar_agent.schemas.agentic import (
    AgenticAcceptanceCriteria,
    AgenticImprovementConfig,
    AgentDiagnosis,
    SupervisorDecision,
)
from qsar_agent.schemas.hyperparameter_optimization import (
    FinalModelSelection,
    HPOConfig,
    OverfittingThresholds,
)
from qsar_agent.tools.overfitting_assessment import assess_overfitting
from tests.test_hyperparameter_optimization import _cv_summary


def _final(mean_cv=0.3, mean_train=0.8, std=0.2):
    summary = _cv_summary(mean_train, mean_cv, std)
    assessment = assess_overfitting(summary)
    return FinalModelSelection(
        source="baseline",
        params={},
        cv_summary=summary,
        assessment=assessment,
        selection_rationale="test",
    )


def _train_csv(tmp_path: Path, n=40, p=6):
    rng = np.random.RandomState(0)
    X = rng.randn(n, p)
    y = X[:, 0] + rng.randn(n) * 0.2
    cols = [f"d{i}" for i in range(p)]
    df = pd.DataFrame(X, columns=cols)
    df["activity"] = y
    path = tmp_path / "dev.csv"
    df.to_csv(path, index=False)
    return path, cols


def test_agentic_loop_mock_provider_locks_model(tmp_path):
    train_path, feats = _train_csv(tmp_path)
    cfg = WorkflowConfig(
        random_seed=0,
        agentic=AgenticImprovementConfig(
            enabled=True,
            max_cycles=1,
            max_total_experiments=3,
            acceptance=AgenticAcceptanceCriteria(
                minimum_mean_cv_r2=0.99,
                require_validation_agent_approval=False,
                require_non_overfit_status=False,
            ),
        ),
    )
    hpo = HPOConfig(enabled=False, cv_folds=3, thresholds=OverfittingThresholds())

    def default_factory(agent_name, response_model, payload):
        if agent_name == "supervisor":
            return SupervisorDecision(
                cycle_index=1,
                selected_proposal_id=None,
                action="stop_no_viable_model",
                rationale="stop",
                decision_source="llm_agent",
            )
        return AgentDiagnosis(
            agent_name=agent_name,
            experiment_id="exp_001",
            failure_category="poor_performance",
            summary="mock",
            recommended_actions=["stop_no_viable_model"],
        )

    provider = MockAgentProvider(default_factory=default_factory)
    loop = AgenticImprovementLoop(
        run_dir=tmp_path,
        workflow_config=cfg,
        hpo_config=hpo,
        development_train_path=train_path,
        selected_features=feats[:3],
        dataset_hash="hash",
        initial_estimator="RandomForestRegressor",
        initial_final_selection=_final(),
        provider=provider,
    )
    state = loop.run()
    assert state.status == "model_locked"
    assert state.lock_record is not None
    assert (tmp_path / "agent_workspace" / "final_agent_report.md").exists()
    # Provider payloads must not contain external-test metrics
    for call in provider.calls:
        assert_no_external_test_fields(call["payload"])


def test_external_test_values_cannot_change_summary_ranking(tmp_path):
    """Mutating external-test artifacts must not alter agent-visible summary metrics."""
    metrics = {
        "mean_cv_r2": 0.55,
        "mean_train_r2": 0.60,
        "train_cv_gap": 0.05,
        "cv_r2_std": 0.04,
        "feature_count": 3,
        "selected_features": ["a", "b", "c"],
        "estimator": "Ridge",
        "agent_dev_size": 30,
    }
    s1 = build_agent_visible_summary(
        experiment_id="exp_001",
        run_dir=tmp_path,
        internal_metrics=metrics,
        estimator="Ridge",
        selected_features=["a", "b", "c"],
    )
    # Write fake external metrics that should be ignored
    (tmp_path / "model_metrics.json").write_text('{"test_r2": 0.99}', encoding="utf-8")
    (tmp_path / "predictions.csv").write_text("y,yhat\n1,2\n", encoding="utf-8")
    s2 = build_agent_visible_summary(
        experiment_id="exp_002",
        run_dir=tmp_path,
        internal_metrics=metrics,
        estimator="Ridge",
        selected_features=["a", "b", "c"],
    )
    assert s1.mean_cv_r2 == s2.mean_cv_r2 == 0.55
    assert s1.external_test_unavailable and s2.external_test_unavailable

    from qsar_agent.agentic.selection import select_best_experiment
    from qsar_agent.schemas.agentic import ExperimentRecord

    r1 = ExperimentRecord(experiment_id="exp_001", internal_metrics=dict(metrics), feature_count=3, status="completed")
    r2 = ExperimentRecord(
        experiment_id="exp_002",
        internal_metrics={**metrics, "mean_cv_r2": 0.56},
        feature_count=3,
        status="completed",
    )
    # Even if external files claim 0.99, ranking uses internal_metrics only
    best, _ = select_best_experiment([r1, r2])
    assert best.experiment_id == "exp_002"


def test_graceful_without_api_key(tmp_path):
    train_path, feats = _train_csv(tmp_path)
    cfg = WorkflowConfig(
        agentic=AgenticImprovementConfig(
            enabled=True,
            max_cycles=1,
            max_total_experiments=2,
            acceptance=AgenticAcceptanceCriteria(
                minimum_mean_cv_r2=0.99,
                require_validation_agent_approval=False,
                require_non_overfit_status=False,
            ),
        )
    )
    loop = AgenticImprovementLoop(
        run_dir=tmp_path,
        workflow_config=cfg,
        hpo_config=HPOConfig(enabled=False, cv_folds=3),
        development_train_path=train_path,
        selected_features=feats[:2],
        dataset_hash="h",
        initial_estimator="Ridge",
        initial_final_selection=_final(),
        provider=None,  # deterministic fallback
    )
    state = loop.run()
    assert state.status == "model_locked"
    # Events should label deterministic fallback somewhere in workspace
    events = (tmp_path / "agent_workspace" / "agent_events.jsonl")
    assert events.exists() or (tmp_path / "agent_workspace" / "project_state.json").exists()


def test_agentic_disabled_default():
    assert WorkflowConfig().agentic.enabled is False
