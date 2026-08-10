"""Executor / screening / labeling tests for agentic experiments."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from qsar_agent.agentic.cv_folds import create_cv_folds, persist_cv_folds
from qsar_agent.agentic.executor import AgenticExperimentExecutor
from qsar_agent.agentic.lock import ExternalEvalLockError, assert_external_eval_allowed
from qsar_agent.agentic.protected_split import carve_agent_validation_split, persist_protected_split, subset_dataframe
from qsar_agent.config import WorkflowConfig
from qsar_agent.schemas.agentic import ExperimentProposal, ExperimentRecord
from qsar_agent.schemas.hyperparameter_optimization import HPOConfig, OverfittingThresholds
from qsar_agent.tools.hyperparameter_optimization import evaluate_baseline_model_cv
from qsar_agent.config import ModelConfig


def _tiny_xy(tmp_path: Path, n: int = 40, p: int = 6):
    rng = np.random.RandomState(0)
    X = rng.randn(n, p)
    y = X[:, 0] * 1.5 + rng.randn(n) * 0.1
    cols = [f"f{i}" for i in range(p)]
    df = pd.DataFrame(X, columns=cols)
    df["activity"] = y
    path = tmp_path / "train.csv"
    df.to_csv(path, index=False)
    return path, cols


def test_identical_cv_folds_across_screened_estimators(tmp_path):
    train_path, feats = _tiny_xy(tmp_path)
    folds = create_cv_folds(40, n_splits=5, random_seed=42)
    path, fold_hash = persist_cv_folds(tmp_path, folds)
    from qsar_agent.agentic.cv_folds import folds_as_sklearn_splits

    splits = folds_as_sklearn_splits(folds)
    r1 = evaluate_baseline_model_cv(
        train_path, feats[:3], ModelConfig(estimator="Ridge", params={"alpha": 1.0}, random_state=0),
        cv_splits=splits, run_dir=tmp_path / "ridge",
    )
    r2 = evaluate_baseline_model_cv(
        train_path, feats[:3], ModelConfig(estimator="ElasticNet", params={"alpha": 1.0, "l1_ratio": 0.5}, random_state=0),
        cv_splits=splits, run_dir=tmp_path / "enet",
    )
    assert len(r1.fold_metrics) == len(r2.fold_metrics) == 5
    # Fold identities are shared via splits; metrics differ by model
    assert fold_hash


def test_controlled_comparison_changes_only_estimator(tmp_path):
    train_path, feats = _tiny_xy(tmp_path)
    folds = create_cv_folds(40, n_splits=4, random_seed=7)
    cv_path, _ = persist_cv_folds(tmp_path, folds)
    parent = ExperimentRecord(
        experiment_id="exp_001",
        estimator="RandomForestRegressor",
        selected_features=feats[:4],
        feature_count=4,
        internal_metrics={"mean_cv_r2": 0.2, "train_cv_gap": 0.3, "cv_r2_std": 0.1, "mean_train_r2": 0.5},
        config_snapshot={},
    )
    hpo = HPOConfig(enabled=False, cv_folds=4, thresholds=OverfittingThresholds())
    ex = AgenticExperimentExecutor(
        run_dir=tmp_path,
        workflow_config=WorkflowConfig(random_seed=7),
        agent_dev_train_path=train_path,
        agent_val_path=None,
        selected_features=feats[:4],
        dataset_hash="x",
        cv_folds_path=cv_path,
        protected_val_indices_path=None,
        hpo_config=hpo,
    )
    proposal = ExperimentProposal(
        proposal_id="p1",
        parent_experiment_id="exp_001",
        proposed_by="modeling",
        hypothesis="try ridge",
        action="compare_registered_estimators",
        configuration_changes={"estimators": ["Ridge", "ElasticNet"], "optimize_top_k": 0},
        experiment_kind="controlled_estimator_comparison",
        multi_component=False,
        component_list=["estimator"],
    )
    rec = ex.execute(proposal, parent=parent, cycle_index=1)
    assert rec.experiment_kind == "controlled_estimator_comparison"
    assert rec.multi_component is False
    assert rec.component_list == ["estimator"]
    assert rec.selected_features == feats[:4]
    assert rec.internal_metrics.get("cv_folds_hash")
    screen = (tmp_path / "agent_workspace" / "experiments" / rec.experiment_id / "internal_results" / "controlled_estimator_screen.json")
    assert screen.exists()


def test_full_pipeline_branch_labeled_multi_component(tmp_path):
    train_path, feats = _tiny_xy(tmp_path, n=35, p=8)
    folds = create_cv_folds(35, n_splits=3, random_seed=3)
    cv_path, _ = persist_cv_folds(tmp_path, folds)
    parent = ExperimentRecord(
        experiment_id="exp_001",
        estimator="RandomForestRegressor",
        selected_features=feats[:3],
        feature_count=3,
        internal_metrics={"mean_cv_r2": 0.1, "train_cv_gap": 0.4, "cv_r2_std": 0.2, "mean_train_r2": 0.5},
    )
    hpo = HPOConfig(enabled=False, cv_folds=3, max_hpo_rounds=1)
    cfg = WorkflowConfig(random_seed=3)
    cfg.sfs.max_features = 3
    cfg.ga.n_generations = 2
    cfg.ga.population_size = 8
    ex = AgenticExperimentExecutor(
        run_dir=tmp_path,
        workflow_config=cfg,
        agent_dev_train_path=train_path,
        agent_val_path=None,
        selected_features=feats[:3],
        dataset_hash="x",
        cv_folds_path=cv_path,
        protected_val_indices_path=None,
        hpo_config=hpo,
    )
    proposal = ExperimentProposal(
        proposal_id="p2",
        parent_experiment_id="exp_001",
        proposed_by="modeling",
        hypothesis="full branch",
        action="try_registered_estimator",
        configuration_changes={"estimator": "Ridge", "mode": "full_pipeline", "run_hpo": False},
        experiment_kind="full_pipeline_branch",
        multi_component=True,
    )
    rec = ex.execute(proposal, parent=parent, cycle_index=1)
    assert rec.experiment_kind == "full_pipeline_branch"
    assert rec.multi_component is True
    assert "feature_selection" in rec.component_list or len(rec.component_list) >= 2


def test_adaptive_selection_uses_agent_dev_csv_not_protected(tmp_path):
    train_path, feats = _tiny_xy(tmp_path, n=50)
    df = pd.read_csv(train_path)
    dev_idx, val_idx, meta = carve_agent_validation_split(df, random_seed=2)
    paths = persist_protected_split(tmp_path, dev_idx, val_idx, meta)
    agent_dev = subset_dataframe(df, dev_idx)
    agent_val = subset_dataframe(df, val_idx)
    agent_dev_path = tmp_path / "agent_dev.csv"
    agent_val_path = tmp_path / "agent_val.csv"
    agent_dev.to_csv(agent_dev_path, index=False)
    agent_val.to_csv(agent_val_path, index=False)
    # Poison protected val activities — FS/HPO must not see them
    poisoned = agent_val.copy()
    poisoned["activity"] = 1e6
    poisoned.to_csv(agent_val_path, index=False)

    folds = create_cv_folds(len(agent_dev), n_splits=3, random_seed=2)
    cv_path, _ = persist_cv_folds(tmp_path, folds)
    parent = ExperimentRecord(
        experiment_id="exp_001",
        estimator="Ridge",
        selected_features=feats[:3],
        internal_metrics={"mean_cv_r2": 0.1, "train_cv_gap": 0.2, "cv_r2_std": 0.1, "mean_train_r2": 0.3},
    )
    ex = AgenticExperimentExecutor(
        run_dir=tmp_path,
        workflow_config=WorkflowConfig(random_seed=2),
        agent_dev_train_path=agent_dev_path,
        agent_val_path=agent_val_path,
        selected_features=feats[:3],
        dataset_hash="x",
        cv_folds_path=cv_path,
        protected_val_indices_path=paths["agent_val_indices_path"],
        hpo_config=HPOConfig(enabled=False, cv_folds=3),
    )
    proposal = ExperimentProposal(
        proposal_id="p3",
        parent_experiment_id="exp_001",
        proposed_by="modeling",
        hypothesis="controlled",
        action="try_registered_estimator",
        configuration_changes={"estimator": "ElasticNet", "mode": "controlled"},
    )
    rec = ex.execute(proposal, parent=parent, cycle_index=1)
    # Mean CV should be finite and not absurdly driven by 1e6 labels
    assert rec.internal_metrics["mean_cv_r2"] < 2.0
    assert rec.internal_metrics["mean_cv_r2"] > -5.0


def test_external_eval_gate_and_post_access(tmp_path):
    from qsar_agent.agentic.lock import lock_model, mark_external_evaluated, assert_agentic_optimization_allowed
    from qsar_agent.schemas.agentic import AgenticProjectState

    state = AgenticProjectState(
        project_id="r",
        initial_run_id="r",
        current_experiment_id="e",
        best_experiment_id="e",
        status="developing",
    )
    with pytest.raises(ExternalEvalLockError):
        assert_external_eval_allowed(state)
    rec = ExperimentRecord(experiment_id="e", configuration_hash="h", config_snapshot={"x": 1})
    state = lock_model(state, rec, selection_rationale="r", selection_record={"x": 1})
    assert_external_eval_allowed(state)
    state = mark_external_evaluated(state)
    with pytest.raises(Exception):
        assert_agentic_optimization_allowed(state)
