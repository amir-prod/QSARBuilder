"""Tests for locked external eval hash checks and read-only post-test audit."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from qsar_agent.agentic.lock import configuration_hash, lock_model
from qsar_agent.agentic.ledger import agent_workspace, save_project_state
from qsar_agent.config import ModelConfig, WorkflowConfig
from qsar_agent.schemas.agentic import (
    AgenticImprovementConfig,
    AgenticProjectState,
    ExperimentRecord,
)
from qsar_agent.schemas.post_test_audit import PostTestAuditCriteria
from qsar_agent.services.model_lock_eval import (
    ConfigurationHashMismatchError,
    build_lock_config_snapshot,
    ensure_model_locked,
    evaluate_locked_winner_external,
    resolve_locked_eval_config,
    save_post_test_audit_criteria_snapshot,
    verify_lock_configuration_hash,
)
from qsar_agent.services.post_test_audit import run_post_test_audit


def _toy_matrices(tmp_path: Path, n_train: int = 40, n_test: int = 12):
    rng = np.random.RandomState(0)
    cols = ["f0", "f1", "f2"]
    Xtr = rng.randn(n_train, 3)
    ytr = Xtr[:, 0] + rng.randn(n_train) * 0.05
    train = pd.DataFrame(Xtr, columns=cols)
    train["activity"] = ytr
    train["compound_id"] = [f"tr{i}" for i in range(n_train)]
    train["canonical_smiles"] = ["CCO"] * n_train
    Xte = rng.randn(n_test, 3)
    yte = Xte[:, 0] + rng.randn(n_test) * 0.05
    test = pd.DataFrame(Xte, columns=cols)
    test["activity"] = yte
    test["compound_id"] = [f"te{i}" for i in range(n_test)]
    test["canonical_smiles"] = ["CCO"] * n_test
    train_path = tmp_path / "train.csv"
    test_path = tmp_path / "test.csv"
    train.to_csv(train_path, index=False)
    test.to_csv(test_path, index=False)
    return train_path, test_path, cols


def test_evaluate_uses_lock_record_config_and_writes_evaluated_config(tmp_path: Path):
    train_path, test_path, cols = _toy_matrices(tmp_path)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    cfg = WorkflowConfig(agentic=AgenticImprovementConfig(enabled=False))
    model_cfg = ModelConfig(estimator="Ridge", params={"alpha": 1.0}, random_state=0, n_jobs=1)
    state = ensure_model_locked(
        run_dir,
        workflow_config=cfg,
        dataset_hash="h",
        estimator="Ridge",
        selected_features=cols,
        final_model_config=model_cfg,
        selection_rationale="test lock",
        selection_record={
            "winning_estimator": "Ridge",
            "selected_features": cols,
            "final_model_config": model_cfg.model_dump(),
        },
    )
    save_post_test_audit_criteria_snapshot(run_dir, cfg.agentic.post_test_audit)
    state, modeling, ad = evaluate_locked_winner_external(
        run_dir,
        agentic_state=state,
        train_path=train_path,
        test_path=test_path,
        use_lock_record_config=True,
    )
    evaluated = json.loads(
        (run_dir / "locked_external" / "evaluated_config.json").read_text(encoding="utf-8")
    )
    assert evaluated["estimator"] == "Ridge"
    assert evaluated["selected_features"] == cols
    assert evaluated["lock_record_configuration_hash"] == state.lock_record.configuration_hash
    assert (run_dir / "locked_external" / "predictions.csv").exists()
    assert modeling is not None
    assert ad is not None


def test_evaluate_strips_expansion_display_label(tmp_path: Path):
    train_path, test_path, cols = _toy_matrices(tmp_path)
    run_dir = tmp_path / "run_label"
    run_dir.mkdir()
    cfg = WorkflowConfig(agentic=AgenticImprovementConfig(enabled=False))
    model_cfg = ModelConfig(
        estimator="PLSRegression",
        params={"n_components": 2, "scale": False, "max_iter": 500},
        random_state=0,
        n_jobs=1,
    )
    state = ensure_model_locked(
        run_dir,
        workflow_config=cfg,
        dataset_hash="h",
        estimator="PLSRegression (sfs_fixed_ga_plus2)",
        selected_features=cols,
        final_model_config=model_cfg,
        selection_rationale="expansion winner",
        selection_record={
            "winning_estimator": "PLSRegression (sfs_fixed_ga_plus2)",
            "selected_features": cols,
            "final_model_config": model_cfg.model_dump(),
            "winner_is_expansion": True,
            "winner_expansion_label": "sfs_fixed_ga_plus2",
        },
    )
    save_post_test_audit_criteria_snapshot(run_dir, cfg.agentic.post_test_audit)
    state, modeling, _ad = evaluate_locked_winner_external(
        run_dir,
        agentic_state=state,
        train_path=train_path,
        test_path=test_path,
        use_lock_record_config=True,
    )
    evaluated = json.loads(
        (run_dir / "locked_external" / "evaluated_config.json").read_text(encoding="utf-8")
    )
    assert evaluated["estimator"] == "PLSRegression"
    assert "(" not in evaluated["estimator"]
    assert modeling is not None


def test_hash_mismatch_fails_safely(tmp_path: Path):
    train_path, test_path, cols = _toy_matrices(tmp_path)
    run_dir = tmp_path / "run_bad"
    run_dir.mkdir()
    cfg = WorkflowConfig()
    model_cfg = ModelConfig(estimator="Ridge", params={"alpha": 1.0}, random_state=0, n_jobs=1)
    state = ensure_model_locked(
        run_dir,
        workflow_config=cfg,
        dataset_hash="h",
        estimator="Ridge",
        selected_features=cols,
        final_model_config=model_cfg,
        selection_rationale="test",
        selection_record={
            "winning_estimator": "Ridge",
            "selected_features": cols,
            "final_model_config": model_cfg.model_dump(),
        },
    )
    # Tamper lock hash
    bad = state.model_copy(
        update={
            "lock_record": state.lock_record.model_copy(
                update={"configuration_hash": "0" * 64}
            )
        }
    )
    save_project_state(run_dir, bad)
    with pytest.raises(ConfigurationHashMismatchError):
        evaluate_locked_winner_external(
            run_dir,
            agentic_state=bad,
            train_path=train_path,
            test_path=test_path,
            use_lock_record_config=True,
        )
    assert not (run_dir / "locked_external" / "predictions.csv").exists()


def test_criteria_snapshot_before_external(tmp_path: Path):
    run_dir = tmp_path / "run_crit"
    run_dir.mkdir()
    criteria = PostTestAuditCriteria(minimum_external_r2=0.42)
    path = save_post_test_audit_criteria_snapshot(run_dir, criteria)
    assert path.exists()
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["minimum_external_r2"] == 0.42
    # Snapshot lives under agent_workspace before unlock
    assert "agent_workspace" in str(path)


def test_audit_read_only_no_ledger_experiments(tmp_path: Path):
    train_path, test_path, cols = _toy_matrices(tmp_path, n_test=25)
    run_dir = tmp_path / "run_audit"
    run_dir.mkdir()
    cfg = WorkflowConfig(
        agentic=AgenticImprovementConfig(
            post_test_audit=PostTestAuditCriteria(
                minimum_external_r2=0.0,
                maximum_cv_test_r2_gap=1.0,
                minimum_ad_coverage=0.0,
                minimum_bootstrap_n=20,
            )
        )
    )
    model_cfg = ModelConfig(estimator="Ridge", params={"alpha": 1.0}, random_state=0, n_jobs=1)
    state = ensure_model_locked(
        run_dir,
        workflow_config=cfg,
        dataset_hash="h",
        estimator="Ridge",
        selected_features=cols,
        final_model_config=model_cfg,
        selection_rationale="test",
        selection_record={
            "winning_estimator": "Ridge",
            "selected_features": cols,
            "final_model_config": model_cfg.model_dump(),
        },
    )
    # Attach fake CV metrics via synthetic experiment in workspace
    ws = agent_workspace(run_dir)
    exp_dir = ws / "experiments" / "deterministic_winner"
    exp_dir.mkdir(parents=True)
    snap = build_lock_config_snapshot(
        estimator="Ridge", selected_features=cols, final_model_config=model_cfg
    )
    # Update lock to use experiment hash of snap (already does via ensure)
    save_post_test_audit_criteria_snapshot(run_dir, cfg.agentic.post_test_audit)
    state, _, _ = evaluate_locked_winner_external(
        run_dir,
        agentic_state=state,
        train_path=train_path,
        test_path=test_path,
        hpo_metadata={"mean_cv_r2": 0.8},
    )
    lock_hash_before = state.lock_record.configuration_hash
    ledger = ws / "experiment_ledger.jsonl"
    ledger_before = ledger.read_text(encoding="utf-8") if ledger.exists() else ""

    audit = run_post_test_audit(run_dir)
    assert (run_dir / "locked_external" / "post_test_audit.json").exists()
    assert (run_dir / "locked_external" / "post_test_audit.md").exists()
    assert audit.primary_outcome in (
        "external_validation_passed",
        "external_validation_failed",
    )
    # No new model artifact from audit
    assert not (run_dir / "locked_external" / "audit_model.joblib").exists()
    # Lock unchanged
    state2 = json.loads((ws / "project_state.json").read_text(encoding="utf-8"))
    assert state2["lock_record"]["configuration_hash"] == lock_hash_before
    ledger_after = ledger.read_text(encoding="utf-8") if ledger.exists() else ""
    assert ledger_after == ledger_before
    assert audit.random_vs_grouped_cv["status"] == "unavailable"


def test_bootstrap_cis_deterministic(tmp_path: Path):
    train_path, test_path, cols = _toy_matrices(tmp_path, n_test=25)
    run_dir = tmp_path / "run_boot"
    run_dir.mkdir()
    criteria = PostTestAuditCriteria(
        minimum_external_r2=0.0,
        maximum_cv_test_r2_gap=1.0,
        minimum_ad_coverage=0.0,
        bootstrap_samples=200,
        bootstrap_seed=42,
        minimum_bootstrap_n=20,
    )
    cfg = WorkflowConfig(agentic=AgenticImprovementConfig(post_test_audit=criteria))
    model_cfg = ModelConfig(estimator="Ridge", params={"alpha": 1.0}, random_state=0, n_jobs=1)
    state = ensure_model_locked(
        run_dir,
        workflow_config=cfg,
        dataset_hash="h",
        estimator="Ridge",
        selected_features=cols,
        final_model_config=model_cfg,
        selection_rationale="t",
        selection_record={
            "winning_estimator": "Ridge",
            "selected_features": cols,
            "final_model_config": model_cfg.model_dump(),
        },
    )
    save_post_test_audit_criteria_snapshot(run_dir, criteria)
    evaluate_locked_winner_external(
        run_dir,
        agentic_state=state,
        train_path=train_path,
        test_path=test_path,
    )
    a1 = run_post_test_audit(run_dir)
    a2 = run_post_test_audit(run_dir)
    assert a1.bootstrap_cis
    assert a1.bootstrap_cis[0].available
    assert a1.bootstrap_cis[0].lower == a2.bootstrap_cis[0].lower
    assert a1.bootstrap_cis[0].upper == a2.bootstrap_cis[0].upper


def test_ad_subgroup_warning_tiny_subset(tmp_path: Path):
    train_path, test_path, cols = _toy_matrices(tmp_path, n_test=8)
    run_dir = tmp_path / "run_ad"
    run_dir.mkdir()
    criteria = PostTestAuditCriteria(
        minimum_external_r2=0.0,
        maximum_cv_test_r2_gap=1.0,
        minimum_ad_coverage=0.0,
        minimum_bootstrap_n=100,  # force skip bootstrap
        minimum_subgroup_n=5,
    )
    cfg = WorkflowConfig(agentic=AgenticImprovementConfig(post_test_audit=criteria))
    model_cfg = ModelConfig(estimator="Ridge", params={"alpha": 1.0}, random_state=0, n_jobs=1)
    state = ensure_model_locked(
        run_dir,
        workflow_config=cfg,
        dataset_hash="h",
        estimator="Ridge",
        selected_features=cols,
        final_model_config=model_cfg,
        selection_rationale="t",
        selection_record={
            "winning_estimator": "Ridge",
            "selected_features": cols,
            "final_model_config": model_cfg.model_dump(),
        },
    )
    save_post_test_audit_criteria_snapshot(run_dir, criteria)
    evaluate_locked_winner_external(
        run_dir,
        agentic_state=state,
        train_path=train_path,
        test_path=test_path,
    )
    # Force a tiny out-of-domain subgroup
    ad_path = run_dir / "locked_external" / "applicability_domain.csv"
    ad = pd.read_csv(ad_path)
    test_idx = ad.index[ad["split"] == "test"]
    ad.loc[test_idx, "in_domain"] = True
    ad.loc[test_idx[:2], "in_domain"] = False
    ad.loc[test_idx[:2], "applicability_domain"] = "high_leverage"
    ad.to_csv(ad_path, index=False)
    # Mirror to run dir
    ad.to_csv(run_dir / "applicability_domain.csv", index=False)

    audit = run_post_test_audit(run_dir)
    assert audit.out_of_domain_metrics is not None
    assert audit.out_of_domain_metrics.n == 2
    assert audit.out_of_domain_metrics.reliable is False
    assert audit.out_of_domain_metrics.warning
    assert "small_external_test" in audit.diagnostic_flags


def test_outcome_flags_match_criteria(tmp_path: Path):
    train_path, test_path, cols = _toy_matrices(tmp_path, n_test=25)
    run_dir = tmp_path / "run_flags"
    run_dir.mkdir()
    # Impossible external R² threshold -> fail
    criteria = PostTestAuditCriteria(
        minimum_external_r2=0.999,
        maximum_cv_test_r2_gap=0.0,
        minimum_ad_coverage=0.0,
        minimum_bootstrap_n=20,
    )
    cfg = WorkflowConfig(agentic=AgenticImprovementConfig(post_test_audit=criteria))
    model_cfg = ModelConfig(estimator="Ridge", params={"alpha": 1.0}, random_state=0, n_jobs=1)
    state = ensure_model_locked(
        run_dir,
        workflow_config=cfg,
        dataset_hash="h",
        estimator="Ridge",
        selected_features=cols,
        final_model_config=model_cfg,
        selection_rationale="t",
        selection_record={
            "winning_estimator": "Ridge",
            "selected_features": cols,
            "final_model_config": model_cfg.model_dump(),
        },
    )
    save_post_test_audit_criteria_snapshot(run_dir, criteria)
    evaluate_locked_winner_external(
        run_dir,
        agentic_state=state,
        train_path=train_path,
        test_path=test_path,
        hpo_metadata={"mean_cv_r2": 0.99},
    )
    audit = run_post_test_audit(run_dir)
    assert audit.primary_outcome == "external_validation_failed"
    assert any("minimum_external_r2" in x for x in audit.failed_criteria)
    assert audit.remediation_allowed is True
    assert audit.recommendations


def test_resolve_locked_config_from_experiment_hash(tmp_path: Path):
    run_dir = tmp_path / "run_exp"
    run_dir.mkdir()
    cols = ["f0", "f1"]
    snap = {
        "action": "accept_model",
        "estimator": "SVR",
        "selected_features": cols,
        "configuration_changes": {"params": {"C": 2.0}},
        "final_model_config": {"estimator": "SVR", "params": {"C": 2.0}},
    }
    exp = ExperimentRecord(
        experiment_id="exp_002",
        hypothesis="h",
        action="accept_model",
        config_snapshot=snap,
        status="completed",
        configuration_hash=configuration_hash(snap),
        estimator="SVR",
        selected_features=cols,
        dataset_hash="d",
    )
    state = AgenticProjectState(
        project_id="p",
        initial_run_id="p",
        current_experiment_id="exp_002",
        best_experiment_id="exp_002",
        status="developing",
    )
    state = lock_model(state, exp, selection_rationale="best", selection_record={})
    # Persist experiment for get_experiment
    from qsar_agent.agentic.ledger import append_experiment_record

    append_experiment_record(run_dir, exp)
    save_project_state(run_dir, state)
    verify_lock_configuration_hash(run_dir, state)
    est, feats, model_cfg, _, _ = resolve_locked_eval_config(run_dir, state)
    assert est == "SVR"
    assert feats == cols
    assert model_cfg.estimator == "SVR"


def test_fork_helper_does_not_mutate_lock_on_audit_display(tmp_path: Path):
    """Audit + criteria load must not alter lock_record."""
    train_path, test_path, cols = _toy_matrices(tmp_path, n_test=25)
    run_dir = tmp_path / "run_lock_stable"
    run_dir.mkdir()
    criteria = PostTestAuditCriteria(minimum_external_r2=0.0, minimum_ad_coverage=0.0)
    cfg = WorkflowConfig(agentic=AgenticImprovementConfig(post_test_audit=criteria))
    model_cfg = ModelConfig(estimator="Ridge", params={"alpha": 1.0}, random_state=0, n_jobs=1)
    state = ensure_model_locked(
        run_dir,
        workflow_config=cfg,
        dataset_hash="h",
        estimator="Ridge",
        selected_features=cols,
        final_model_config=model_cfg,
        selection_rationale="t",
        selection_record={
            "winning_estimator": "Ridge",
            "selected_features": cols,
            "final_model_config": model_cfg.model_dump(),
        },
    )
    save_post_test_audit_criteria_snapshot(run_dir, criteria)
    state, _, _ = evaluate_locked_winner_external(
        run_dir,
        agentic_state=state,
        train_path=train_path,
        test_path=test_path,
    )
    before = json.loads(
        (run_dir / "agent_workspace" / "project_state.json").read_text(encoding="utf-8")
    )
    run_post_test_audit(run_dir)
    after = json.loads(
        (run_dir / "agent_workspace" / "project_state.json").read_text(encoding="utf-8")
    )
    assert before["lock_record"] == after["lock_record"]
