"""Unit tests for agentic improvement core (acceptance, lock, folds, veto, selection)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from qsar_agent.agentic.acceptance import evaluate_acceptance
from qsar_agent.agentic.actions import get_action_spec, validate_action_params
from qsar_agent.agentic.artifact_view import AgentArtifactView, assert_no_external_test_fields
from qsar_agent.agentic.cv_folds import create_cv_folds, hash_cv_folds, persist_cv_folds, load_cv_folds
from qsar_agent.agentic.duplicate import duplicate_check_key, find_duplicate
from qsar_agent.agentic.hard_failures import build_validation_review, confirm_hard_failures, flags_from_runtime
from qsar_agent.agentic.lock import (
    AgenticOptimizationForbiddenError,
    ExternalEvalLockError,
    assert_agentic_optimization_allowed,
    assert_external_eval_allowed,
    lock_model,
    mark_external_evaluated,
)
from qsar_agent.agentic.protected_split import (
    assert_no_protected_targets_in_training,
    carve_agent_validation_split,
)
from qsar_agent.agentic.routing import infer_failure_signals, select_specialists
from qsar_agent.agentic.selection import select_best_experiment
from qsar_agent.schemas.agentic import (
    AgenticAcceptanceCriteria,
    AgenticProjectState,
    AgentVisibleSummary,
    ExperimentRecord,
)


def test_acceptance_calculation_fails_low_cv():
    result = evaluate_acceptance(
        {
            "mean_cv_r2": 0.4,
            "mean_train_r2": 0.5,
            "train_cv_gap": 0.1,
            "cv_r2_std": 0.05,
        },
        AgenticAcceptanceCriteria(require_non_overfit_status=False),
    )
    assert not result.accepted
    assert any("mean_cv_r2" in f for f in result.failed_criteria)


def test_acceptance_passes_good_metrics():
    result = evaluate_acceptance(
        {
            "mean_cv_r2": 0.7,
            "mean_train_r2": 0.75,
            "train_cv_gap": 0.05,
            "cv_r2_std": 0.04,
            "n_folds": 5,
        },
        AgenticAcceptanceCriteria(
            minimum_mean_cv_r2=0.60,
            require_non_overfit_status=True,
            require_validation_agent_approval=False,
        ),
    )
    assert result.accepted


def test_action_allowlist_and_invalid_rejection():
    assert get_action_spec("try_registered_estimator").executable
    with pytest.raises(ValueError):
        get_action_spec("run_arbitrary_python")
    with pytest.raises(Exception):
        validate_action_params("try_registered_estimator", {"estimator": "sklearn.ensemble.RF"})


def test_parameter_bounds_enforced():
    with pytest.raises(Exception):
        validate_action_params("refine_hyperparameters", {"max_candidates": 999})
    ok = validate_action_params("compare_registered_estimators", {"estimators": ["Ridge", "SVR"]})
    assert len(ok.estimators) == 2


def test_duplicate_experiment_detection():
    key = duplicate_check_key("try_registered_estimator", {"estimator": "Ridge", "mode": "controlled"})
    rec = ExperimentRecord(
        experiment_id="exp_001",
        action="try_registered_estimator",
        config_snapshot={
            "duplicate_check_key": key,
            "configuration_changes": {"estimator": "Ridge", "mode": "controlled"},
        },
    )
    assert find_duplicate([rec], "try_registered_estimator", {"estimator": "Ridge", "mode": "controlled"})


def test_best_experiment_selection_and_practical_equivalence():
    # Higher CV R² but more complex; within tolerance prefer fewer features.
    a = ExperimentRecord(
        experiment_id="exp_a",
        internal_metrics={"mean_cv_r2": 0.70, "train_cv_gap": 0.10, "cv_r2_std": 0.05, "accepted": True},
        feature_count=4,
        status="completed",
    )
    b = ExperimentRecord(
        experiment_id="exp_b",
        internal_metrics={"mean_cv_r2": 0.705, "train_cv_gap": 0.11, "cv_r2_std": 0.06, "accepted": True},
        feature_count=10,
        status="completed",
    )
    best, rationale = select_best_experiment([a, b], practical_equivalence_tolerance=0.01)
    assert best is not None
    assert best.experiment_id == "exp_a"
    assert "practical equivalence" in rationale.lower() or "simpler" in rationale.lower()


def test_cv_folds_identical_hash(tmp_path):
    folds = create_cv_folds(20, n_splits=5, random_seed=42)
    path, h1 = persist_cv_folds(tmp_path, folds)
    folds2, h2 = load_cv_folds(path)
    assert h1 == h2 == hash_cv_folds(folds2)
    assert folds2 == folds


def test_protected_split_no_overlap_in_training():
    df = pd.DataFrame({"activity": np.arange(40.0)})
    dev, val, meta = carve_agent_validation_split(df, agent_validation_fraction=0.2, random_seed=1)
    assert len(set(dev) & set(val)) == 0
    assert meta["protected_validation_available"]
    with pytest.raises(RuntimeError):
        assert_no_protected_targets_in_training(dev.tolist() + val.tolist()[:1], val, context="test")


def test_validation_hard_veto_requires_deterministic_flag():
    confirmed = confirm_hard_failures(
        ["acceptance_criteria_failed", "external_test_access_attempted"],
        deterministic_flags={"acceptance_criteria_failed": True},
    )
    assert confirmed == ["acceptance_criteria_failed"]
    review = build_validation_review(
        acceptance_passed=True,
        deterministic_flags=flags_from_runtime(),
        soft_rejection_recommended=True,
    )
    assert not review.hard_veto
    assert review.soft_rejection_recommended
    assert review.approved


def test_external_eval_fails_before_lock():
    state = AgenticProjectState(
        project_id="p",
        initial_run_id="p",
        current_experiment_id="e",
        best_experiment_id="e",
        status="developing",
    )
    with pytest.raises(ExternalEvalLockError):
        assert_external_eval_allowed(state)


def test_agentic_forbidden_after_external_access():
    rec = ExperimentRecord(
        experiment_id="e1",
        config_snapshot={"a": 1},
        configuration_hash="abc",
        dataset_hash="d",
    )
    state = AgenticProjectState(
        project_id="p",
        initial_run_id="p",
        current_experiment_id="e1",
        best_experiment_id="e1",
        status="developing",
    )
    state = lock_model(state, rec, selection_rationale="test", selection_record={"ok": True})
    state = mark_external_evaluated(state)
    with pytest.raises(AgenticOptimizationForbiddenError):
        assert_agentic_optimization_allowed(state)


def test_agent_payload_denies_external_fields():
    summary = AgentVisibleSummary(experiment_id="e1")
    view = AgentArtifactView(experiment_id="e1", summary=summary)
    payload = view.to_agent_payload()
    assert_no_external_test_fields(payload)
    with pytest.raises(ValueError):
        assert_no_external_test_fields({"test_r2": 0.9})


def test_specialist_routing_max_two():
    signals = infer_failure_signals(
        {"overfitting_status": "overfit", "samples_per_feature_ratio": 2.0, "agent_dev_size": 20}
    )
    specs = select_specialists(signals, max_specialists=2)
    assert len(specs) <= 2


def test_legacy_action_alias():
    from qsar_agent.schemas.agentic import normalize_action

    assert normalize_action("try_fallback_estimator") == "try_registered_estimator"
