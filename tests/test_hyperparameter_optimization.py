"""Tests for hyperparameter optimization."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from qsar_agent.config import ModelConfig
from qsar_agent.schemas.hyperparameter_optimization import (
    AgentGridProposal,
    CVSummary,
    HPOConfig,
    OverfittingThresholds,
)
from qsar_agent.tools.hyperparameter_optimization import (
    count_grid_combinations,
    get_fallback_grid,
    run_hyperparameter_search,
    run_iterative_hyperparameter_optimization,
    sanitize_param_grid,
    select_final_model_config,
)
from qsar_agent.tools.overfitting_assessment import assess_overfitting


def _cv_summary(mean_train=0.9, mean_cv=0.5, std=0.05) -> CVSummary:
    return CVSummary(
        mean_train_r2=mean_train,
        mean_cv_r2=mean_cv,
        std_cv_r2=std,
        mean_train_rmse=0.1,
        mean_cv_rmse=0.2,
        mean_train_mae=0.1,
        mean_cv_mae=0.2,
        train_cv_r2_gap=mean_train - mean_cv,
        n_folds=3,
    )


@pytest.fixture
def tiny_train_csv(tmp_path):
    df = pd.DataFrame(
        {
            "compound_id": [f"C{i}" for i in range(12)],
            "canonical_smiles": ["CCO"] * 12,
            "activity": [float(i) for i in range(12)],
            "feat_a": [i * 0.1 for i in range(12)],
            "feat_b": [i * 0.2 for i in range(12)],
            "feat_c": [i * 0.05 for i in range(12)],
        }
    )
    path = tmp_path / "train.csv"
    df.to_csv(path, index=False)
    return path


def test_grid_sanitizer_removes_invalid_hyperparameters():
    grid = {
        "n_estimators": [100, 9999],
        "invalid_param": [1, 2],
        "max_depth": [5, 10],
    }
    result = sanitize_param_grid("RandomForestRegressor", grid, max_candidates=120)
    assert "invalid_param" not in result.sanitized_grid
    assert "n_estimators" in result.sanitized_grid
    assert 9999 not in result.sanitized_grid["n_estimators"]


def test_grid_sanitizer_respects_max_candidates():
    grid = {
        "n_estimators": [100, 200, 300, 400, 500],
        "max_depth": [3, 5, 8, 12, 20],
        "min_samples_split": [2, 4, 8],
    }
    result = sanitize_param_grid("RandomForestRegressor", grid, max_candidates=30)
    assert result.candidate_count <= 30


def test_grid_sanitizer_bootstrap_max_samples():
    grid = {
        "bootstrap": [False],
        "max_samples": [0.7, 0.9],
        "n_estimators": [100, 200],
        "max_depth": [5, 10],
    }
    result = sanitize_param_grid("RandomForestRegressor", grid, max_candidates=120)
    assert "max_samples" not in result.sanitized_grid


def test_grid_sanitizer_drops_max_samples_when_bootstrap_mixed():
    grid = {
        "bootstrap": [True, False],
        "max_samples": [0.7, 0.9],
        "n_estimators": [100, 200],
        "max_depth": [5, 10],
    }
    result = sanitize_param_grid("RandomForestRegressor", grid, max_candidates=120)
    assert "max_samples" not in result.sanitized_grid
    assert True in result.sanitized_grid["bootstrap"]
    assert False in result.sanitized_grid["bootstrap"]


def test_hpo_search_uses_training_data_only(tiny_train_csv, tmp_path):
    grid = {
        "n_estimators": [100, 200],
        "max_depth": [3, 5],
        "min_samples_leaf": [1, 2],
    }
    cfg = HPOConfig(cv_folds=3, max_candidates_per_round=20, n_jobs=1)
    candidates, best_params, summary, _ = run_hyperparameter_search(
        tiny_train_csv,
        ["feat_a", "feat_b"],
        grid,
        cfg,
        ModelConfig(n_estimators=100, max_depth=3, n_jobs=1),
        tmp_path,
        round_index=1,
    )
    assert candidates
    assert "n_estimators" in best_params
    assert summary.mean_cv_r2 <= 1.0


def test_hpo_controller_stops_early_when_baseline_acceptable(tiny_train_csv, tmp_path):
    good_summary = _cv_summary(0.72, 0.68, 0.04)
    assessment = assess_overfitting(good_summary)

    with patch(
        "qsar_agent.tools.hyperparameter_optimization.evaluate_baseline_model_cv"
    ) as mock_cv:
        from qsar_agent.schemas.hyperparameter_optimization import BaselineCVResult, FoldMetrics

        mock_cv.return_value = BaselineCVResult(
            fold_metrics=[
                FoldMetrics(
                    fold=1,
                    train_r2=0.72,
                    val_r2=0.68,
                    train_rmse=0.1,
                    val_rmse=0.2,
                    train_mae=0.1,
                    val_mae=0.2,
                )
            ],
            summary=good_summary,
            fold_metrics_path="",
            summary_path="",
        )
        with patch(
            "qsar_agent.tools.hyperparameter_optimization.assess_overfitting",
            return_value=assessment,
        ):
            result = run_iterative_hyperparameter_optimization(
                tiny_train_csv,
                ["feat_a", "feat_b"],
                ModelConfig(n_jobs=1),
                HPOConfig(cv_folds=3, n_jobs=1, thresholds=OverfittingThresholds()),
                tmp_path,
            )
    assert result.rounds_completed == 0
    assert result.final_selection.source == "baseline"


def test_hpo_controller_runs_rounds_when_overfit(tiny_train_csv, tmp_path):
    overfit_summary = _cv_summary(0.95, 0.45, 0.05)
    overfit_assessment = assess_overfitting(overfit_summary)
    good_summary = _cv_summary(0.70, 0.66, 0.04)
    good_assessment = assess_overfitting(good_summary)

    with patch(
        "qsar_agent.tools.hyperparameter_optimization.evaluate_baseline_model_cv"
    ) as mock_cv:
        from qsar_agent.schemas.hyperparameter_optimization import BaselineCVResult, FoldMetrics

        mock_cv.return_value = BaselineCVResult(
            fold_metrics=[
                FoldMetrics(
                    fold=1,
                    train_r2=0.95,
                    val_r2=0.45,
                    train_rmse=0.1,
                    val_rmse=0.2,
                    train_mae=0.1,
                    val_mae=0.2,
                )
            ],
            summary=overfit_summary,
            fold_metrics_path="",
            summary_path="",
        )
        assessments = [overfit_assessment, good_assessment]

        with patch(
            "qsar_agent.tools.hyperparameter_optimization.assess_overfitting",
            side_effect=assessments,
        ):
            with patch(
                "qsar_agent.tools.hyperparameter_optimization.run_hyperparameter_search"
            ) as mock_search:
                mock_search.return_value = (
                    [],
                    {"n_estimators": 100, "max_depth": 5},
                    good_summary,
                    sanitize_param_grid(
                        "RandomForestRegressor",
                        get_fallback_grid("RandomForestRegressor", "overfit"),
                        max_candidates=10,
                    ),
                )
                result = run_iterative_hyperparameter_optimization(
                    tiny_train_csv,
                    ["feat_a", "feat_b"],
                    ModelConfig(n_jobs=1),
                    HPOConfig(cv_folds=3, max_hpo_rounds=3, n_jobs=1),
                    tmp_path,
                    grid_proposer=None,
                )
    assert result.rounds_completed == 1
    log_text = (tmp_path / "hpo_iteration_log.md").read_text(encoding="utf-8")
    assert "HPO round 1/3" in log_text


def test_final_model_config_from_hpo_round():
    baseline_summary = _cv_summary(0.95, 0.45, 0.05)
    baseline_assessment = assess_overfitting(baseline_summary)
    from qsar_agent.schemas.hyperparameter_optimization import (
        GridSanitizationResult,
        HPORoundResult,
    )

    round_summary = _cv_summary(0.72, 0.68, 0.04)
    round_assessment = assess_overfitting(round_summary)
    rr = HPORoundResult(
        round_index=1,
        sanitization=GridSanitizationResult(
            original_grid={},
            sanitized_grid={},
            candidate_count=4,
        ),
        candidates=[],
        best_params={"n_estimators": 200, "max_depth": 5},
        best_cv_summary=round_summary,
        assessment=round_assessment,
        candidates_searched=4,
    )
    selection = select_final_model_config(
        baseline_summary,
        {"n_estimators": 100, "max_depth": 10},
        baseline_assessment,
        [rr],
        OverfittingThresholds(),
    )
    assert selection.source == "hpo_round_1"


def test_agent_invalid_grid_falls_back():
    from qsar_agent.agents.qsar_agent import propose_hyperparameter_grid
    from qsar_agent.tools.overfitting_assessment import assess_overfitting

    assessment = assess_overfitting(_cv_summary(0.95, 0.45, 0.05))
    with patch("qsar_agent.agents.qsar_agent.get_openai_api_key", return_value=None):
        proposal = propose_hyperparameter_grid(
            round_index=1,
            model_type="RandomForestRegressor",
            baseline_assessment=assessment,
            previous_hpo_results=[],
            constraints={"max_candidates": 50},
        )
    assert proposal.search_strategy == "fallback"
    assert proposal.proposed_grid


def test_missing_openai_key_allows_deterministic_hpo(tiny_train_csv, tmp_path):
    overfit_summary = _cv_summary(0.95, 0.40, 0.06)
    with patch(
        "qsar_agent.tools.hyperparameter_optimization.evaluate_baseline_model_cv"
    ) as mock_cv:
        from qsar_agent.schemas.hyperparameter_optimization import BaselineCVResult, FoldMetrics

        mock_cv.return_value = BaselineCVResult(
            fold_metrics=[
                FoldMetrics(
                    fold=1,
                    train_r2=0.95,
                    val_r2=0.40,
                    train_rmse=0.1,
                    val_rmse=0.2,
                    train_mae=0.1,
                    val_mae=0.2,
                )
            ],
            summary=overfit_summary,
            fold_metrics_path="",
            summary_path="",
        )
        result = run_iterative_hyperparameter_optimization(
            tiny_train_csv,
            ["feat_a", "feat_b"],
            ModelConfig(n_jobs=1),
            HPOConfig(
                cv_folds=3,
                max_hpo_rounds=1,
                max_candidates_per_round=8,
                n_jobs=1,
                thresholds=OverfittingThresholds(minimum_cv_r2=0.30),
            ),
            tmp_path,
            grid_proposer=None,
        )
    assert result.enabled
    assert (tmp_path / "hpo_iteration_log.json").exists()


def test_run_manifest_includes_hpo_metadata(tiny_train_csv, tmp_path):
    from qsar_agent.tools.final_model import train_and_evaluate_final_model

    test_df = tiny_train_csv.read_text(encoding="utf-8")
    test_path = tmp_path / "test.csv"
    test_path.write_text(test_df, encoding="utf-8")
    hpo_meta = {
        "enabled": True,
        "rounds_completed": 1,
        "final_model_source": "hpo_round_1",
        "final_params": {"n_estimators": 100},
    }
    result = train_and_evaluate_final_model(
        tiny_train_csv,
        test_path,
        tmp_path,
        ["feat_a", "feat_b"],
        ModelConfig(n_jobs=1),
        hpo_metadata=hpo_meta,
    )
    import json

    manifest = json.loads(Path(result.manifest_path).read_text(encoding="utf-8"))
    assert manifest["hyperparameter_optimization"]["final_model_source"] == "hpo_round_1"
