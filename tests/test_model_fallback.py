"""Tests for multi-model fallback selection."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from qsar_agent.config import ModelConfig
from qsar_agent.schemas.hyperparameter_optimization import FinalModelSelection
from qsar_agent.tools.hyperparameter_optimization import select_best_across_models
from qsar_agent.tools.overfitting_assessment import assess_overfitting
from tests.test_hyperparameter_optimization import _cv_summary


def _final_selection(mean_cv: float, mean_train: float, std: float = 0.04):
    summary = _cv_summary(mean_train, mean_cv, std)
    assessment = assess_overfitting(summary)
    return FinalModelSelection(
        source="baseline",
        params={},
        cv_summary=summary,
        assessment=assessment,
        selection_rationale="test",
    )


def test_select_best_across_models_prefers_acceptable():
    good = _final_selection(0.65, 0.70)
    poor = _final_selection(0.55, 0.90)
    poor_assessment = assess_overfitting(_cv_summary(0.90, 0.55, 0.05))
    poor = poor.model_copy(update={"assessment": poor_assessment})

    result = select_best_across_models(
        [
            {
                "estimator": "RandomForestRegressor",
                "selected_features": ["a", "b"],
                "final_selection": poor,
                "model_config": ModelConfig(),
            },
            {
                "estimator": "SVR",
                "selected_features": ["a", "c"],
                "final_selection": good,
                "model_config": ModelConfig(estimator="SVR", params={"C": 1.0}),
            },
        ]
    )
    assert result["winning_estimator"] == "SVR"


def test_model_fallback_skips_when_rf_acceptable(tmp_path):
    from qsar_agent.config import WorkflowConfig
    from qsar_agent.schemas.feature_selection import FeatureCountSelection, GAResult, SFSResult
    from qsar_agent.schemas.hyperparameter_optimization import HPOResult
    from qsar_agent.schemas.model_fallback import ModelBranchResult
    from qsar_agent.tools.model_fallback import run_model_fallback_if_needed

    good = _final_selection(0.65, 0.70)
    hpo_result = HPOResult(
        enabled=True,
        rounds_completed=0,
        max_rounds=3,
        final_selection=good,
        final_model_config=ModelConfig().model_dump(),
    )

    branch = ModelBranchResult(
        estimator="RandomForestRegressor",
        model_config_snapshot=hpo_result.final_model_config,
        sfs=SFSResult(
            results=[],
            max_features_evaluated=0,
            results_csv_path="",
            selected_features_json_path="",
            plot_png_path="",
            plot_svg_path="",
        ),
        feature_count=FeatureCountSelection(
            best_cv_r2=0.7,
            best_feature_count=2,
            selected_feature_count=2,
            selected_cv_r2=0.65,
            explanation="",
            selection_json_path="",
            explanation_md_path="",
        ),
        ga=GAResult(
            selected_features=["a"],
            best_fitness=0.7,
            history_csv_path="",
            selected_features_path="",
            configuration_path="",
            convergence_png_path="",
            convergence_svg_path="",
        ),
        hpo_result=hpo_result,
    )

    result = run_model_fallback_if_needed(
        branch,
        train_path="dummy.csv",
        run_dir=tmp_path,
        workflow_config=WorkflowConfig(),
        hpo_config=MagicMock(),
    )
    assert not result.triggered
    assert result.cross_model_selection.winning_estimator == "RandomForestRegressor"


def test_model_fallback_compares_sfs_subset_when_rf_ga_acceptable(tmp_path):
    from qsar_agent.config import WorkflowConfig
    from qsar_agent.schemas.feature_selection import FeatureCountSelection, GAResult, SFSResult
    from qsar_agent.schemas.hyperparameter_optimization import HPOResult
    from qsar_agent.schemas.model_fallback import ModelBranchResult
    from qsar_agent.tools.model_fallback import run_model_fallback_if_needed

    ga_sel = _final_selection(0.65, 0.70)
    sfs_sel = _final_selection(0.80, 0.85)
    hpo_result = HPOResult(
        enabled=True,
        rounds_completed=0,
        max_rounds=3,
        final_selection=ga_sel,
        final_model_config=ModelConfig().model_dump(),
    )
    branch = ModelBranchResult(
        estimator="RandomForestRegressor",
        model_config_snapshot=hpo_result.final_model_config,
        branch_dir=str(tmp_path),
        sfs=SFSResult(
            results=[],
            max_features_evaluated=0,
            results_csv_path="",
            selected_features_json_path="",
            plot_png_path="",
            plot_svg_path="",
        ),
        feature_count=FeatureCountSelection(
            best_cv_r2=0.8,
            best_feature_count=2,
            selected_feature_count=2,
            selected_cv_r2=0.8,
            explanation="",
            selection_json_path="",
            explanation_md_path="",
        ),
        ga=GAResult(
            selected_features=["ga_a"],
            best_fitness=0.65,
            history_csv_path="",
            selected_features_path="",
            configuration_path="",
            convergence_png_path="",
            convergence_svg_path="",
        ),
        hpo_result=hpo_result,
    )
    sfs_child = branch.model_copy(
        update={
            "is_expansion": True,
            "expansion_label": "sfs_subset",
            "branch_dir": str(tmp_path / "sfs_subset"),
            "ga": GAResult(
                selected_features=["sfs_a", "sfs_b"],
                best_fitness=0.8,
                history_csv_path="",
                selected_features_path="",
                configuration_path="",
                convergence_png_path="",
                convergence_svg_path="",
            ),
            "hpo_result": HPOResult(
                enabled=True,
                rounds_completed=0,
                max_rounds=3,
                final_selection=sfs_sel,
                final_model_config=ModelConfig().model_dump(),
            ),
        }
    )
    branch = branch.model_copy(update={"sfs_subset": sfs_child})

    result = run_model_fallback_if_needed(
        branch,
        train_path="dummy.csv",
        run_dir=tmp_path,
        workflow_config=WorkflowConfig(),
        hpo_config=MagicMock(),
    )
    assert not result.triggered
    assert result.cross_model_selection is not None
    assert result.cross_model_selection.winner_expansion_label == "sfs_subset"
    assert result.cross_model_selection.selected_features == ["sfs_a", "sfs_b"]
    assert len(result.cross_model_selection.compared_models) == 2
