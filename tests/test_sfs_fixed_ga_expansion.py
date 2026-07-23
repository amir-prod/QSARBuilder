"""Tests for SFS-fixed GA expansion recovery."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from qsar_agent.config import GAConfig, ModelConfig, SFSFixedGAExpansionSettings, WorkflowConfig
from qsar_agent.schemas.feature_selection import FeatureCountSelection, GAResult, SFSResult, SFSResultRow
from qsar_agent.schemas.hyperparameter_optimization import FinalModelSelection, HPOResult
from qsar_agent.schemas.model_fallback import ModelBranchResult
from qsar_agent.tools.hyperparameter_optimization import select_best_across_models
from qsar_agent.tools.overfitting_assessment import assess_overfitting
from qsar_agent.tools.sfs_fixed_ga_expansion import run_sfs_fixed_ga_expansion
from tests.test_hyperparameter_optimization import _cv_summary


def _final_selection(mean_cv: float, mean_train: float, std: float = 0.04, source: str = "baseline"):
    summary = _cv_summary(mean_train, mean_cv, std)
    assessment = assess_overfitting(summary)
    return FinalModelSelection(
        source=source,
        params={},
        cv_summary=summary,
        assessment=assessment,
        selection_rationale="test",
    )


def _branch(*, acceptable: bool, k: int = 2) -> ModelBranchResult:
    if acceptable:
        fs = _final_selection(0.65, 0.70)
    else:
        # unstable / poor: high gap
        fs = _final_selection(0.40, 0.90, std=0.20)

    sfs_features = [f"f{i}" for i in range(1, k + 1)]
    return ModelBranchResult(
        estimator="RandomForestRegressor",
        model_config_snapshot=ModelConfig().model_dump(),
        branch_dir="",
        sfs=SFSResult(
            results=[
                SFSResultRow(
                    n_features=k,
                    mean_train_r2=0.8,
                    mean_cv_r2=0.6,
                    std_cv_r2=0.05,
                    selected_features=sfs_features,
                )
            ],
            max_features_evaluated=k,
            results_csv_path="",
            selected_features_json_path="",
            plot_png_path="",
            plot_svg_path="",
        ),
        feature_count=FeatureCountSelection(
            best_cv_r2=0.6,
            best_feature_count=k,
            selected_feature_count=k,
            selected_cv_r2=0.6,
            explanation="",
            selection_json_path="",
            explanation_md_path="",
        ),
        ga=GAResult(
            selected_features=sfs_features,
            best_fitness=0.6,
            history_csv_path="",
            selected_features_path="",
            configuration_path="",
            convergence_png_path="",
            convergence_svg_path="",
        ),
        hpo_result=HPOResult(
            enabled=True,
            rounds_completed=0,
            max_rounds=3,
            final_selection=fs,
            final_model_config=ModelConfig().model_dump(),
        ),
    )


def test_expansion_skipped_when_acceptable(tmp_path):
    branch = _branch(acceptable=True)
    branch = branch.model_copy(update={"branch_dir": str(tmp_path)})
    result = run_sfs_fixed_ga_expansion(
        branch,
        train_path=tmp_path / "missing.csv",
        run_dir=tmp_path,
        model_config=ModelConfig(),
        ga_config=GAConfig(population_size=5, n_generations=1),
        hpo_config=MagicMock(),
    )
    assert result is None
    assert not (tmp_path / "sfs_fixed_ga_plus2").exists()


def test_expansion_writes_subdir_and_summary(tmp_path):
    branch = _branch(acceptable=False, k=2)
    branch = branch.model_copy(update={"branch_dir": str(tmp_path)})

    fake_ga = GAResult(
        selected_features=["f1", "f2", "x1", "x2"],
        best_fitness=0.55,
        history_csv_path="",
        selected_features_path="",
        configuration_path="",
        convergence_png_path="",
        convergence_svg_path="",
    )
    better_fs = _final_selection(0.66, 0.72)
    fake_hpo = HPOResult(
        enabled=True,
        rounds_completed=1,
        max_rounds=3,
        final_selection=better_fs,
        final_model_config=ModelConfig().model_dump(),
    )

    with (
        patch(
            "qsar_agent.tools.sfs_fixed_ga_expansion._remaining_descriptor_count",
            return_value=10,
        ),
        patch(
            "qsar_agent.tools.sfs_fixed_ga_expansion.run_genetic_algorithm",
            return_value=fake_ga,
        ) as mock_ga,
        patch(
            "qsar_agent.tools.sfs_fixed_ga_expansion.run_iterative_hyperparameter_optimization",
            return_value=fake_hpo,
        ),
        patch("qsar_agent.tools.sfs_fixed_ga_expansion.pd.read_csv") as mock_csv,
    ):
        mock_csv.return_value = MagicMock(__len__=lambda self: 20)
        expansion = run_sfs_fixed_ga_expansion(
            branch,
            train_path=tmp_path / "train.csv",
            run_dir=tmp_path,
            model_config=ModelConfig(),
            ga_config=GAConfig(),
            hpo_config=MagicMock(),
            expansion_settings=SFSFixedGAExpansionSettings(extra_features=2),
        )

    assert expansion is not None
    assert expansion.is_expansion
    assert expansion.expansion_label == "sfs_fixed_ga_plus2"
    assert expansion.ga.selected_features == ["f1", "f2", "x1", "x2"]
    assert (tmp_path / "sfs_fixed_ga_plus2" / "expansion_summary.json").exists()
    mock_ga.assert_called_once()
    assert mock_ga.call_args[0][2] == 2
    assert mock_ga.call_args.kwargs.get("fixed_features") == ["f1", "f2"]


def test_select_best_prefers_expansion_when_better():
    poor = _final_selection(0.40, 0.90, std=0.20)
    good = _final_selection(0.65, 0.70)

    result = select_best_across_models(
        [
            {
                "estimator": "RandomForestRegressor",
                "base_estimator": "RandomForestRegressor",
                "selected_features": ["a", "b"],
                "final_selection": poor,
                "model_config": ModelConfig(),
                "is_expansion": False,
            },
            {
                "estimator": "RandomForestRegressor (sfs_fixed_ga_plus2)",
                "base_estimator": "RandomForestRegressor",
                "selected_features": ["a", "b", "c", "d"],
                "final_selection": good,
                "model_config": ModelConfig(),
                "is_expansion": True,
                "expansion_label": "sfs_fixed_ga_plus2",
            },
        ]
    )
    assert result["winner_is_expansion"] is True
    assert result["winner_expansion_label"] == "sfs_fixed_ga_plus2"
    assert "sfs_fixed_ga_plus2" in result["winning_estimator"]
    assert len(result["selected_features"]) == 4


def test_fallback_includes_expansion_candidate(tmp_path):
    from qsar_agent.tools.model_fallback import run_model_fallback_if_needed

    rf = _branch(acceptable=False, k=2)
    rf = rf.model_copy(update={"branch_dir": str(tmp_path)})

    expansion = rf.model_copy(
        update={
            "is_expansion": True,
            "expansion_label": "sfs_fixed_ga_plus2",
            "ga": GAResult(
                selected_features=["f1", "f2", "x1", "x2"],
                best_fitness=0.7,
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
                final_selection=_final_selection(0.68, 0.72),
                final_model_config=ModelConfig().model_dump(),
            ),
            "branch_dir": str(tmp_path / "sfs_fixed_ga_plus2"),
        }
    )
    rf = rf.model_copy(update={"expansion": expansion})

    cfg = WorkflowConfig(model_fallback={"enabled": False})
    result = run_model_fallback_if_needed(
        rf,
        train_path=tmp_path / "train.csv",
        run_dir=tmp_path,
        workflow_config=cfg,
        hpo_config=MagicMock(),
    )
    assert not result.triggered
    assert result.cross_model_selection is not None
    assert result.cross_model_selection.winner_is_expansion
    assert len(result.cross_model_selection.compared_models) == 2
