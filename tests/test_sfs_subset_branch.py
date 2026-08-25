"""Tests for SFS-subset competing branches."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from qsar_agent.config import ModelConfig, SFSSubsetBranchSettings
from qsar_agent.schemas.feature_selection import FeatureCountSelection, GAResult, SFSResult, SFSResultRow
from qsar_agent.schemas.hyperparameter_optimization import (
    BaselineCVResult,
    FinalModelSelection,
    HPOResult,
)
from qsar_agent.schemas.model_fallback import ModelBranchResult
from qsar_agent.tools.branch_external_evaluation import flatten_branches
from qsar_agent.tools.hyperparameter_optimization import select_best_across_models
from qsar_agent.tools.overfitting_assessment import assess_overfitting
from qsar_agent.tools.sfs_subset_branch import attach_sfs_subset_branches, run_sfs_subset_branch
from tests.test_hyperparameter_optimization import _cv_summary


def _final_selection(
    mean_cv: float,
    mean_train: float,
    std: float = 0.04,
    source: str = "baseline",
):
    summary = _cv_summary(mean_train, mean_cv, std)
    assessment = assess_overfitting(summary)
    return FinalModelSelection(
        source=source,
        params={"n_estimators": 100, "max_depth": 10},
        cv_summary=summary,
        assessment=assessment,
        selection_rationale="test",
    )


def _branch(*, ga_features: list[str], sfs_features: list[str], tmp_path: Path | None = None):
    k = len(sfs_features)
    fs = _final_selection(0.55, 0.80, std=0.20)
    return ModelBranchResult(
        estimator="RandomForestRegressor",
        model_config_snapshot=ModelConfig().model_dump(),
        branch_dir=str(tmp_path) if tmp_path is not None else "",
        sfs=SFSResult(
            results=[
                SFSResultRow(
                    n_features=k,
                    mean_train_r2=0.9,
                    mean_cv_r2=0.75,
                    std_cv_r2=0.05,
                    selected_features=sfs_features,
                    combined_r2=0.75,
                )
            ],
            max_features_evaluated=k,
            results_csv_path="",
            selected_features_json_path="",
            plot_png_path="",
            plot_svg_path="",
        ),
        feature_count=FeatureCountSelection(
            best_cv_r2=0.75,
            best_feature_count=k,
            selected_feature_count=k,
            selected_cv_r2=0.75,
            explanation="",
            selection_json_path="",
            explanation_md_path="",
        ),
        ga=GAResult(
            selected_features=ga_features,
            best_fitness=0.55,
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


def test_sfs_subset_skipped_when_features_match_ga(tmp_path):
    branch = _branch(ga_features=["a", "b"], sfs_features=["b", "a"], tmp_path=tmp_path)
    baseline, hpo = run_sfs_subset_branch(
        branch,
        train_path=tmp_path / "missing.csv",
        run_dir=tmp_path,
        model_config=ModelConfig(),
        hpo_config=MagicMock(enabled=True),
    )
    assert baseline is None
    assert hpo is None
    assert not (tmp_path / "sfs_subset").exists()


def test_sfs_subset_runs_hpo_and_labels_baseline(tmp_path):
    branch = _branch(
        ga_features=["ga1", "ga2"],
        sfs_features=["sfs1", "sfs2"],
        tmp_path=tmp_path,
    )
    fs = _final_selection(0.72, 0.78)
    fake_hpo = HPOResult(
        enabled=True,
        rounds_completed=0,
        max_rounds=3,
        baseline_cv=BaselineCVResult(
            fold_metrics=[],
            summary=fs.cv_summary,
            fold_metrics_path="",
            summary_path="",
        ),
        baseline_assessment=fs.assessment,
        final_selection=fs,
        final_model_config=ModelConfig().model_dump(),
    )
    with patch(
        "qsar_agent.tools.sfs_subset_branch.run_iterative_hyperparameter_optimization",
        return_value=fake_hpo,
    ), patch("qsar_agent.tools.sfs_subset_branch.pd.read_csv") as mock_csv:
        mock_csv.return_value = MagicMock(__len__=lambda self: 20)
        baseline, hpo = run_sfs_subset_branch(
            branch,
            train_path=tmp_path / "train.csv",
            run_dir=tmp_path,
            model_config=ModelConfig(),
            hpo_config=MagicMock(enabled=True),
        )

    assert baseline is not None
    assert hpo is None
    assert baseline.is_expansion
    assert baseline.expansion_label == "sfs_subset"
    assert baseline.ga.selected_features == ["sfs1", "sfs2"]
    assert (tmp_path / "sfs_subset" / "sfs_subset_summary.json").exists()
    assert (tmp_path / "sfs_subset" / "sfs_subset_features.json").exists()


def test_sfs_subset_emits_hpo_child_when_source_not_baseline(tmp_path):
    branch = _branch(
        ga_features=["ga1", "ga2"],
        sfs_features=["sfs1", "sfs2"],
        tmp_path=tmp_path,
    )
    baseline_summary = _cv_summary(0.90, 0.70, 0.04)
    baseline_assessment = assess_overfitting(baseline_summary)
    hpo_fs = _final_selection(0.66, 0.72, source="hpo_round_1")
    fake_hpo = HPOResult(
        enabled=True,
        rounds_completed=1,
        max_rounds=3,
        baseline_cv=BaselineCVResult(
            fold_metrics=[],
            summary=baseline_summary,
            fold_metrics_path="",
            summary_path="",
        ),
        baseline_assessment=baseline_assessment,
        final_selection=hpo_fs,
        final_model_config=ModelConfig(n_estimators=200).model_dump(),
    )
    with patch(
        "qsar_agent.tools.sfs_subset_branch.run_iterative_hyperparameter_optimization",
        return_value=fake_hpo,
    ), patch("qsar_agent.tools.sfs_subset_branch.pd.read_csv") as mock_csv:
        mock_csv.return_value = MagicMock(__len__=lambda self: 20)
        baseline, hpo = run_sfs_subset_branch(
            branch,
            train_path=tmp_path / "train.csv",
            run_dir=tmp_path,
            model_config=ModelConfig(),
            hpo_config=MagicMock(enabled=True),
            settings=SFSSubsetBranchSettings(),
        )

    assert baseline is not None
    assert hpo is not None
    assert baseline.expansion_label == "sfs_subset"
    assert hpo.expansion_label == "sfs_subset_hpo"
    assert hpo.hpo_result.final_selection.source == "hpo_round_1"
    assert baseline.hpo_result.final_selection.source == "baseline"
    assert Path(hpo.branch_dir).name == "sfs_subset_hpo"


def test_select_best_prefers_stronger_sfs_subset():
    poor = _final_selection(0.40, 0.90, std=0.20)
    good = _final_selection(0.68, 0.74)
    result = select_best_across_models(
        [
            {
                "estimator": "RandomForestRegressor",
                "base_estimator": "RandomForestRegressor",
                "selected_features": ["ga1", "ga2"],
                "final_selection": poor,
                "model_config": ModelConfig(),
                "is_expansion": False,
            },
            {
                "estimator": "RandomForestRegressor (sfs_subset)",
                "base_estimator": "RandomForestRegressor",
                "selected_features": ["sfs1", "sfs2"],
                "final_selection": good,
                "model_config": ModelConfig(),
                "is_expansion": True,
                "expansion_label": "sfs_subset",
            },
        ]
    )
    assert result["winner_is_expansion"] is True
    assert result["winner_expansion_label"] == "sfs_subset"
    assert result["selected_features"] == ["sfs1", "sfs2"]


def test_flatten_includes_sfs_subset_children(tmp_path):
    parent = _branch(
        ga_features=["ga1"],
        sfs_features=["sfs1"],
        tmp_path=tmp_path / "parent",
    )
    (tmp_path / "parent").mkdir(parents=True, exist_ok=True)
    sfs_dir = tmp_path / "parent" / "sfs_subset"
    hpo_dir = tmp_path / "parent" / "sfs_subset_hpo"
    sfs_dir.mkdir(parents=True)
    hpo_dir.mkdir(parents=True)
    subset = parent.model_copy(
        update={
            "is_expansion": True,
            "expansion_label": "sfs_subset",
            "branch_dir": str(sfs_dir),
            "hpo_result": HPOResult(
                enabled=True,
                rounds_completed=0,
                max_rounds=3,
                final_selection=_final_selection(0.70, 0.75),
                final_model_config=ModelConfig().model_dump(),
            ),
        }
    )
    subset_hpo = parent.model_copy(
        update={
            "is_expansion": True,
            "expansion_label": "sfs_subset_hpo",
            "branch_dir": str(hpo_dir),
            "hpo_result": HPOResult(
                enabled=True,
                rounds_completed=1,
                max_rounds=3,
                final_selection=_final_selection(0.66, 0.72, source="hpo_round_1"),
                final_model_config=ModelConfig().model_dump(),
            ),
        }
    )
    parent = parent.model_copy(update={"sfs_subset": subset, "sfs_subset_hpo": subset_hpo})
    flat = flatten_branches(parent)
    labels = {b.expansion_label for b in flat if b.is_expansion}
    assert "sfs_subset" in labels
    assert "sfs_subset_hpo" in labels
    assert len(flat) == 3


def test_attach_sfs_subset_updates_parent(tmp_path):
    branch = _branch(
        ga_features=["ga1", "ga2"],
        sfs_features=["sfs1", "sfs2"],
        tmp_path=tmp_path,
    )
    fs = _final_selection(0.72, 0.78)
    fake_hpo = HPOResult(
        enabled=True,
        rounds_completed=0,
        max_rounds=3,
        baseline_cv=BaselineCVResult(
            fold_metrics=[],
            summary=fs.cv_summary,
            fold_metrics_path="",
            summary_path="",
        ),
        baseline_assessment=fs.assessment,
        final_selection=fs,
        final_model_config=ModelConfig().model_dump(),
    )
    with patch(
        "qsar_agent.tools.sfs_subset_branch.run_iterative_hyperparameter_optimization",
        return_value=fake_hpo,
    ), patch("qsar_agent.tools.sfs_subset_branch.pd.read_csv") as mock_csv:
        mock_csv.return_value = MagicMock(__len__=lambda self: 20)
        updated = attach_sfs_subset_branches(
            branch,
            train_path=tmp_path / "train.csv",
            run_dir=tmp_path,
            model_config=ModelConfig(),
            hpo_config=MagicMock(enabled=True),
        )
    assert updated.sfs_subset is not None
    assert updated.sfs_subset_hpo is None
    assert updated.sfs_subset.ga.selected_features == ["sfs1", "sfs2"]
