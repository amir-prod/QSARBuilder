"""Tests for per-branch external evaluation (scatter + Williams)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from qsar_agent.config import ModelConfig
from qsar_agent.schemas.feature_selection import FeatureCountSelection, GAResult, SFSResult
from qsar_agent.schemas.hyperparameter_optimization import FinalModelSelection, HPOResult
from qsar_agent.schemas.model_fallback import ModelBranchResult
from qsar_agent.tools.branch_external_evaluation import (
    evaluate_branch_on_external_test,
    evaluate_branches_on_external_test,
    find_winning_branch,
    flatten_branches,
    promote_branch_artifacts_to_run_dir,
)
from qsar_agent.tools.overfitting_assessment import assess_overfitting
from tests.test_hyperparameter_optimization import _cv_summary


def _final_selection(mean_cv: float = 0.6, mean_train: float = 0.7):
    summary = _cv_summary(mean_train, mean_cv, 0.04)
    assessment = assess_overfitting(summary)
    return FinalModelSelection(
        source="baseline",
        params={"n_estimators": 50, "max_depth": 4},
        cv_summary=summary,
        assessment=assessment,
        selection_rationale="test",
    )


def _make_train_test(tmp_path: Path, n: int = 20):
    rng = np.random.default_rng(0)
    rows = {
        "compound_id": [f"C{i}" for i in range(n)],
        "canonical_smiles": ["CCO"] * n,
        "activity": rng.normal(size=n),
        "original_row_index": list(range(n)),
        "feat_a": rng.normal(size=n),
        "feat_b": rng.normal(size=n),
    }
    df = pd.DataFrame(rows)
    train = df.iloc[:16].copy()
    test = df.iloc[16:].copy()
    train_path = tmp_path / "train.csv"
    test_path = tmp_path / "test.csv"
    train.to_csv(train_path, index=False)
    test.to_csv(test_path, index=False)
    return train_path, test_path


def _branch(tmp_path: Path, *, estimator: str = "RandomForestRegressor", subdir: str = "") -> ModelBranchResult:
    branch_dir = tmp_path / subdir if subdir else tmp_path
    branch_dir.mkdir(parents=True, exist_ok=True)
    cfg = ModelConfig(estimator=estimator, n_estimators=30, max_depth=3, n_jobs=1)
    return ModelBranchResult(
        estimator=estimator,
        model_config_snapshot=cfg.model_dump(),
        branch_dir=str(branch_dir),
        sfs=SFSResult(
            results=[],
            max_features_evaluated=0,
            results_csv_path="",
            selected_features_json_path="",
            plot_png_path="",
            plot_svg_path="",
        ),
        feature_count=FeatureCountSelection(
            best_cv_r2=0.6,
            best_feature_count=2,
            selected_feature_count=2,
            selected_cv_r2=0.55,
            explanation="",
            selection_json_path="",
            explanation_md_path="",
        ),
        ga=GAResult(
            selected_features=["feat_a", "feat_b"],
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
            final_selection=_final_selection(),
            final_model_config=cfg.model_dump(),
        ),
    )


def test_evaluate_branch_writes_scatter_and_williams(tmp_path):
    train_path, test_path = _make_train_test(tmp_path)
    branch = _branch(tmp_path / "rf_branch", subdir="")
    art, modeling, ad = evaluate_branch_on_external_test(
        branch,
        train_path=train_path,
        test_path=test_path,
    )
    assert Path(art.scatter_png_path).exists()
    assert Path(art.williams_png_path).exists()
    assert Path(art.predictions_path).exists()
    assert Path(modeling.scatter_png_path).exists()
    assert Path(ad.williams_png_path).exists()


def test_evaluate_two_branches_and_promote_winner(tmp_path):
    train_path, test_path = _make_train_test(tmp_path)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    rf = _branch(run_dir)
    svr_dir = run_dir / "fallback_models" / "svr"
    svr = _branch(svr_dir, estimator="SVR")
    svr = svr.model_copy(
        update={
            "model_config_snapshot": ModelConfig(
                estimator="SVR", params={"C": 1.0}, n_jobs=1
            ).model_dump()
        }
    )

    results = evaluate_branches_on_external_test(
        [rf, svr],
        train_path=train_path,
        test_path=test_path,
    )
    assert len(results) == 2
    assert (Path(rf.branch_dir) / "prediction_scatter.png").exists()
    assert (Path(svr.branch_dir) / "williams_plot.png").exists()

    promoted = promote_branch_artifacts_to_run_dir(svr.branch_dir, run_dir)
    assert (run_dir / "prediction_scatter.png").exists()
    assert (run_dir / "williams_plot.png").exists()
    assert "prediction_scatter.png" in promoted

    winner = find_winning_branch(
        flatten_branches(rf, svr),
        winning_estimator="SVR",
        selected_features=["feat_a", "feat_b"],
    )
    assert winner is not None
    assert Path(winner.branch_dir).resolve() == Path(svr.branch_dir).resolve()


def test_flatten_includes_expansion(tmp_path):
    parent = _branch(tmp_path / "parent")
    expansion = parent.model_copy(
        update={
            "is_expansion": True,
            "expansion_label": "sfs_fixed_ga_plus2",
            "branch_dir": str(tmp_path / "parent" / "sfs_fixed_ga_plus2"),
        }
    )
    Path(expansion.branch_dir).mkdir(parents=True, exist_ok=True)
    parent = parent.model_copy(update={"expansion": expansion})
    flat = flatten_branches(parent)
    assert len(flat) == 2
    assert any(b.is_expansion for b in flat)
