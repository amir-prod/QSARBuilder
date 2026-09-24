"""Tests for the specialist agents: leakage-free preprocessing and CV scoring."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from qsar_agent.agents.roles import (
    DataAgent,
    DescriptorAgent,
    ModelingAgent,
    ValidationAgent,
    encode_targets,
)
from qsar_agent.schemas.agentic import FeatureRecipe, IterationPlan, ModelPlan

EXAMPLE_DIR = Path(__file__).resolve().parent.parent / "example"
REGRESSION_CSV = EXAMPLE_DIR / "agentic_regression_sample.csv"
CLASSIFICATION_CSV = EXAMPLE_DIR / "agentic_classification_sample.csv"


@pytest.fixture
def regression_bundle(tmp_path):
    return DataAgent(tmp_path, random_seed=42).prepare(
        dataset_path=REGRESSION_CSV,
        smiles_column="smiles",
        activity_column="pIC50",
        id_column="compound_id",
        test_size=0.2,
    )


@pytest.fixture
def classification_bundle(tmp_path):
    return DataAgent(tmp_path, random_seed=42).prepare(
        dataset_path=CLASSIFICATION_CSV,
        smiles_column="smiles",
        activity_column="active",
        id_column="compound_id",
        split_method="stratified",
        test_size=0.2,
    )


# -- DataAgent -----------------------------------------------------------------


def test_the_split_is_disjoint_and_complete(regression_bundle):
    train = set(regression_bundle.train_idx.tolist())
    test = set(regression_bundle.test_idx.tolist())
    assert train.isdisjoint(test)
    assert len(train | test) == len(regression_bundle.frame)


def test_the_task_is_detected(regression_bundle, classification_bundle):
    assert regression_bundle.task == "regression"
    assert classification_bundle.task == "classification"


def test_split_assignments_are_written(tmp_path, regression_bundle):
    frame = pd.read_csv(tmp_path / "agentic_split_assignments.csv")
    assert set(frame["split"]) == {"train", "test"}
    assert len(frame) == len(regression_bundle.frame)


def test_the_summary_describes_a_regression_endpoint(regression_bundle):
    summary = regression_bundle.summary()
    assert summary["task"] == "regression"
    assert "activity_stats" in summary
    assert summary["n_train"] + summary["n_test"] == summary["n_compounds"]


def test_the_summary_describes_class_balance(classification_bundle):
    summary = classification_bundle.summary()
    assert "class_counts" in summary
    assert len(summary["class_counts"]) == 2


def test_compound_ids_track_indices(regression_bundle):
    ids = regression_bundle.compound_ids(regression_bundle.test_idx)
    assert len(ids) == regression_bundle.n_test
    assert all(isinstance(value, str) for value in ids)


# -- DescriptorAgent -----------------------------------------------------------


def test_matrices_match_the_split_sizes(regression_bundle):
    matrices = DescriptorAgent(regression_bundle).build(
        FeatureRecipe(blocks=["rdkit_descriptors"])
    )
    assert matrices.X_train.shape[0] == regression_bundle.n_train
    assert matrices.X_test.shape[0] == regression_bundle.n_test
    assert matrices.X_train.shape[1] == matrices.X_test.shape[1]
    assert len(matrices.feature_names) == matrices.X_train.shape[1]


def test_scaling_is_fitted_on_training_rows_only(regression_bundle):
    matrices = DescriptorAgent(regression_bundle).build(
        FeatureRecipe(blocks=["rdkit_descriptors"], scale_features=True)
    )
    # Training columns are standardised; test columns generally are not.
    assert np.allclose(matrices.X_train.mean(axis=0), 0.0, atol=1e-6)
    assert not np.allclose(matrices.X_test.mean(axis=0), 0.0, atol=1e-6)


def test_disabling_scaling_leaves_raw_values(regression_bundle):
    matrices = DescriptorAgent(regression_bundle).build(
        FeatureRecipe(blocks=["rdkit_descriptors"], scale_features=False)
    )
    assert not np.allclose(matrices.X_train.mean(axis=0), 0.0, atol=1e-6)


def test_a_tighter_correlation_filter_keeps_fewer_features(regression_bundle):
    agent = DescriptorAgent(regression_bundle)
    loose = agent.build(FeatureRecipe(blocks=["rdkit_descriptors"], correlation_threshold=0.99))
    tight = agent.build(FeatureRecipe(blocks=["rdkit_descriptors"], correlation_threshold=0.70))
    assert len(tight.feature_names) < len(loose.feature_names)


def test_no_correlation_filter_keeps_the_most_features(regression_bundle):
    agent = DescriptorAgent(regression_bundle)
    filtered = agent.build(
        FeatureRecipe(blocks=["rdkit_descriptors"], correlation_threshold=0.95)
    )
    unfiltered = agent.build(
        FeatureRecipe(blocks=["rdkit_descriptors"], correlation_threshold=None)
    )
    assert len(unfiltered.feature_names) >= len(filtered.feature_names)


def test_univariate_selection_caps_the_feature_count(regression_bundle):
    y = encode_targets("regression", regression_bundle.frame["activity"])[0]
    matrices = DescriptorAgent(regression_bundle).build(
        FeatureRecipe(blocks=["rdkit_descriptors"], univariate_top_k=15),
        y_train_encoded=y[regression_bundle.train_idx],
    )
    assert len(matrices.feature_names) == 15


def test_univariate_selection_is_skipped_without_targets(regression_bundle):
    matrices = DescriptorAgent(regression_bundle).build(
        FeatureRecipe(blocks=["rdkit_descriptors"], univariate_top_k=15)
    )
    assert len(matrices.feature_names) > 15


def test_removing_every_feature_is_an_error(regression_bundle):
    with pytest.raises(ValueError, match="standard deviation"):
        DescriptorAgent(regression_bundle).build(
            FeatureRecipe(blocks=["rdkit_descriptors"], variance_threshold=1e12)
        )


def test_dropping_domain_outliers_shrinks_the_training_set(regression_bundle):
    agent = DescriptorAgent(regression_bundle)
    kept = agent.build(FeatureRecipe(blocks=["rdkit_descriptors"], univariate_top_k=20))
    dropped = agent.build(
        FeatureRecipe(
            blocks=["rdkit_descriptors"], univariate_top_k=20, drop_ad_outliers=True
        ),
        y_train_encoded=encode_targets("regression", regression_bundle.frame["activity"])[0][
            regression_bundle.train_idx
        ],
    )
    assert dropped.X_train.shape[0] <= kept.X_train.shape[0]
    assert dropped.train_idx.shape[0] == dropped.X_train.shape[0]


def test_preprocessing_notes_are_recorded(regression_bundle):
    matrices = DescriptorAgent(regression_bundle).build(
        FeatureRecipe(blocks=["rdkit_descriptors"], correlation_threshold=0.80)
    )
    assert any("correlated" in note for note in matrices.notes)


def test_raw_features_are_cached_across_recipes(regression_bundle):
    agent = DescriptorAgent(regression_bundle)
    agent.build(FeatureRecipe(blocks=["rdkit_descriptors"], correlation_threshold=0.95))
    agent.build(FeatureRecipe(blocks=["rdkit_descriptors"], correlation_threshold=0.80))
    assert len(agent._raw_cache) == 1


# -- ModelingAgent -------------------------------------------------------------


def test_cross_validation_reports_both_halves_of_each_fold(regression_bundle):
    matrices = DescriptorAgent(regression_bundle).build(
        FeatureRecipe(blocks=["rdkit_descriptors"], univariate_top_k=20)
    )
    y = encode_targets("regression", regression_bundle.frame["activity"])[0]
    agent = ModelingAgent("regression", cv_folds=3)
    plan = IterationPlan(
        features=FeatureRecipe(blocks=["rdkit_descriptors"]),
        model=ModelPlan(estimator="RandomForest", hyperparameters={"n_estimators": 20}),
    )
    summary = agent.cross_validate(agent.build(plan), matrices.X_train, y[matrices.train_idx])
    assert summary.folds == 3
    assert len(summary.fold_scores) == 3
    assert summary.train_cv_gap == pytest.approx(
        summary.mean_train_score - summary.mean_cv_score
    )
    # A forest fits its training folds better than held-out folds.
    assert summary.mean_train_score >= summary.mean_cv_score


def test_fold_count_is_capped_by_the_smallest_class():
    agent = ModelingAgent("classification", cv_folds=10)
    y = np.array([0] * 20 + [1] * 3)
    assert agent.effective_folds(y) == 3


def test_fold_count_never_drops_below_two():
    agent = ModelingAgent("classification", cv_folds=5)
    assert agent.effective_folds(np.array([0] * 10 + [1])) == 2


def test_regression_folds_are_capped_by_sample_count():
    agent = ModelingAgent("regression", cv_folds=10)
    assert agent.effective_folds(np.arange(4.0)) == 4


def test_the_splitter_is_stratified_for_classification():
    from sklearn.model_selection import StratifiedKFold

    agent = ModelingAgent("classification", cv_folds=3)
    assert isinstance(agent.splitter(np.array([0] * 10 + [1] * 10)), StratifiedKFold)


# -- ValidationAgent -----------------------------------------------------------


def test_validation_measures_every_gate(regression_bundle):
    matrices = DescriptorAgent(regression_bundle).build(
        FeatureRecipe(blocks=["rdkit_descriptors"], univariate_top_k=20)
    )
    y = encode_targets("regression", regression_bundle.frame["activity"])[0]
    modeling = ModelingAgent("regression", cv_folds=3)
    plan = IterationPlan(
        features=FeatureRecipe(blocks=["rdkit_descriptors"]),
        model=ModelPlan(estimator="RandomForest", hyperparameters={"n_estimators": 30}),
    )
    validator = ValidationAgent("regression", scramble_repeats=3)
    report, fitted = validator.validate(
        modeling_agent=modeling,
        estimator=modeling.build(plan),
        matrices=matrices,
        y_train=y[matrices.train_idx],
        y_test=y[matrices.test_idx],
        test_ids=regression_bundle.compound_ids(matrices.test_idx),
    )
    assert report.primary_metric == "r2"
    assert {"r2", "rmse", "mae"} <= set(report.test_metrics)
    assert report.y_scrambling is not None
    assert report.y_scrambling.n_repeats == 3
    assert report.applicability_domain is not None
    assert report.n_features == len(matrices.feature_names)
    assert hasattr(fitted, "predict")


def test_classification_validation_reports_class_metrics(classification_bundle):
    matrices = DescriptorAgent(classification_bundle).build(
        FeatureRecipe(blocks=["rdkit_descriptors"], univariate_top_k=20)
    )
    y, encoder = encode_targets("classification", classification_bundle.frame["activity"])
    assert encoder is not None
    modeling = ModelingAgent("classification", cv_folds=3)
    plan = IterationPlan(
        features=FeatureRecipe(blocks=["rdkit_descriptors"]),
        model=ModelPlan(estimator="RandomForest", hyperparameters={"n_estimators": 30}),
    )
    validator = ValidationAgent("classification", scramble_repeats=3)
    report, _fitted = validator.validate(
        modeling_agent=modeling,
        estimator=modeling.build(plan),
        matrices=matrices,
        y_train=y[matrices.train_idx],
        y_test=y[matrices.test_idx],
        test_ids=classification_bundle.compound_ids(matrices.test_idx),
    )
    assert report.primary_metric == "roc_auc"
    assert {"roc_auc", "mcc", "balanced_accuracy"} <= set(report.test_metrics)


# -- target encoding -----------------------------------------------------------


def test_regression_targets_stay_numeric():
    y, encoder = encode_targets("regression", pd.Series([1.5, 2.5, 3.5]))
    assert encoder is None
    assert y.dtype == float


def test_classification_labels_round_trip_through_the_encoder():
    y, encoder = encode_targets("classification", pd.Series(["active", "inactive", "active"]))
    assert encoder is not None
    assert set(y.tolist()) == {0, 1}
    assert list(encoder.inverse_transform(y)) == ["active", "inactive", "active"]
