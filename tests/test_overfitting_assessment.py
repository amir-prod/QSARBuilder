"""Tests for overfitting assessment."""

from qsar_agent.schemas.hyperparameter_optimization import CVSummary, OverfittingThresholds
from qsar_agent.tools.overfitting_assessment import assess_overfitting


def _summary(mean_train, mean_cv, std_cv) -> CVSummary:
    return CVSummary(
        mean_train_r2=mean_train,
        mean_cv_r2=mean_cv,
        std_cv_r2=std_cv,
        mean_train_rmse=0.1,
        mean_cv_rmse=0.2,
        mean_train_mae=0.1,
        mean_cv_mae=0.2,
        train_cv_r2_gap=mean_train - mean_cv,
        n_folds=5,
    )


def test_overfitting_assessment_overfit():
    result = assess_overfitting(_summary(0.95, 0.55, 0.05))
    assert result.status == "overfit"
    assert result.is_overfit
    assert not result.is_acceptable


def test_overfitting_assessment_acceptable():
    result = assess_overfitting(_summary(0.72, 0.65, 0.06))
    assert result.status == "good"
    assert result.is_acceptable
    assert not result.is_overfit


def test_overfitting_assessment_underfit():
    result = assess_overfitting(_summary(0.25, 0.20, 0.04))
    assert result.status == "underfit"
    assert result.is_underfit
    assert not result.is_acceptable


def test_overfitting_assessment_unstable():
    result = assess_overfitting(_summary(0.70, 0.62, 0.25))
    assert result.status == "unstable"
    assert result.is_unstable


def test_overfitting_assessment_poor_performance():
    result = assess_overfitting(_summary(0.60, 0.35, 0.05))
    assert result.status in ("poor_performance", "overfit")
