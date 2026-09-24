"""Tests for the unified regression and classification metrics."""

from __future__ import annotations

import numpy as np
import pytest

from qsar_agent.tools.metrics import (
    classification_metrics,
    compute_metrics,
    primary_metric_name,
    regression_metrics,
)


def test_perfect_regression_scores_one():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    metrics = regression_metrics(y, y)
    assert metrics["r2"] == pytest.approx(1.0)
    assert metrics["rmse"] == pytest.approx(0.0)
    assert metrics["mae"] == pytest.approx(0.0)


def test_regression_rmse_and_mae_are_computed_correctly():
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([2.0, 2.0, 2.0])
    metrics = regression_metrics(y_true, y_pred)
    assert metrics["mae"] == pytest.approx(2.0 / 3.0)
    assert metrics["rmse"] == pytest.approx(np.sqrt(2.0 / 3.0))


def test_regression_r2_is_negative_for_worse_than_mean():
    metrics = regression_metrics(np.array([1.0, 2.0, 3.0]), np.array([9.0, 9.0, 9.0]))
    assert metrics["r2"] < 0


def test_perfect_classification_scores_one():
    y = np.array([0, 1, 0, 1])
    metrics = classification_metrics(y, y, y_score=np.array([0.1, 0.9, 0.2, 0.8]))
    assert metrics["roc_auc"] == pytest.approx(1.0)
    assert metrics["mcc"] == pytest.approx(1.0)
    assert metrics["balanced_accuracy"] == pytest.approx(1.0)


def test_classification_reports_sensitivity_and_specificity():
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 1, 1, 1])
    metrics = classification_metrics(y_true, y_pred)
    assert metrics["sensitivity"] == pytest.approx(1.0)
    assert metrics["specificity"] == pytest.approx(0.5)


def test_balanced_accuracy_exposes_majority_class_prediction():
    # 90% of one class: raw accuracy looks fine, balanced accuracy does not.
    y_true = np.array([0] * 90 + [1] * 10)
    y_pred = np.zeros(100, dtype=int)
    metrics = classification_metrics(y_true, y_pred)
    assert metrics["accuracy"] == pytest.approx(0.90)
    assert metrics["balanced_accuracy"] == pytest.approx(0.50)
    assert metrics["mcc"] == pytest.approx(0.0)


def test_roc_auc_is_nan_without_scores():
    metrics = classification_metrics(np.array([0, 1, 0, 1]), np.array([0, 1, 1, 0]))
    assert np.isnan(metrics["roc_auc"])


def test_roc_auc_is_nan_for_a_single_class():
    metrics = classification_metrics(
        np.array([1, 1, 1]), np.array([1, 1, 1]), y_score=np.array([0.5, 0.6, 0.7])
    )
    assert np.isnan(metrics["roc_auc"])


def test_multiclass_roc_auc_uses_one_vs_rest():
    y_true = np.array([0, 1, 2, 0, 1, 2])
    proba = np.eye(3)[y_true].astype(float)
    metrics = classification_metrics(y_true, y_true, y_score=proba)
    assert metrics["roc_auc"] == pytest.approx(1.0)


def test_primary_metric_per_task():
    assert primary_metric_name("regression") == "r2"
    assert primary_metric_name("classification") == "roc_auc"
    with pytest.raises(ValueError, match="Unknown task"):
        primary_metric_name("clustering")


def test_compute_metrics_dispatches_on_task():
    y = np.array([0, 1, 0, 1])
    assert "rmse" in compute_metrics("regression", y, y)
    assert "mcc" in compute_metrics("classification", y, y)
    with pytest.raises(ValueError, match="Unknown task"):
        compute_metrics("ranking", y, y)
