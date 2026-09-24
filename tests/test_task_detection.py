"""Tests for regression vs classification endpoint detection."""

from __future__ import annotations

import pandas as pd
import pytest

from qsar_agent.tools.task_detection import detect_task


def test_continuous_activity_is_regression():
    result = detect_task([5.2, 5.5, 4.8, 6.1, 7.3, 5.9, 4.4, 6.6])
    assert result.task == "regression"
    assert result.n_unique_values == 8
    assert result.n_classes is None


def test_binary_integer_activity_is_classification():
    result = detect_task([0, 1, 1, 0, 1, 0, 0, 1])
    assert result.task == "classification"
    assert result.n_classes == 2
    assert result.class_counts == {"0": 4, "1": 4}
    assert result.imbalance_ratio == pytest.approx(1.0)


def test_imbalance_ratio_reported_for_skewed_classes():
    result = detect_task([0] * 90 + [1] * 10)
    assert result.task == "classification"
    assert result.imbalance_ratio == pytest.approx(9.0)
    assert result.positive_fraction == pytest.approx(0.10)


def test_string_labels_are_classification():
    result = detect_task(["active", "inactive", "active", "inactive"])
    assert result.task == "classification"
    assert result.n_classes == 2


def test_few_integer_levels_are_classification():
    result = detect_task([1, 2, 3, 1, 2, 3, 1, 2])
    assert result.task == "classification"
    assert result.n_classes == 3


def test_many_integer_levels_are_regression():
    # 30 distinct integers exceeds the class-level cap, so it stays continuous.
    result = detect_task(list(range(30)))
    assert result.task == "regression"


def test_forced_task_overrides_inference_but_records_it():
    result = detect_task([0, 1, 1, 0, 1], forced="regression")
    assert result.task == "regression"
    assert "forced" in result.reason.lower()


def test_forced_classification_computes_class_counts():
    result = detect_task([1.0, 2.0, 1.0, 2.0, 1.0], forced="classification")
    assert result.task == "classification"
    assert result.n_classes == 2


def test_empty_activity_is_an_error():
    with pytest.raises(ValueError, match="no usable values"):
        detect_task(pd.Series([None, None], dtype="object"))


def test_missing_values_are_ignored():
    result = detect_task(pd.Series([0, 1, None, 1, 0]))
    assert result.task == "classification"
    assert sum(result.class_counts.values()) == 4
