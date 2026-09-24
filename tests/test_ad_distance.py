"""Tests for the k-NN distance-to-model applicability domain."""

from __future__ import annotations

import numpy as np
import pytest

from qsar_agent.tools.ad_distance import (
    evaluate_applicability_domain,
    knn_ad_threshold,
    training_outlier_mask,
)


@pytest.fixture
def tight_training_cluster():
    rng = np.random.default_rng(3)
    return rng.normal(loc=0.0, scale=1.0, size=(60, 3))


def test_compounds_from_the_training_distribution_are_in_domain(tight_training_cluster):
    rng = np.random.default_rng(4)
    X_test = rng.normal(loc=0.0, scale=1.0, size=(20, 3))
    summary, mask = evaluate_applicability_domain(tight_training_cluster, X_test)
    assert summary.test_coverage_pct > 80.0
    assert mask.sum() == summary.test_in_domain_count


def test_distant_compounds_fall_outside_the_domain(tight_training_cluster):
    X_test = np.full((5, 3), 50.0)
    summary, mask = evaluate_applicability_domain(tight_training_cluster, X_test)
    assert summary.test_in_domain_count == 0
    assert summary.test_coverage_pct == pytest.approx(0.0)
    assert not mask.any()


def test_out_of_domain_identifiers_are_reported(tight_training_cluster):
    X_test = np.vstack([np.zeros((2, 3)), np.full((2, 3), 40.0)])
    summary, _mask = evaluate_applicability_domain(
        tight_training_cluster, X_test, test_ids=["near1", "near2", "far1", "far2"]
    )
    assert set(summary.out_of_domain_ids) == {"far1", "far2"}


def test_default_identifiers_are_positional(tight_training_cluster):
    summary, _mask = evaluate_applicability_domain(
        tight_training_cluster, np.full((2, 3), 40.0)
    )
    assert summary.out_of_domain_ids == ["0", "1"]


def test_a_larger_z_factor_widens_the_domain(tight_training_cluster):
    X_test = np.full((10, 3), 3.5)
    strict, _ = evaluate_applicability_domain(tight_training_cluster, X_test, z_factor=0.5)
    lenient, _ = evaluate_applicability_domain(tight_training_cluster, X_test, z_factor=10.0)
    assert lenient.test_coverage_pct >= strict.test_coverage_pct
    assert lenient.threshold > strict.threshold


def test_threshold_is_mean_plus_z_times_std(tight_training_cluster):
    threshold, distances, _index = knn_ad_threshold(tight_training_cluster, k=5, z_factor=3.0)
    expected = distances.mean() + 3.0 * distances.std(ddof=0)
    assert threshold == pytest.approx(expected)


def test_k_is_clamped_to_the_training_set_size():
    X_train = np.random.default_rng(5).normal(size=(4, 2))
    summary, _mask = evaluate_applicability_domain(X_train, X_train, k=25)
    assert summary.k <= 4


def test_a_single_training_compound_cannot_define_a_domain():
    with pytest.raises(ValueError, match="(?i)at least 2"):
        knn_ad_threshold(np.zeros((1, 3)))


def test_training_outlier_mask_flags_the_planted_outlier():
    rng = np.random.default_rng(6)
    X_train = np.vstack([rng.normal(scale=0.5, size=(50, 2)), np.full((1, 2), 25.0)])
    mask = training_outlier_mask(X_train)
    assert mask[-1]
    assert mask.sum() < 5


def test_a_homogeneous_training_set_has_no_outliers():
    X_train = np.tile(np.arange(20.0).reshape(-1, 1), (1, 2))
    assert not training_outlier_mask(X_train).any()
