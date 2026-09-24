"""Tests for the y-randomization chance-correlation control."""

from __future__ import annotations

import numpy as np
import pytest

from qsar_agent.tools.model_zoo import build_estimator
from qsar_agent.tools.y_scrambling import run_y_scrambling


@pytest.fixture
def learnable_regression():
    rng = np.random.default_rng(7)
    X = rng.normal(size=(80, 4))
    y = 3.0 * X[:, 0] - 2.0 * X[:, 1] + rng.normal(scale=0.2, size=80)
    return X, y


@pytest.fixture
def learnable_classification():
    rng = np.random.default_rng(11)
    X = rng.normal(size=(80, 4))
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    return X, y


def test_scrambled_regression_scores_are_near_or_below_zero(learnable_regression):
    X, y = learnable_regression
    result = run_y_scrambling(
        X=X,
        y=y,
        estimator=build_estimator("regression", "Ridge"),
        task="regression",
        real_cv_score=0.95,
        n_repeats=5,
        cv_folds=3,
        random_seed=0,
    )
    assert result.mean_scrambled_score < 0.2
    assert result.margin > 0.5
    assert result.n_repeats == 5
    assert len(result.scrambled_scores) == 5


def test_scrambled_classification_auc_sits_near_one_half(learnable_classification):
    X, y = learnable_classification
    result = run_y_scrambling(
        X=X,
        y=y,
        estimator=build_estimator("classification", "LogisticRegression"),
        task="classification",
        real_cv_score=0.95,
        n_repeats=5,
        cv_folds=3,
        random_seed=0,
    )
    assert 0.25 < result.mean_scrambled_score < 0.75


def test_margin_is_real_minus_mean_scrambled(learnable_regression):
    X, y = learnable_regression
    result = run_y_scrambling(
        X=X,
        y=y,
        estimator=build_estimator("regression", "Ridge"),
        task="regression",
        real_cv_score=0.80,
        n_repeats=4,
        cv_folds=3,
        random_seed=1,
    )
    assert result.margin == pytest.approx(0.80 - result.mean_scrambled_score)
    assert result.real_cv_score == pytest.approx(0.80)


def test_pure_noise_leaves_no_margin_over_chance():
    rng = np.random.default_rng(13)
    X = rng.normal(size=(60, 5))
    y = rng.normal(size=60)
    result = run_y_scrambling(
        X=X,
        y=y,
        estimator=build_estimator("regression", "Ridge"),
        task="regression",
        real_cv_score=-0.15,
        n_repeats=6,
        cv_folds=3,
        random_seed=2,
    )
    # Real labels are noise too, so scrambling cannot be much worse.
    assert result.margin < 0.25
    assert result.fraction_scrambled_at_least_real > 0.0


def test_results_are_reproducible_for_a_fixed_seed(learnable_regression):
    X, y = learnable_regression
    kwargs = dict(
        X=X,
        y=y,
        estimator=build_estimator("regression", "Ridge"),
        task="regression",
        real_cv_score=0.9,
        n_repeats=4,
        cv_folds=3,
        random_seed=42,
    )
    first = run_y_scrambling(**kwargs)
    second = run_y_scrambling(**kwargs)
    assert first.scrambled_scores == second.scrambled_scores


def test_mismatched_lengths_are_rejected():
    with pytest.raises(ValueError, match="rows"):
        run_y_scrambling(
            X=np.zeros((10, 2)),
            y=np.zeros(9),
            estimator=build_estimator("regression", "Ridge"),
            task="regression",
            real_cv_score=0.5,
        )


def test_estimator_may_be_a_factory(learnable_regression):
    X, y = learnable_regression
    result = run_y_scrambling(
        X=X,
        y=y,
        estimator=lambda: build_estimator("regression", "Ridge"),
        task="regression",
        real_cv_score=0.9,
        n_repeats=3,
        cv_folds=3,
        random_seed=5,
    )
    assert len(result.scrambled_scores) == 3


def test_single_class_target_cannot_be_scrambled():
    with pytest.raises(ValueError, match="no valid scores"):
        run_y_scrambling(
            X=np.random.default_rng(0).normal(size=(30, 3)),
            y=np.ones(30, dtype=int),
            estimator=build_estimator("classification", "LogisticRegression"),
            task="classification",
            real_cv_score=0.9,
            n_repeats=3,
            cv_folds=3,
        )
