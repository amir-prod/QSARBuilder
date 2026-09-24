"""Tests for the task-aware estimator registry."""

from __future__ import annotations

import numpy as np
import pytest

from qsar_agent.tools.model_zoo import (
    build_estimator,
    decision_scores,
    default_hyperparameters,
    describe_zoo,
    list_estimators,
    validate_hyperparameters,
)


def test_both_tasks_expose_estimators():
    assert "RandomForest" in list_estimators("regression")
    assert "RandomForest" in list_estimators("classification")
    assert "LogisticRegression" in list_estimators("classification")
    assert "Ridge" in list_estimators("regression")


def test_regression_and_classification_zoos_differ():
    assert "Ridge" not in list_estimators("classification")
    assert "LogisticRegression" not in list_estimators("regression")


def test_unknown_task_is_rejected():
    with pytest.raises(ValueError, match="Unknown task"):
        list_estimators("survival")


@pytest.mark.parametrize("task", ["regression", "classification"])
def test_every_registered_estimator_builds_and_fits(task):
    rng = np.random.default_rng(0)
    X = rng.normal(size=(40, 6))
    y = (
        X[:, 0] * 2 + rng.normal(scale=0.1, size=40)
        if task == "regression"
        else (X[:, 0] > 0).astype(int)
    )
    for name in list_estimators(task):
        model = build_estimator(task, name, random_state=0, n_jobs=1)
        model.fit(X, y)
        assert len(model.predict(X)) == 40


def test_hyperparameters_override_defaults():
    model = build_estimator("regression", "RandomForest", {"n_estimators": 7})
    assert model.get_params()["n_estimators"] == 7


def test_unknown_estimator_names_the_alternatives():
    with pytest.raises(ValueError, match="Available:"):
        build_estimator("regression", "NeuralOracle")


def test_unknown_hyperparameter_is_rejected_with_its_name():
    with pytest.raises(ValueError, match="not_a_real_param"):
        validate_hyperparameters("regression", "RandomForest", {"not_a_real_param": 1})


def test_class_weight_rejected_on_estimators_without_support():
    with pytest.raises(ValueError, match="class_weight"):
        validate_hyperparameters("classification", "KNN", {"class_weight": "balanced"})


def test_class_weight_accepted_on_random_forest():
    model = build_estimator("classification", "RandomForest", {"class_weight": "balanced"})
    assert model.get_params()["class_weight"] == "balanced"


def test_logistic_regression_is_not_given_a_deprecated_n_jobs():
    # sklearn warns that n_jobs is a no-op here, so the zoo must not set it.
    model = build_estimator("classification", "LogisticRegression", n_jobs=4)
    assert model.get_params().get("n_jobs") is None


def test_random_forest_does_receive_n_jobs():
    model = build_estimator("regression", "RandomForest", n_jobs=3)
    assert model.get_params()["n_jobs"] == 3


def test_pls_clamps_components_to_available_features():
    model = build_estimator("regression", "PLS", {"n_components": 50})
    X = np.random.default_rng(1).normal(size=(20, 3))
    y = X[:, 0]
    model.fit(X, y)
    assert model.n_components <= 3
    assert model.predict(X).ndim == 1


def test_knn_clamps_neighbours_to_sample_count():
    model = build_estimator("regression", "KNN", {"n_neighbors": 99})
    X = np.random.default_rng(2).normal(size=(6, 2))
    model.fit(X, X[:, 0])
    assert model.n_neighbors <= 6


def test_describe_zoo_lists_defaults_and_tunables():
    entries = describe_zoo("regression")
    by_name = {entry["name"]: entry for entry in entries}
    assert "n_estimators" in by_name["RandomForest"]["default_hyperparameters"]
    assert "max_depth" in by_name["RandomForest"]["tunable_hyperparameters"]
    # The catalogue must not advertise plumbing the strategist should not set.
    assert "n_jobs" not in by_name["RandomForest"]["tunable_hyperparameters"]


def test_default_hyperparameters_are_a_copy():
    first = default_hyperparameters("regression", "RandomForest")
    first["n_estimators"] = -1
    assert default_hyperparameters("regression", "RandomForest")["n_estimators"] != -1


def test_decision_scores_prefers_probabilities():
    rng = np.random.default_rng(3)
    X = rng.normal(size=(30, 3))
    y = (X[:, 0] > 0).astype(int)
    model = build_estimator("classification", "RandomForest", {"n_estimators": 10})
    model.fit(X, y)
    scores = decision_scores(model, X)
    assert scores is not None
    assert scores.shape == (30,)
    assert scores.min() >= 0.0 and scores.max() <= 1.0
