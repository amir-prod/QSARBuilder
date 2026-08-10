"""Tests for expanded model catalog and compatibility."""

from __future__ import annotations

import pytest

from qsar_agent.agentic.compatibility import reject_arbitrary_import_path, validate_model_compatibility
from qsar_agent.config import ModelConfig
from qsar_agent.models.registry import (
    build_estimator_from_config,
    get_default_model_config,
    get_model_specification,
    list_available_estimators,
    list_registered_estimators,
    sanitize_param_grid,
)


SKLEARN_CATALOG = [
    "RandomForestRegressor",
    "ExtraTreesRegressor",
    "HistGradientBoostingRegressor",
    "GradientBoostingRegressor",
    "AdaBoostRegressor",
    "SVR",
    "KNeighborsRegressor",
    "ElasticNet",
    "Ridge",
    "PLSRegression",
]


@pytest.mark.parametrize("name", SKLEARN_CATALOG)
def test_registered_estimator_constructs(name):
    cfg = get_default_model_config(name, random_state=0, n_jobs=1)
    est = build_estimator_from_config(cfg)
    assert est is not None


@pytest.mark.parametrize("name", SKLEARN_CATALOG)
def test_registered_estimator_small_fit(name):
    import numpy as np

    cfg = get_default_model_config(name, random_state=0, n_jobs=1)
    est = build_estimator_from_config(cfg)
    X = np.random.RandomState(0).randn(30, 5)
    y = X[:, 0] + 0.1 * np.random.RandomState(1).randn(30)
    est.fit(X, y)
    pred = est.predict(X)
    assert len(pred) == 30


def test_optional_dependency_detection():
    for name in ("XGBRegressor", "CatBoostRegressor", "LGBMRegressor"):
        spec = get_model_specification(name)
        assert spec.estimator_name == name
        if not spec.available:
            assert spec.missing_dependency


def test_missing_optional_raises_on_build():
    for name in ("XGBRegressor", "CatBoostRegressor", "LGBMRegressor"):
        spec = get_model_specification(name)
        if spec.available:
            continue
        with pytest.raises(ValueError, match="unavailable|dependency"):
            get_default_model_config(name)


def test_compatibility_and_import_path_rejection():
    result = validate_model_compatibility("Ridge", n_train_samples=40, n_features=5)
    assert result.compatible
    assert "feature_scaling" in result.required_preprocessing
    with pytest.raises(ValueError):
        reject_arbitrary_import_path("sklearn.linear_model.Ridge")
    bad = validate_model_compatibility("NotAModel", n_train_samples=40, n_features=5)
    assert not bad.compatible


def test_pls_and_knn_bounds_in_sanitize():
    pls = sanitize_param_grid(
        "PLSRegression",
        {"n_components": [1, 5, 50]},
        n_features=3,
        n_train_samples=10,
        max_candidates=20,
    )
    assert max(pls.sanitized_grid["n_components"]) <= 3
    knn = sanitize_param_grid(
        "KNeighborsRegressor",
        {"n_neighbors": [1, 5, 15]},
        max_candidates=20,
    )
    assert knn.sanitized_grid


def test_svr_grid_shrink():
    result = sanitize_param_grid(
        "SVR",
        {
            "C": [0.1, 1.0, 10.0, 50.0],
            "epsilon": [0.01, 0.05, 0.1, 0.2],
            "gamma": ["scale", 0.01, 0.1],
            "kernel": ["rbf", "linear"],
        },
        max_candidates=10,
    )
    from qsar_agent.models.registry import count_grid_combinations

    assert count_grid_combinations(result.sanitized_grid) <= 10


def test_random_state_propagation():
    cfg = get_default_model_config("Ridge", random_state=123)
    assert cfg.random_state == 123
    est = build_estimator_from_config(cfg)
    assert getattr(est, "random_state", 123) == 123


def test_list_registered_includes_optional():
    all_names = list_registered_estimators(include_unavailable=True)
    for name in SKLEARN_CATALOG:
        assert name in all_names
    assert "XGBRegressor" in all_names
    avail = list_available_estimators()
    assert "RandomForestRegressor" in avail
