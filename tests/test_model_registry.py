"""Tests for model registry."""

from __future__ import annotations

import pytest

from qsar_agent.config import ModelConfig
from qsar_agent.models.registry import (
    build_estimator_from_config,
    get_default_model_config,
    get_fallback_grid,
    model_simplicity_score,
    sanitize_param_grid,
)
from qsar_agent.services import build_estimator


@pytest.mark.parametrize(
    "estimator",
    [
        "RandomForestRegressor",
        "PLSRegression",
        "ExtraTreesRegressor",
        "SVR",
        "KNeighborsRegressor",
    ],
)
def test_build_estimator_supported_types(estimator):
    cfg = get_default_model_config(estimator, random_state=42, n_jobs=1)
    model = build_estimator(cfg)
    assert model is not None
    assert build_estimator_from_config(cfg) is not None


def test_svr_grid_sanitization():
    grid = {
        "C": [0.1, 1.0, 1000.0],
        "epsilon": [0.1],
        "kernel": ["rbf"],
        "bad_param": [1],
    }
    result = sanitize_param_grid("SVR", grid, max_candidates=20)
    assert "bad_param" not in result.sanitized_grid
    assert 1000.0 not in result.sanitized_grid["C"]


def test_pls_fallback_grids_differ_by_status():
    overfit = get_fallback_grid("PLSRegression", "overfit")
    underfit = get_fallback_grid("PLSRegression", "underfit")
    assert overfit != underfit


def test_model_simplicity_prefers_fewer_pls_components():
    simple = model_simplicity_score("PLSRegression", {"n_components": 3})
    complex_ = model_simplicity_score("PLSRegression", {"n_components": 20})
    assert simple < complex_


def test_model_config_rf_backward_compat():
    cfg = ModelConfig(n_estimators=200, max_depth=8)
    est = build_estimator(cfg)
    assert est.n_estimators == 200


def test_rf_criterion_rejects_friedman_mse():
    grid = {
        "n_estimators": [100, 200],
        "criterion": ["squared_error", "friedman_mse", "poisson"],
        "max_depth": [5, 10],
    }
    result = sanitize_param_grid("RandomForestRegressor", grid, max_candidates=20)
    assert "friedman_mse" not in result.sanitized_grid["criterion"]
    assert set(result.sanitized_grid["criterion"]).issubset(
        {"squared_error", "absolute_error", "poisson"}
    )


def test_build_estimator_clears_max_samples_when_bootstrap_false():
    cfg = ModelConfig(bootstrap=False, max_samples=0.7, n_jobs=1)
    est = build_estimator(cfg)
    assert est.bootstrap is False
    assert est.max_samples is None
