"""Shared modeling utilities."""

from __future__ import annotations

from typing import Any

from sklearn.ensemble import RandomForestRegressor

from qsar_agent.config import ModelConfig


def build_estimator(config: ModelConfig | dict[str, Any] | None = None):
    if isinstance(config, dict):
        cfg = ModelConfig(**{**ModelConfig().model_dump(), **config})
    else:
        cfg = config or ModelConfig()
    if cfg.estimator == "RandomForestRegressor":
        kwargs: dict[str, Any] = {
            "n_estimators": cfg.n_estimators,
            "max_depth": cfg.max_depth,
            "min_samples_split": cfg.min_samples_split,
            "min_samples_leaf": cfg.min_samples_leaf,
            "max_features": cfg.max_features,
            "bootstrap": cfg.bootstrap,
            "criterion": cfg.criterion,
            "random_state": cfg.random_state,
            "n_jobs": cfg.n_jobs,
        }
        if cfg.bootstrap and cfg.max_samples is not None:
            kwargs["max_samples"] = cfg.max_samples
        return RandomForestRegressor(**kwargs)
    raise ValueError(f"Unsupported estimator: {cfg.estimator}")
