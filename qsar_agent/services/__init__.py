"""Shared modeling utilities."""

from __future__ import annotations

from sklearn.ensemble import RandomForestRegressor

from qsar_agent.config import ModelConfig


def build_estimator(config: ModelConfig | None = None):
    cfg = config or ModelConfig()
    if cfg.estimator == "RandomForestRegressor":
        return RandomForestRegressor(
            n_estimators=cfg.n_estimators,
            max_depth=cfg.max_depth,
            random_state=cfg.random_state,
            n_jobs=cfg.n_jobs,
        )
    raise ValueError(f"Unsupported estimator: {cfg.estimator}")
