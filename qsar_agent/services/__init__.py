"""Shared modeling utilities."""

from __future__ import annotations

from typing import Any

from qsar_agent.config import ModelConfig
from qsar_agent.models.registry import build_estimator_from_config


def build_estimator(config: ModelConfig | dict[str, Any] | None = None):
    return build_estimator_from_config(config)
