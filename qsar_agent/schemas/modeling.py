"""Modeling schemas."""

from __future__ import annotations

from pydantic import BaseModel, Field


class Metrics(BaseModel):
    r2: float
    rmse: float
    mae: float
    n_samples: int


class ModelingResult(BaseModel):
    train_metrics: Metrics
    val_metrics: Metrics | None = None
    test_metrics: Metrics
    selected_features: list[str]
    predictions_path: str
    metrics_path: str
    model_path: str
    scatter_png_path: str
    scatter_svg_path: str
    manifest_path: str
    hpo_enabled: bool = False
    hpo_rounds_completed: int = 0
    final_model_source: str = "baseline"
