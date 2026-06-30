"""Feature selection schemas."""

from __future__ import annotations

from pydantic import BaseModel, Field


class SFSResultRow(BaseModel):
    n_features: int
    mean_train_r2: float
    mean_cv_r2: float
    std_cv_r2: float
    selected_features: list[str]


class SFSResult(BaseModel):
    results: list[SFSResultRow]
    max_features_evaluated: int
    results_csv_path: str
    selected_features_json_path: str
    plot_png_path: str
    plot_svg_path: str


class FeatureCountSelection(BaseModel):
    best_cv_r2: float
    best_feature_count: int
    selected_feature_count: int
    selected_cv_r2: float
    explanation: str
    selection_json_path: str
    explanation_md_path: str


class GAResult(BaseModel):
    selected_features: list[str]
    best_fitness: float
    history_csv_path: str
    selected_features_path: str
    configuration_path: str
    convergence_png_path: str
    convergence_svg_path: str
