"""UMAP split schemas."""

from __future__ import annotations

from pydantic import BaseModel, Field


class ClusterInfo(BaseModel):
    cluster_id: int
    size: int


class SplitResult(BaseModel):
    train_count: int
    test_count: int
    test_fraction_actual: float
    n_clusters: int
    cluster_sizes: list[ClusterInfo]
    train_activity_mean: float
    train_activity_std: float
    test_activity_mean: float
    test_activity_std: float
    train_path: str
    test_path: str
    split_assignments_path: str
    umap_coordinates_path: str
    umap_plot_png: str
    umap_plot_svg: str
    split_report_path: str
    warnings: list[str] = Field(default_factory=list)
