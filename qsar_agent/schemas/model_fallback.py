"""Schemas for multi-model fallback."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from qsar_agent.config import ModelConfig
from qsar_agent.schemas.feature_selection import FeatureCountSelection, GAResult, SFSResult
from qsar_agent.schemas.hyperparameter_optimization import FinalModelSelection, HPOResult


class ModelBranchResult(BaseModel):
    estimator: str
    model_config_snapshot: dict[str, Any] = Field(default_factory=dict)
    branch_dir: str = ""
    runtime_seconds: float | None = None
    sfs: SFSResult
    feature_count: FeatureCountSelection
    ga: GAResult
    hpo_result: HPOResult
    # Optional SFS-fixed + GA expansion branch (separate folder artifacts).
    expansion: ModelBranchResult | None = None
    # One-SE SFS subset with default hyperparameters.
    sfs_subset: ModelBranchResult | None = None
    # Same SFS subset after HPO, only when HPO selected non-baseline params.
    sfs_subset_hpo: ModelBranchResult | None = None
    # True when this branch itself is an expansion (for comparison labels).
    is_expansion: bool = False
    expansion_label: str = ""


class CrossModelSelection(BaseModel):
    winning_estimator: str
    selected_features: list[str]
    final_model_config: dict[str, Any]
    final_selection: FinalModelSelection
    selection_rationale: str
    warning: str = ""
    compared_models: list[dict[str, Any]] = Field(default_factory=list)
    winner_is_expansion: bool = False
    winner_expansion_label: str = ""


class BranchExternalArtifacts(BaseModel):
    """Paths and metrics from train-fit + external-test eval of one branch."""

    estimator: str
    label: str
    is_expansion: bool = False
    expansion_label: str = ""
    branch_dir: str = ""
    selected_features: list[str] = Field(default_factory=list)
    predictions_path: str = ""
    model_path: str = ""
    metrics_path: str = ""
    scatter_png_path: str = ""
    scatter_svg_path: str = ""
    williams_png_path: str = ""
    williams_svg_path: str = ""
    ad_report_path: str = ""
    ad_classifications_path: str = ""
    residual_png_path: str = ""
    residual_svg_path: str = ""
    train_r2: float = 0.0
    val_r2: float | None = None
    test_r2: float = 0.0
    runtime_seconds: float | None = None


class ModelFallbackResult(BaseModel):
    triggered: bool
    fallback_models_tried: list[str] = Field(default_factory=list)
    rf_branch: ModelBranchResult
    fallback_branches: list[ModelBranchResult] = Field(default_factory=list)
    cross_model_selection: CrossModelSelection | None = None
    comparison_json_path: str = ""
    comparison_md_path: str = ""
    comparison_csv_path: str = ""
    branch_external_artifacts: list[BranchExternalArtifacts] = Field(default_factory=list)
