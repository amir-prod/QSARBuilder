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
    sfs: SFSResult
    feature_count: FeatureCountSelection
    ga: GAResult
    hpo_result: HPOResult


class CrossModelSelection(BaseModel):
    winning_estimator: str
    selected_features: list[str]
    final_model_config: dict[str, Any]
    final_selection: FinalModelSelection
    selection_rationale: str
    warning: str = ""
    compared_models: list[dict[str, Any]] = Field(default_factory=list)


class ModelFallbackResult(BaseModel):
    triggered: bool
    fallback_models_tried: list[str] = Field(default_factory=list)
    rf_branch: ModelBranchResult
    fallback_branches: list[ModelBranchResult] = Field(default_factory=list)
    cross_model_selection: CrossModelSelection | None = None
    comparison_json_path: str = ""
    comparison_md_path: str = ""
    comparison_csv_path: str = ""
