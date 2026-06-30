"""Default configuration for the QSAR Agent workflow."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class UMAPConfig(BaseModel):
    n_neighbors: int = 15
    n_components: int = 2
    min_dist: float = 0.1
    metric: str = "euclidean"


class ClusteringConfig(BaseModel):
    min_clusters: int = 2
    max_clusters: int = 10
    random_state: int = 60


class PreprocessingConfig(BaseModel):
    missing_value_threshold: float = 0.20
    near_constant_std_threshold: float = 0.01
    correlation_threshold: float = 0.95


class ModelConfig(BaseModel):
    estimator: str = "RandomForestRegressor"
    n_estimators: int = 100
    max_depth: int = 10
    random_state: int = 42
    n_jobs: int = -1


class DescriptorConfig(BaseModel):
    enable_3d: bool = False


class GAConfig(BaseModel):
    population_size: int = 50
    n_generations: int = 30
    crossover_prob: float = 0.7
    mutation_prob: float = 0.2
    tournament_size: int = 3
    cv_folds: int = 5
    n_jobs: int = -1
    random_seed: int = 42


class SFSConfig(BaseModel):
    max_features: int = 20
    cv_folds: int = 5
    random_seed: int = 42
    n_jobs: int = -1


class WorkflowConfig(BaseModel):
    test_fraction: float = 0.20
    random_seed: int = 42
    output_dir: str = "outputs"
    umap: UMAPConfig = Field(default_factory=UMAPConfig)
    clustering: ClusteringConfig = Field(default_factory=ClusteringConfig)
    preprocessing: PreprocessingConfig = Field(default_factory=PreprocessingConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    descriptors: DescriptorConfig = Field(default_factory=DescriptorConfig)
    ga: GAConfig = Field(default_factory=GAConfig)
    sfs: SFSConfig = Field(default_factory=SFSConfig)
    smiles_column: str = ""
    activity_column: str = ""
    id_column: str | None = None
    min_valid_compounds: int = 20

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump()


def get_openai_model() -> str:
    return os.environ.get("OPENAI_MODEL", "gpt-4o-mini")


def get_openai_api_key() -> str | None:
    key = os.environ.get("OPENAI_API_KEY")
    if key:
        return key
    try:
        import streamlit as st

        return st.secrets.get("OPENAI_API_KEY")
    except Exception:
        return None


def default_output_dir() -> Path:
    return Path("outputs")
