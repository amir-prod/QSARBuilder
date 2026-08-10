"""Default configuration for the QSAR Agent workflow."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from qsar_agent.schemas.agentic import (
    AgenticAcceptanceCriteria,
    AgenticImprovementConfig,
)


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
    max_depth: int | None = 10
    min_samples_split: int = 2
    min_samples_leaf: int = 1
    max_features: str | float = "sqrt"
    bootstrap: bool = True
    max_samples: float | None = None
    criterion: str = "squared_error"
    random_state: int = 42
    n_jobs: int = -1
    params: dict[str, Any] = Field(default_factory=dict)


class HPOSettings(BaseModel):
    enabled: bool = True
    max_hpo_rounds: int = 3
    cv_folds: int = 5
    max_candidates_per_round: int = 120
    min_cv_improvement: float = 0.02
    overfit_gap_threshold: float = 0.15
    severe_overfit_gap_threshold: float = 0.25
    minimum_cv_r2: float = 0.50
    cv_std_threshold: float = 0.15
    minimum_train_r2: float = 0.40
    n_jobs: int = -1
    openai_model: str = ""


class DescriptorConfig(BaseModel):
    backends: list[str] = Field(default_factory=lambda: ["RDKit", "Mordred"])
    run_geometry_optimization: bool = False
    num_workers: int = 4
    xtb_timeout: int = 600
    external_descriptors_path: str | None = None


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


class ModelFallbackSettings(BaseModel):
    enabled: bool = True
    estimators: list[str] = Field(
        default_factory=lambda: [
            "PLSRegression",
            "ExtraTreesRegressor",
            "SVR",
            "KNeighborsRegressor",
        ]
    )


class SFSFixedGAExpansionSettings(BaseModel):
    """When a model is not acceptable, freeze SFS features and GA-add extras."""

    enabled: bool = True
    extra_features: int = 2
    output_subdir: str = "sfs_fixed_ga_plus2"


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
    hpo: HPOSettings = Field(default_factory=HPOSettings)
    model_fallback: ModelFallbackSettings = Field(default_factory=ModelFallbackSettings)
    sfs_fixed_ga_expansion: SFSFixedGAExpansionSettings = Field(
        default_factory=SFSFixedGAExpansionSettings
    )
    agentic: AgenticImprovementConfig = Field(default_factory=AgenticImprovementConfig)
    smiles_column: str = ""
    activity_column: str = ""
    id_column: str | None = None
    min_valid_compounds: int = 20

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump()


# Re-export for callers that import agentic settings from config.
__all_agentic__ = (
    "AgenticAcceptanceCriteria",
    "AgenticImprovementConfig",
)


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def load_env_file(env_path: Path | None = None) -> bool:
    """
    Load environment variables from a .env file.

    Searches the project root by default. Returns True if a file was loaded.
    Existing environment variables are not overwritten.
    """
    try:
        from dotenv import load_dotenv
    except ImportError:
        return False

    path = env_path or (_project_root() / ".env")
    if not path.exists():
        return False
    load_dotenv(path, override=False)
    return True


# Load .env when this module is imported (Streamlit, CLI, tests).
load_env_file()


def get_openai_model() -> str:
    model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
    return model.strip().strip('"').strip("'")


def _normalize_api_key(value: str | None) -> str | None:
    if not value:
        return None
    key = str(value).strip().strip('"').strip("'")
    return key or None


def get_openai_api_key_source() -> tuple[str | None, str | None]:
    """
    Resolve OpenAI API key and where it came from.

    Priority: environment / .env → Streamlit secrets → Streamlit sidebar paste.
    Returns ``(key, source)`` where source is one of
    ``environment``, ``streamlit_secrets``, ``ui``, or ``None``.
    """
    key = _normalize_api_key(os.environ.get("OPENAI_API_KEY"))
    if key:
        return key, "environment"

    try:
        import streamlit as st

        secret = _normalize_api_key(st.secrets.get("OPENAI_API_KEY"))
        if secret:
            return secret, "streamlit_secrets"

        ui_key = _normalize_api_key(st.session_state.get("openai_api_key"))
        if ui_key:
            return ui_key, "ui"
    except Exception:
        pass

    return None, None


def get_openai_api_key() -> str | None:
    key, _source = get_openai_api_key_source()
    return key


def default_output_dir() -> Path:
    return Path("outputs")
