"""Strict allowlisted action registry with Pydantic input schemas."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

from qsar_agent.schemas.agentic import AllowedAction, ExperimentKind, normalize_action


class ActionSpec(BaseModel):
    action: AllowedAction
    executable: bool
    requires_user_approval: bool = False
    experiment_kind: ExperimentKind
    multi_component: bool = False
    description: str
    # Data Quality / informational actions that cannot mutate science in v1
    diagnostic_only: bool = False


class RefineHyperparametersParams(BaseModel):
    estimator: str | None = None
    max_candidates: int = Field(default=40, ge=1, le=60)
    status_hint: Literal["overfit", "underfit", "unstable", "default", "poor_performance"] = "default"
    param_grid: dict[str, list[Any]] | None = None


class FeatureCountParams(BaseModel):
    feature_count: int = Field(ge=1, le=100)
    relative_delta: int | None = None


class SFSFixedGAParams(BaseModel):
    extra_features: int = Field(default=2, ge=1, le=10)


class TryRegisteredEstimatorParams(BaseModel):
    estimator: str
    mode: Literal["controlled", "full_pipeline"] = "controlled"
    run_hpo: bool = False
    max_candidates: int = Field(default=40, ge=1, le=60)

    @field_validator("estimator")
    @classmethod
    def no_import_path(cls, v: str) -> str:
        if "." in v or "/" in v or "\\" in v:
            raise ValueError("estimator must be a registry name, not an import path")
        return v


class CompareRegisteredEstimatorsParams(BaseModel):
    estimators: list[str] = Field(min_length=1, max_length=5)
    optimize_top_k: int = Field(default=2, ge=0, le=2)

    @field_validator("estimators")
    @classmethod
    def validate_names(cls, v: list[str]) -> list[str]:
        for name in v:
            if "." in name or "/" in name:
                raise ValueError("estimators must be registry names, not import paths")
        # unique preserve order
        seen: set[str] = set()
        out: list[str] = []
        for name in v:
            if name not in seen:
                seen.add(name)
                out.append(name)
        return out[:5]


class RecommendUnregisteredParams(BaseModel):
    estimator_name: str
    model_family: str = "unknown"
    rationale: str = ""
    required_package: str | None = None


class RequestApprovalParams(BaseModel):
    topic: str
    rationale: str
    proposed_change: dict[str, Any] = Field(default_factory=dict)
    # Only topics with deterministic handlers may be approved for execution
    executable_topic: bool = False


class EmptyParams(BaseModel):
    reason: str = ""


ACTION_REGISTRY: dict[str, ActionSpec] = {
    "accept_model": ActionSpec(
        action="accept_model",
        executable=True,
        experiment_kind="stop",
        description="Accept current best model and stop agentic improvement.",
    ),
    "refine_hyperparameters": ActionSpec(
        action="refine_hyperparameters",
        executable=True,
        experiment_kind="hyperparameter_refinement",
        description="Bounded HPO refinement on fixed features.",
    ),
    "reduce_feature_count": ActionSpec(
        action="reduce_feature_count",
        executable=True,
        experiment_kind="feature_count_change",
        multi_component=True,
        description="Reduce selected feature count and re-run GA/HPO.",
    ),
    "expand_feature_count": ActionSpec(
        action="expand_feature_count",
        executable=True,
        experiment_kind="feature_count_change",
        multi_component=True,
        description="Expand selected feature count and re-run GA/HPO.",
    ),
    "run_sfs_fixed_ga_expansion": ActionSpec(
        action="run_sfs_fixed_ga_expansion",
        executable=True,
        experiment_kind="sfs_fixed_ga_expansion",
        multi_component=True,
        description="Freeze SFS features and GA-add extras.",
    ),
    "try_registered_estimator": ActionSpec(
        action="try_registered_estimator",
        executable=True,
        experiment_kind="controlled_estimator_comparison",
        description="Try a registry estimator (controlled or full pipeline).",
    ),
    "compare_registered_estimators": ActionSpec(
        action="compare_registered_estimators",
        executable=True,
        experiment_kind="controlled_estimator_comparison",
        description="Screen multiple registered estimators with shared folds/features.",
    ),
    "recommend_unregistered_estimator": ActionSpec(
        action="recommend_unregistered_estimator",
        executable=False,
        diagnostic_only=True,
        experiment_kind="diagnostic_only",
        description="Record unregistered estimator recommendation without execution.",
    ),
    "request_model_dependency_approval": ActionSpec(
        action="request_model_dependency_approval",
        executable=False,
        requires_user_approval=True,
        diagnostic_only=True,
        experiment_kind="diagnostic_only",
        description="Ask user about installing/enabling an optional model dependency.",
    ),
    "request_user_approval": ActionSpec(
        action="request_user_approval",
        executable=False,
        requires_user_approval=True,
        diagnostic_only=True,
        experiment_kind="diagnostic_only",
        description="Pause for user approval (only for executable topics).",
    ),
    "request_user_input": ActionSpec(
        action="request_user_input",
        executable=False,
        diagnostic_only=True,
        experiment_kind="diagnostic_only",
        description="Request informational user input; no dataset mutation in v1.",
    ),
    "stop_budget_exhausted": ActionSpec(
        action="stop_budget_exhausted",
        executable=True,
        experiment_kind="stop",
        description="Stop because experiment/cycle budget is exhausted.",
    ),
    "stop_no_viable_model": ActionSpec(
        action="stop_no_viable_model",
        executable=True,
        experiment_kind="stop",
        description="Stop because no scientifically defensible action remains.",
    ),
}


PARAM_MODELS: dict[str, type[BaseModel]] = {
    "refine_hyperparameters": RefineHyperparametersParams,
    "reduce_feature_count": FeatureCountParams,
    "expand_feature_count": FeatureCountParams,
    "run_sfs_fixed_ga_expansion": SFSFixedGAParams,
    "try_registered_estimator": TryRegisteredEstimatorParams,
    "compare_registered_estimators": CompareRegisteredEstimatorsParams,
    "recommend_unregistered_estimator": RecommendUnregisteredParams,
    "request_model_dependency_approval": RequestApprovalParams,
    "request_user_approval": RequestApprovalParams,
    "request_user_input": EmptyParams,
    "accept_model": EmptyParams,
    "stop_budget_exhausted": EmptyParams,
    "stop_no_viable_model": EmptyParams,
}


# Topics that may be approved AND executed in v1 (none for data mutation)
EXECUTABLE_APPROVAL_TOPICS: frozenset[str] = frozenset(
    {
        "enable_optional_model_dependency",  # still does not auto-install; only records approval
        "expand_experiment_budget",
    }
)


def get_action_spec(action: str) -> ActionSpec:
    action_n = normalize_action(action)
    if action_n not in ACTION_REGISTRY:
        raise ValueError(f"Action not allowlisted: {action}")
    return ACTION_REGISTRY[action_n]


def validate_action_params(action: str, params: dict[str, Any]) -> BaseModel:
    action_n = normalize_action(action)
    get_action_spec(action_n)
    model = PARAM_MODELS.get(action_n, EmptyParams)
    return model.model_validate(params or {})


def is_action_allowed(action: str) -> bool:
    try:
        get_action_spec(action)
        return True
    except ValueError:
        return False


def resolve_experiment_kind(action: str, params: dict[str, Any]) -> tuple[ExperimentKind, bool, list[str]]:
    """Return experiment_kind, multi_component, component_list."""
    action_n = normalize_action(action)
    spec = get_action_spec(action_n)
    if action_n == "try_registered_estimator":
        mode = params.get("mode", "controlled")
        if mode == "full_pipeline":
            return (
                "full_pipeline_branch",
                True,
                ["feature_selection", "genetic_algorithm", "hyperparameter_optimization", "estimator"],
            )
        return ("controlled_estimator_comparison", False, ["estimator"])
    if action_n == "compare_registered_estimators":
        return ("controlled_estimator_comparison", False, ["estimator"])
    if action_n in ("reduce_feature_count", "expand_feature_count"):
        return (
            "feature_count_change",
            True,
            ["feature_count_selection", "genetic_algorithm", "hyperparameter_optimization"],
        )
    if action_n == "run_sfs_fixed_ga_expansion":
        return (
            "sfs_fixed_ga_expansion",
            True,
            ["sfs_fixed_features", "genetic_algorithm", "hyperparameter_optimization"],
        )
    if action_n == "refine_hyperparameters":
        return ("hyperparameter_refinement", False, ["hyperparameter_optimization"])
    return (spec.experiment_kind, spec.multi_component, [])
