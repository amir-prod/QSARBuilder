"""Pydantic models for the post-handoff modeling-improvement agent."""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field


class PipelinePhase(str, Enum):
    DEVELOPMENT = "DEVELOPMENT"
    FROZEN = "FROZEN"
    EXTERNAL_EVALUATED = "EXTERNAL_EVALUATED"


GAObjectiveName = Literal[
    "mean_cv_r2",
    "cv_r2_minus_complexity",
    "cv_r2_minus_variance",
    "cv_r2_minus_overfit_gap",
    "balanced_r2_rmse",
    "pareto_r2_stability_feature_count",
    "regularized_cv_score",
]

FeatureSelectionMethodName = Literal[
    "genetic_algorithm",
    "sequential_forward",
    "sequential_backward",
    "rfe",
    "elastic_net",
    "mutual_information",
    "model_embedded",
    "stability_selection",
    "pca",
    "pls",
]

StabilityStatus = Literal["stable", "mixed", "unstable"]
ExclusionRecommendation = Literal["retain", "audit", "restrict_domain", "propose_exclusion"]
ConfidenceLevel = Literal["low", "medium", "high"]
MetricDirection = Literal["increase", "decrease", "maintain_or_improve", "maintain"]


APPROVED_TOOL_NAMES = (
    "validate_handoff",
    "evaluate_requirements",
    "run_model_search",
    "run_feature_selection_search",
    "run_representation_experiment",
    "run_robustness_analysis",
    "run_residual_analysis",
    "run_applicability_domain_analysis",
    "detect_persistent_outliers",
    "audit_compound",
    "run_exclusion_sensitivity_analysis",
    "compare_experiments",
    "request_new_capability",
    "freeze_pipeline",
    "evaluate_sealed_test",
)


class FailedRequirement(BaseModel):
    name: str
    observed: float | int | None = None
    required: float | int | None = None
    message: str = ""


class RequirementEvaluation(BaseModel):
    acceptance_status: Literal["passed", "failed"]
    passed_requirements: list[str] = Field(default_factory=list)
    failed_requirements: list[FailedRequirement] = Field(default_factory=list)
    train_cv_gap: float | None = None
    train_cv_gap_definition: str = (
        "mean_inner_fold_train_r2 - mean_inner_fold_validation_r2"
    )
    refit_train_cv_gap: float | None = None
    metrics: dict[str, float | None] = Field(default_factory=dict)


class EvidenceItem(BaseModel):
    observation: str
    value: float | int | str | None = None
    interpretation: str = ""


class AgentAction(BaseModel):
    tool_name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class ExpectedEffect(BaseModel):
    cv_r2: MetricDirection | str | None = None
    train_cv_gap: MetricDirection | str | None = None
    extra: dict[str, Any] = Field(default_factory=dict)


class SuccessConditions(BaseModel):
    minimum_cv_r2: float | None = None
    maximum_train_cv_gap: float | None = None
    extra: dict[str, Any] = Field(default_factory=dict)


class AgentDecision(BaseModel):
    diagnosis: str
    evidence: list[EvidenceItem] = Field(default_factory=list)
    hypothesis: str
    action: AgentAction
    expected_effect: ExpectedEffect = Field(default_factory=ExpectedEffect)
    success_conditions: SuccessConditions = Field(default_factory=SuccessConditions)
    reason_existing_results_are_insufficient: str = ""
    confidence: ConfidenceLevel = "medium"


class GAObjective(BaseModel):
    name: GAObjectiveName = "mean_cv_r2"
    cv_r2_weight: float = 1.0
    overfit_gap_penalty: float = 0.5
    cv_std_penalty: float = 0.25
    feature_count_penalty: float = 0.05
    rmse_weight: float = 0.1


class FeatureStabilityReport(BaseModel):
    selection_frequency: dict[str, float] = Field(default_factory=dict)
    mean_pairwise_jaccard: float = 0.0
    stable_features: list[str] = Field(default_factory=list)
    unstable_features: list[str] = Field(default_factory=list)
    stability_status: StabilityStatus = "unstable"


class PersistentOutlierReport(BaseModel):
    compound_id: str
    oof_response_outlier_frequency: float
    structural_outlier_frequency: float
    model_families_flagging: list[str] = Field(default_factory=list)
    possible_data_quality_issue: str | None = None
    recommended_action: ExclusionRecommendation = "retain"


class ExclusionProposal(BaseModel):
    compound_id: str
    proposed_reason: str
    evidence: list[str] = Field(default_factory=list)
    source_of_verification: str = ""
    models_and_runs_flagging: list[str] = Field(default_factory=list)
    oof_residual_frequency: float | None = None
    structural_outlier_frequency: float | None = None
    expected_scientific_effect: str = ""
    required_approval: bool = True
    case: Literal["invalid_data", "outside_domain", "difficult_valid"] = "difficult_valid"


class CapabilityRequest(BaseModel):
    capability: str
    scientific_reason: str
    evidence_from_current_results: list[dict[str, Any]] = Field(default_factory=list)
    required_inputs: list[str] = Field(default_factory=list)
    expected_outputs: list[str] = Field(default_factory=list)
    existing_tools_considered: list[str] = Field(default_factory=list)
    why_existing_tools_are_insufficient: str = ""
    leakage_risks: list[str] = Field(default_factory=list)
    reproducibility_risks: list[str] = Field(default_factory=list)
    compute_risks: list[str] = Field(default_factory=list)
    suggested_deterministic_implementation: str = ""
    approval_required: bool = True


class CandidateRanking(BaseModel):
    experiment_id: str
    eligible: bool
    failed_requirements: list[str] = Field(default_factory=list)
    selection_score: float = 0.0
    rank: int = 0
    selection_reason: str = ""


class HandoffValidationResult(BaseModel):
    passed: bool
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    package: dict[str, Any] | None = None
    dataset_hash: str = ""
    development_split_hash: str = ""
    sealed_test_hash: str = ""


class ToolResult(BaseModel):
    experiment_id: str
    tool_name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    status: Literal["completed", "skipped", "failed", "rejected"] = "completed"
    metrics: dict[str, float | None] = Field(default_factory=dict)
    selected_features: list[str] = Field(default_factory=list)
    artifact_paths: dict[str, str] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    runtime_seconds: float | None = None
    parent_experiment_id: str | None = None
    extra: dict[str, Any] = Field(default_factory=dict)


class ModelingAgentState(BaseModel):
    """Typed LangGraph state for the modeling-improvement agent."""

    project_id: str = ""
    run_dir: str = ""
    dataset_hash: str = ""
    development_split_hash: str = ""
    sealed_test_hash: str = ""
    phase: PipelinePhase = PipelinePhase.DEVELOPMENT
    requirements: dict[str, Any] = Field(default_factory=dict)
    handoff_validation: dict[str, Any] | None = None
    dataset_summary: dict[str, Any] = Field(default_factory=dict)
    available_representations: list[str] = Field(default_factory=list)
    available_model_families: list[str] = Field(default_factory=list)
    completed_experiments: list[dict[str, Any]] = Field(default_factory=list)
    completed_experiment_ids: list[str] = Field(default_factory=list)
    current_best_candidate: dict[str, Any] | None = None
    failed_requirements: list[dict[str, Any]] = Field(default_factory=list)
    current_diagnosis: str = ""
    current_hypothesis: str = ""
    proposed_action: dict[str, Any] | None = None
    last_tool_result: dict[str, Any] | None = None
    last_decision: dict[str, Any] | None = None
    adaptive_experiments_used: int = 0
    compute_budget_used: float = 0.0
    stagnation_count: int = 0
    pending_capability_request: dict[str, Any] | None = None
    pending_exclusion_proposal: dict[str, Any] | None = None
    stopping_reason: str = ""
    route: str = ""
    validation_passed: bool = False
    validation_errors: list[str] = Field(default_factory=list)
    acceptance_status: str = "unknown"
    development_view: dict[str, Any] | None = None
    sealed_test_result: dict[str, Any] | None = None
    agent_iteration: int = 0
    started_at: str = ""
    use_openai: bool = True
    openai_model: str = ""
    exclusion_decision: dict[str, Any] | None = None
    report_path: str = ""
    rankings: list[dict[str, Any]] = Field(default_factory=list)
    best_cv_r2_history: list[float] = Field(default_factory=list)
    model_families_used: list[str] = Field(default_factory=list)
    representation_changes_used: int = 0
    action_error: str = ""
    requirement_evaluation: dict[str, Any] | None = None

    model_config = {"extra": "allow"}
