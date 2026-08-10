"""Pydantic schemas for the agentic model-improvement system."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


AllowedAction = Literal[
    "accept_model",
    "refine_hyperparameters",
    "reduce_feature_count",
    "expand_feature_count",
    "run_sfs_fixed_ga_expansion",
    "try_registered_estimator",
    "compare_registered_estimators",
    "recommend_unregistered_estimator",
    "request_model_dependency_approval",
    "request_user_approval",
    "request_user_input",
    "stop_budget_exhausted",
    "stop_no_viable_model",
    # Legacy alias accepted on deserialize only
    "try_fallback_estimator",
]

ExperimentKind = Literal[
    "initial_deterministic",
    "controlled_estimator_comparison",
    "full_pipeline_branch",
    "hyperparameter_refinement",
    "feature_count_change",
    "sfs_fixed_ga_expansion",
    "diagnostic_only",
    "stop",
]

ProjectStatus = Literal[
    "developing",
    "awaiting_approval",
    "model_locked",
    "external_evaluated",
    "completed",
    "stopped",
    "failed",
]

DecisionSource = Literal[
    "llm_agent",
    "deterministic_fallback",
    "deterministic_code",
    "user_approval",
]

ModelFamily = Literal[
    "linear",
    "kernel",
    "neighbor",
    "bagging",
    "boosting",
    "latent_variable",
    "neural",
    "graph",
]

HardFailureCondition = Literal[
    "external_test_access_attempted",
    "external_eval_before_lock",
    "agentic_after_external_access",
    "preprocessing_fit_on_non_train",
    "feature_selection_used_protected_targets",
    "hpo_used_protected_targets",
    "acceptance_criteria_failed",
    "invalid_action",
    "incompatible_estimator",
    "duplicate_experiment",
    "budget_exhausted",
    "missing_lock_prerequisites",
]


class MetricEvidence(BaseModel):
    name: str
    value: float | int | str | bool | None
    source_artifact: str
    source_field: str


class AgenticAcceptanceCriteria(BaseModel):
    minimum_mean_cv_r2: float = 0.60
    maximum_train_cv_gap: float = 0.15
    maximum_cv_r2_std: float = 0.15
    minimum_mean_train_r2: float | None = None
    require_non_overfit_status: bool = True
    require_validation_agent_approval: bool = True
    # Agent-validation gate (protected holdout within development data)
    minimum_agent_val_r2: float | None = None
    maximum_cv_agent_val_gap: float | None = 0.20


class AgenticImprovementConfig(BaseModel):
    enabled: bool = False
    max_cycles: int = 3
    max_total_experiments: int = 8
    max_specialist_calls_per_cycle: int = 2
    max_retries_per_agent_call: int = 1
    require_approval_for_data_changes: bool = True
    prevent_external_test_access: bool = True
    model: str = ""
    acceptance: AgenticAcceptanceCriteria = Field(default_factory=AgenticAcceptanceCriteria)
    practical_equivalence_tolerance: float = 0.01
    agent_validation_fraction: float = 0.20
    max_screen_models: int = 5
    max_optimize_after_screen: int = 2
    # Hard caps (enforced regardless of UI)
    hard_max_cycles: int = 10
    hard_max_experiments: int = 20
    hard_max_retries: int = 2
    hard_max_candidates_per_grid: int = 60


class ModelSpecification(BaseModel):
    estimator_name: str
    display_name: str
    family: ModelFamily
    import_path: str
    available: bool
    missing_dependency: str | None = None
    supports_dense: bool = True
    supports_sparse: bool = False
    requires_scaling: bool = False
    supports_multioutput: bool = False
    minimum_training_samples: int | None = None
    maximum_recommended_features_ratio: float | None = None
    default_parameters: dict[str, Any] = Field(default_factory=dict)
    bounded_hpo_space: dict[str, list[Any]] = Field(default_factory=dict)
    parameter_constraints: dict[str, Any] = Field(default_factory=dict)
    expected_strengths: list[str] = Field(default_factory=list)
    expected_limitations: list[str] = Field(default_factory=list)
    computational_cost: Literal["low", "medium", "high"] = "medium"
    interpretability_level: Literal["high", "medium", "low"] = "medium"
    deterministic_with_seed: bool = True


class ModelRecommendation(BaseModel):
    estimator_name: str
    is_registered: bool
    model_family: str
    rationale: str
    evidence: list[MetricEvidence] = Field(default_factory=list)
    expected_advantage: str = ""
    expected_limitation: str = ""
    preprocessing_requirements: list[str] = Field(default_factory=list)
    estimated_cost: str = "medium"
    priority: int = 1
    requires_dependency_approval: bool = False
    missing_dependency: str | None = None


class ModelCompatibilityResult(BaseModel):
    estimator_name: str
    compatible: bool
    blocking_reasons: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    required_preprocessing: list[str] = Field(default_factory=list)
    estimated_candidate_count: int = 0
    estimated_cost: str = "medium"


class AgentDiagnosis(BaseModel):
    agent_name: str
    experiment_id: str
    failure_category: str
    summary: str
    evidence: list[MetricEvidence] = Field(default_factory=list)
    hypotheses: list[str] = Field(default_factory=list)
    confidence: float = 0.5
    warnings: list[str] = Field(default_factory=list)
    decision_source: DecisionSource = "llm_agent"
    recommended_actions: list[AllowedAction] = Field(default_factory=list)


class ExperimentProposal(BaseModel):
    proposal_id: str
    parent_experiment_id: str
    proposed_by: str
    hypothesis: str
    action: AllowedAction
    configuration_changes: dict[str, Any] = Field(default_factory=dict)
    expected_effect: str = ""
    scientific_rationale: str = ""
    estimated_cost: str = "medium"
    requires_user_approval: bool = False
    approval_reason: str | None = None
    duplicate_check_key: str = ""
    experiment_kind: ExperimentKind = "controlled_estimator_comparison"
    multi_component: bool = False
    component_list: list[str] = Field(default_factory=list)
    decision_source: DecisionSource = "llm_agent"


class SupervisorDecision(BaseModel):
    cycle_index: int
    selected_proposal_id: str | None = None
    action: AllowedAction
    rationale: str
    evidence_considered: list[MetricEvidence] = Field(default_factory=list)
    rejected_proposals: list[dict[str, Any]] = Field(default_factory=list)
    stopping_reason: str | None = None
    specialists_consulted: list[str] = Field(default_factory=list)
    decision_source: DecisionSource = "llm_agent"


class AcceptanceResult(BaseModel):
    accepted: bool
    evidence: list[MetricEvidence] = Field(default_factory=list)
    failed_criteria: list[str] = Field(default_factory=list)
    explanation: str = ""
    overfit_status: str | None = None


class ValidationReview(BaseModel):
    approved: bool
    hard_veto: bool = False
    hard_failure_conditions: list[HardFailureCondition] = Field(default_factory=list)
    soft_rejection_recommended: bool = False
    warnings: list[str] = Field(default_factory=list)
    additional_validation_proposals: list[str] = Field(default_factory=list)
    evidence: list[MetricEvidence] = Field(default_factory=list)
    summary: str = ""
    decision_source: DecisionSource = "deterministic_code"


class ModelLockRecord(BaseModel):
    locked_experiment_id: str
    configuration_hash: str
    locked_at: str
    selection_rationale: str
    selection_record: dict[str, Any] = Field(default_factory=dict)
    dataset_hash: str = ""
    cv_folds_hash: str = ""


class ExperimentRecord(BaseModel):
    experiment_id: str
    parent_experiment_id: str | None = None
    cycle_index: int = 0
    hypothesis: str = ""
    action: AllowedAction | str = "accept_model"
    config_snapshot: dict[str, Any] = Field(default_factory=dict)
    status: str = "completed"
    artifact_directory: str = ""
    internal_metrics: dict[str, Any] = Field(default_factory=dict)
    comparison_to_parent: dict[str, Any] = Field(default_factory=dict)
    conclusion: str = ""
    created_at: str = ""
    dataset_hash: str = ""
    external_test_accessed: bool = False
    experiment_kind: ExperimentKind = "initial_deterministic"
    multi_component: bool = False
    component_list: list[str] = Field(default_factory=list)
    cv_folds_hash: str | None = None
    reused_artifacts: list[str] = Field(default_factory=list)
    newly_generated_artifacts: list[str] = Field(default_factory=list)
    random_seed: int = 42
    decision_source: DecisionSource = "deterministic_code"
    configuration_hash: str = ""
    estimator: str | None = None
    selected_features: list[str] = Field(default_factory=list)
    feature_count: int | None = None


class AgenticProjectState(BaseModel):
    project_id: str
    initial_run_id: str
    current_experiment_id: str
    best_experiment_id: str
    cycle_index: int = 0
    experiment_count: int = 0
    status: ProjectStatus = "developing"
    acceptance_criteria: AgenticAcceptanceCriteria = Field(
        default_factory=AgenticAcceptanceCriteria
    )
    external_test_locked: bool = True
    pending_approval: dict[str, Any] | None = None
    locked_experiment_id: str | None = None
    lock_record: ModelLockRecord | None = None
    external_test_accessed: bool = False
    cv_folds_hash: str | None = None
    agent_dev_indices_path: str | None = None
    agent_val_indices_path: str | None = None
    stop_requested: bool = False
    stopping_reason: str | None = None
    last_acceptance: AcceptanceResult | None = None
    last_validation_review: ValidationReview | None = None


class AgentVisibleSummary(BaseModel):
    experiment_id: str
    dataset_size: int | None = None
    validation_counts: dict[str, Any] = Field(default_factory=dict)
    development_split_size: int | None = None
    agent_dev_size: int | None = None
    agent_val_size: int | None = None
    descriptor_count_before: int | None = None
    descriptor_count_after: int | None = None
    feature_count: int | None = None
    selected_feature_names: list[str] = Field(default_factory=list)
    samples_per_feature_ratio: float | None = None
    mean_train_r2: float | None = None
    mean_cv_r2: float | None = None
    cv_r2_std: float | None = None
    train_cv_gap: float | None = None
    mean_cv_rmse: float | None = None
    mean_cv_mae: float | None = None
    agent_val_r2: float | None = None
    overfitting_status: str | None = None
    overfitting_acceptable: bool | None = None
    hpo_rounds: int | None = None
    best_parameters: dict[str, Any] = Field(default_factory=dict)
    estimator: str | None = None
    model_complexity_summary: str | None = None
    sfs_summary: dict[str, Any] = Field(default_factory=dict)
    ga_summary: dict[str, Any] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
    source_artifact_paths: dict[str, str] = Field(default_factory=dict)
    external_test_unavailable: bool = True
    external_test_statement: str = (
        "External-test information is unavailable during agentic development. "
        "Agents must not request or use external-test metrics, predictions, "
        "residuals, applicability-domain classifications, or scatter plots."
    )
    cv_folds_hash: str | None = None
    experiment_kind: ExperimentKind | None = None


class AgentEvent(BaseModel):
    timestamp: str
    cycle_index: int | None = None
    experiment_id: str | None = None
    agent: str | None = None
    event_type: str
    input_artifact_refs: list[str] = Field(default_factory=list)
    validated_response: dict[str, Any] | None = None
    selected_action: str | None = None
    tool_execution: dict[str, Any] | None = None
    error: str | None = None
    retry_or_fallback: str | None = None
    approval_state: str | None = None
    token_usage: dict[str, Any] | None = None
    decision_source: DecisionSource | None = None


LEGACY_ACTION_ALIASES: dict[str, AllowedAction] = {
    "try_fallback_estimator": "try_registered_estimator",
}


def normalize_action(action: str) -> AllowedAction:
    mapped = LEGACY_ACTION_ALIASES.get(action, action)
    return mapped  # type: ignore[return-value]
