"""Structured modeling handoff package — single source of truth for MD/JSON/CSV."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from qsar_agent.schemas.hyperparameter_optimization import FoldMetrics
from qsar_agent.schemas.modeling import Metrics


PlotStatus = Literal["available", "unavailable"]
ExperimentStatus = Literal["completed", "failed", "skipped"]
WorkflowStatusValue = Literal["completed", "failed", "cancelled", "pending", "running"]


class PlotReference(BaseModel):
    name: str
    status: PlotStatus
    relative_path: str | None = None
    reason: str = ""


class GitProvenance(BaseModel):
    available: bool
    commit: str = ""
    dirty: bool = False
    reason: str = ""


class RandomSeeds(BaseModel):
    workflow: int
    sfs: int
    ga: int
    hpo: int
    model: int
    clustering: int


class StageStatusRecord(BaseModel):
    stage: str
    status: str
    message: str = ""


class RunMetadata(BaseModel):
    run_id: str
    started_at: str
    completed_at: str
    git: GitProvenance
    random_seeds: RandomSeeds
    configuration: dict[str, Any] = Field(default_factory=dict)
    package_versions: dict[str, str] = Field(default_factory=dict)
    workflow_status: str
    stages: list[StageStatusRecord] = Field(default_factory=list)


class AcceptanceCriteria(BaseModel):
    primary_metric: str
    minimum_cv_r2: float
    overfit_gap_threshold: float
    severe_overfit_gap_threshold: float
    cv_std_threshold: float
    minimum_train_r2: float
    min_cv_improvement: float


class ProblemDefinition(BaseModel):
    task: str
    target: str
    target_transformation: str
    units: str
    primary_metric: str
    acceptance_criteria: AcceptanceCriteria


class CurationStep(BaseModel):
    step: str
    n_compounds: int | None = None
    n_features: int | None = None
    n_removed: int | None = None
    notes: str = ""


class DuplicateOverlap(BaseModel):
    train_val: list[str] = Field(default_factory=list)
    train_test: list[str] = Field(default_factory=list)
    val_test: list[str] = Field(default_factory=list)
    any_overlap: bool = False


class DatasetAudit(BaseModel):
    curation_steps: list[CurationStep] = Field(default_factory=list)
    invalid_structures: int
    duplicates: int
    missing_or_invalid_activity: int
    descriptors_with_missing: int
    target_statistics: dict[str, float | None] = Field(default_factory=dict)
    feature_counts: dict[str, int] = Field(default_factory=dict)
    train_size: int
    validation_size: int
    test_size: int
    split_strategy: str
    dataset_hash: str
    duplicate_overlap: DuplicateOverlap


class LeakageSafeguards(BaseModel):
    test_lock_status: str
    test_compound_id_hash: str
    preprocessing_scope: str
    feature_selection_scope: str
    duplicate_overlap: DuplicateOverlap
    test_results_used_for_selection: bool
    selection_criterion: str
    confirmation: str
    selection_records: list[dict[str, Any]] = Field(default_factory=list)


class RepresentationPreprocessing(BaseModel):
    descriptor_backends: list[str] = Field(default_factory=list)
    fingerprint_settings: dict[str, Any] = Field(default_factory=dict)
    geometry_optimization: bool = False
    three_d_descriptors_included: bool = False
    filters: dict[str, float] = Field(default_factory=dict)
    scaling: str
    imputation: str
    pipeline_order: list[str] = Field(default_factory=list)
    preprocessor_relative_path: str | None = None


class ValidationDesign(BaseModel):
    cv_method: str
    folds: int
    repeats: int
    shuffle: bool
    seed: int
    tuning_method: str
    search_budget: int
    optimization_metric: str
    combined_score_description: str


class ArtifactIndex(BaseModel):
    observed_vs_predicted: PlotReference
    williams: PlotReference
    residuals: PlotReference
    cv_predictions: str | None = None
    test_predictions: str | None = None
    config: str | None = None
    pipeline: str | None = None
    source_predictions: str = ""
    source_model: str = ""


class ExperimentMetrics(BaseModel):
    train_r2: float | None = None
    train_rmse: float | None = None
    train_mae: float | None = None
    train_n: int | None = None
    cv_r2: float | None = None
    cv_rmse: float | None = None
    cv_mae: float | None = None
    cv_r2_std: float | None = None
    train_cv_r2_gap: float | None = None
    val_r2: float | None = None
    val_rmse: float | None = None
    val_mae: float | None = None
    val_n: int | None = None


class ExternalTestMetrics(BaseModel):
    reported_after_selection: bool = True
    r2: float | None = None
    rmse: float | None = None
    mae: float | None = None
    n: int | None = None


class DiagnosticFlags(BaseModel):
    status: str = ""
    is_acceptable: bool | None = None
    is_overfit: bool | None = None
    is_underfit: bool | None = None
    is_unstable: bool | None = None
    is_severe_overfit: bool | None = None


class ExperimentAD(BaseModel):
    method: str
    warning_leverage: float | None = None
    residual_threshold: float | None = None
    structural_outlier_ids: list[str] = Field(default_factory=list)
    response_outlier_ids: list[str] = Field(default_factory=list)
    structural_outlier_count: int = 0
    response_outlier_count: int = 0
    outliers_by_partition: dict[str, dict[str, list[str]]] = Field(default_factory=dict)
    handling_decision: str
    handling_justification: str


class ExperimentRecord(BaseModel):
    run_id: str
    representation: str
    feature_selection_method: str
    model: str
    hyperparameters: dict[str, Any] = Field(default_factory=dict)
    feature_count: int
    selected_feature_names: list[str] = Field(default_factory=list)
    metrics: ExperimentMetrics
    external_test: ExternalTestMetrics = Field(default_factory=ExternalTestMetrics)
    per_fold_scores: list[FoldMetrics] = Field(default_factory=list)
    diagnostic_flags: DiagnosticFlags = Field(default_factory=DiagnosticFlags)
    warnings: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    failure_reason: str = ""
    runtime_seconds: float | None = None
    status: ExperimentStatus
    artifacts: ArtifactIndex
    applicability_domain: ExperimentAD | None = None
    is_winner: bool = False


class LargestErrorCompound(BaseModel):
    compound_id: str
    split: str
    activity: float
    predicted_activity: float
    residual: float
    abs_residual: float
    applicability_domain: str = ""


class RangePerformance(BaseModel):
    range_label: str
    n: int
    r2: float | None = None
    rmse: float | None = None
    mae: float | None = None
    activity_min: float | None = None
    activity_max: float | None = None


class DomainPerformance(BaseModel):
    n: int
    r2: float | None = None
    rmse: float | None = None
    mae: float | None = None


class ResidualDiagnostics(BaseModel):
    mean: float | None = None
    std: float | None = None
    residual_vs_predicted_correlation: float | None = None


class ErrorAnalysis(BaseModel):
    winner_run_id: str
    largest_error_compounds: list[LargestErrorCompound] = Field(default_factory=list)
    target_range_performance: list[RangePerformance] = Field(default_factory=list)
    inside_domain: DomainPerformance
    outside_domain: DomainPerformance
    residual_diagnostics: ResidualDiagnostics


class WinnerADResults(BaseModel):
    winner_run_id: str
    method: str
    warning_leverage: float | None = None
    residual_threshold: float
    structural_outlier_count: int
    response_outlier_count: int
    structural_outlier_ids: list[str] = Field(default_factory=list)
    response_outlier_ids: list[str] = Field(default_factory=list)
    outliers_by_partition: dict[str, dict[str, list[str]]] = Field(default_factory=dict)
    handling_decision: str
    handling_justification: str


class CompletedSearch(BaseModel):
    name: str
    completed: bool
    detail: str = ""


class WorkflowConclusion(BaseModel):
    best_run_id: str
    selection_criterion: str
    acceptance_status: bool
    failed_criteria: list[str] = Field(default_factory=list)
    completed_searches: list[CompletedSearch] = Field(default_factory=list)
    winner_model: str = ""
    winner_train_metrics: Metrics | None = None
    winner_cv_r2: float | None = None
    winner_val_metrics: Metrics | None = None


class AgentConstraints(BaseModel):
    permitted_actions: list[str] = Field(default_factory=list)
    prohibited_actions: list[str] = Field(default_factory=list)
    iteration_budget: dict[str, int] = Field(default_factory=dict)
    compute_budget: dict[str, int] = Field(default_factory=dict)
    approval_required_actions: list[str] = Field(default_factory=list)
    stopping_conditions: list[str] = Field(default_factory=list)


class HandoffPackage(BaseModel):
    """Single structured results object for the modeling handoff."""

    run_metadata: RunMetadata
    problem_definition: ProblemDefinition
    dataset_audit: DatasetAudit
    leakage_safeguards: LeakageSafeguards
    representation_preprocessing: RepresentationPreprocessing
    validation_design: ValidationDesign
    experiments: list[ExperimentRecord] = Field(default_factory=list)
    applicability_domain: WinnerADResults
    error_analysis: ErrorAnalysis
    conclusion: WorkflowConclusion
    agent_constraints: AgentConstraints
