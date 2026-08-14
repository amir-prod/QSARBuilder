"""Hyperparameter optimization schemas."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


OverfittingStatus = Literal[
    "good",
    "overfit",
    "underfit",
    "unstable",
    "poor_performance",
]


class OverfittingAssessment(BaseModel):
    status: OverfittingStatus
    is_acceptable: bool
    is_overfit: bool
    is_underfit: bool
    is_unstable: bool
    is_severe_overfit: bool = False
    mean_train_r2: float
    mean_cv_r2: float
    train_cv_r2_gap: float
    cv_r2_std: float
    warnings: list[str] = Field(default_factory=list)
    explanation: str


class OverfittingThresholds(BaseModel):
    overfit_gap_threshold: float = 0.15
    severe_overfit_gap_threshold: float = 0.25
    minimum_cv_r2: float = 0.50
    cv_std_threshold: float = 0.15
    minimum_train_r2: float = 0.40


class HPOConfig(BaseModel):
    enabled: bool = True
    max_hpo_rounds: int = 3
    cv_folds: int = 5
    max_candidates_per_round: int = 120
    min_cv_improvement: float = 0.02
    random_seed: int = 42
    n_jobs: int = -1
    use_randomized_search_fallback: bool = True
    openai_model: str = ""
    thresholds: OverfittingThresholds = Field(default_factory=OverfittingThresholds)


class FoldMetrics(BaseModel):
    fold: int
    train_r2: float
    val_r2: float
    train_rmse: float
    val_rmse: float
    train_mae: float
    val_mae: float


class CVSummary(BaseModel):
    mean_train_r2: float
    mean_cv_r2: float
    std_cv_r2: float
    mean_train_rmse: float
    mean_cv_rmse: float
    mean_train_mae: float
    mean_cv_mae: float
    train_cv_r2_gap: float
    n_folds: int
    holdout_val_r2: float | None = None


class BaselineCVResult(BaseModel):
    fold_metrics: list[FoldMetrics]
    summary: CVSummary
    fold_metrics_path: str
    summary_path: str


class AgentGridProposal(BaseModel):
    round_index: int
    reasoning_summary: str
    search_strategy: str
    proposed_grid: dict[str, list[Any]]
    expected_effect_on_overfitting: str
    expected_effect_on_underfitting: str
    computational_cost_estimate: str
    warnings: list[str] = Field(default_factory=list)


class GridSanitizationResult(BaseModel):
    original_grid: dict[str, list[Any]]
    sanitized_grid: dict[str, list[Any]]
    removed_params: list[str] = Field(default_factory=list)
    removed_values: dict[str, list[Any]] = Field(default_factory=dict)
    shrink_steps: list[str] = Field(default_factory=list)
    candidate_count: int
    used_randomized_search: bool = False
    warnings: list[str] = Field(default_factory=list)


class CandidateResult(BaseModel):
    rank: int
    params: dict[str, Any]
    mean_train_r2: float
    mean_cv_r2: float
    std_cv_r2: float
    mean_train_rmse: float
    mean_cv_rmse: float
    mean_train_mae: float
    mean_cv_mae: float
    train_cv_r2_gap: float
    is_best: bool = False


class HPORoundResult(BaseModel):
    round_index: int
    agent_proposal: AgentGridProposal | None = None
    sanitization: GridSanitizationResult
    candidates: list[CandidateResult]
    best_params: dict[str, Any]
    best_cv_summary: CVSummary
    assessment: OverfittingAssessment
    candidates_searched: int
    agent_grid_path: str = ""
    agent_explanation_path: str = ""
    search_results_path: str = ""
    best_params_path: str = ""
    cv_summary_path: str = ""
    assessment_path: str = ""
    performance_png_path: str = ""
    performance_svg_path: str = ""
    grid_sanitization_path: str = ""


ModelSource = Literal[
    "baseline",
    "hpo_round_1",
    "hpo_round_2",
    "hpo_round_3",
]


class FinalModelSelection(BaseModel):
    source: ModelSource
    params: dict[str, Any]
    cv_summary: CVSummary
    assessment: OverfittingAssessment
    selection_rationale: str
    warning: str = ""
    compared_candidates: list[dict[str, Any]] = Field(default_factory=list)


class HPOResult(BaseModel):
    enabled: bool
    rounds_completed: int
    max_rounds: int
    baseline_cv: BaselineCVResult | None = None
    baseline_assessment: OverfittingAssessment | None = None
    baseline_assessment_path: str = ""
    final_assessment: OverfittingAssessment | None = None
    final_assessment_path: str = ""
    rounds: list[HPORoundResult] = Field(default_factory=list)
    final_selection: FinalModelSelection | None = None
    hpo_triggered: bool = False
    hpo_trigger_reason: str = ""
    iteration_log_json_path: str = ""
    iteration_log_md_path: str = ""
    final_selection_json_path: str = ""
    final_selection_explanation_path: str = ""
    all_rounds_summary_path: str = ""
    summary_plot_png_path: str = ""
    summary_plot_svg_path: str = ""
    summary_csv_path: str = ""
    agent_fallback_log_path: str = ""
    final_model_config: dict[str, Any] = Field(default_factory=dict)
