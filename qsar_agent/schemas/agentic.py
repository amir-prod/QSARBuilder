"""Schemas for the agentic QSAR loop.

These types are the contract between the LLM planner and the deterministic
tools: every plan the Strategist emits is validated against them before any
model is trained, and every result the agents report back is one of them.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

TaskType = Literal["regression", "classification"]

SplitMethod = Literal["random", "stratified", "scaffold"]

FeatureBlock = Literal[
    "rdkit_descriptors",
    "morgan_fp",
    "morgan_counts",
    "maccs_keys",
    "rdkit_fp",
    "atom_pair_fp",
]

DiagnosisLabel = Literal[
    "accepted",
    "overfit",
    "underfit",
    "unstable_cv",
    "chance_correlation",
    "poor_generalisation",
    "narrow_applicability_domain",
    "class_imbalance",
    "insufficient_data",
]


class TaskDetection(BaseModel):
    """Outcome of inspecting the activity column."""

    task: TaskType
    reason: str
    n_unique_values: int
    n_classes: int | None = None
    class_counts: dict[str, int] = Field(default_factory=dict)
    positive_fraction: float | None = None
    imbalance_ratio: float | None = None


class FeatureRecipe(BaseModel):
    """Everything needed to reproduce a feature matrix from SMILES."""

    blocks: list[FeatureBlock] = Field(min_length=1)
    fingerprint_bits: int = 1024
    fingerprint_radius: int = 2
    variance_threshold: float = 0.0
    correlation_threshold: float | None = 0.95
    univariate_top_k: int | None = None
    scale_features: bool = True
    drop_ad_outliers: bool = False

    @field_validator("blocks")
    @classmethod
    def _unique_blocks(cls, value: list[str]) -> list[str]:
        seen: list[str] = []
        for block in value:
            if block not in seen:
                seen.append(block)
        return seen

    def signature(self) -> str:
        parts = [
            "+".join(self.blocks),
            f"bits={self.fingerprint_bits}",
            f"r={self.fingerprint_radius}",
            f"var={self.variance_threshold}",
            f"corr={self.correlation_threshold}",
            f"topk={self.univariate_top_k}",
            f"scale={int(self.scale_features)}",
            f"adout={int(self.drop_ad_outliers)}",
        ]
        return "|".join(parts)


class ModelPlan(BaseModel):
    """Estimator choice plus its hyperparameters."""

    estimator: str
    hyperparameters: dict[str, Any] = Field(default_factory=dict)

    def signature(self) -> str:
        params = ",".join(f"{k}={self.hyperparameters[k]}" for k in sorted(self.hyperparameters))
        return f"{self.estimator}({params})"


class IterationPlan(BaseModel):
    """A complete, executable plan for one iteration."""

    features: FeatureRecipe
    model: ModelPlan
    rationale: str = ""

    def signature(self) -> str:
        return f"{self.features.signature()}::{self.model.signature()}"


class CVSummary(BaseModel):
    """Cross-validation scores on the training set."""

    folds: int
    primary_metric: str
    mean_train_score: float
    mean_cv_score: float
    std_cv_score: float
    train_cv_gap: float
    fold_scores: list[float] = Field(default_factory=list)


class YScramblingResult(BaseModel):
    """y-randomization control against chance correlation."""

    n_repeats: int
    primary_metric: str
    real_cv_score: float
    scrambled_scores: list[float] = Field(default_factory=list)
    mean_scrambled_score: float
    std_scrambled_score: float
    max_scrambled_score: float
    margin: float
    fraction_scrambled_at_least_real: float


class ADSummary(BaseModel):
    """k-nearest-neighbour distance-to-model applicability domain."""

    method: str = "knn_distance"
    k: int = 5
    z_factor: float = 3.0
    threshold: float
    train_mean_distance: float
    test_in_domain_count: int
    test_total_count: int
    test_coverage_pct: float
    out_of_domain_ids: list[str] = Field(default_factory=list)


class ValidationReport(BaseModel):
    """Everything the ValidationAgent measured for one candidate model."""

    task: TaskType
    primary_metric: str
    n_train: int
    n_test: int
    n_features: int
    train_metrics: dict[str, float]
    test_metrics: dict[str, float]
    cv: CVSummary
    y_scrambling: YScramblingResult | None = None
    applicability_domain: ADSummary | None = None
    warnings: list[str] = Field(default_factory=list)


class CriterionCheck(BaseModel):
    """One acceptance criterion evaluated against an observed value."""

    name: str
    description: str
    observed: float | None
    threshold: float
    direction: Literal["min", "max"]
    passed: bool

    def render(self) -> str:
        symbol = ">=" if self.direction == "min" else "<="
        observed = "n/a" if self.observed is None else f"{self.observed:.4f}"
        flag = "PASS" if self.passed else "FAIL"
        return f"[{flag}] {self.name}: {observed} {symbol} {self.threshold:.4f}"


class AcceptanceVerdict(BaseModel):
    """Aggregate pass/fail over every enabled criterion."""

    accepted: bool
    checks: list[CriterionCheck] = Field(default_factory=list)

    @property
    def failed_checks(self) -> list[CriterionCheck]:
        return [c for c in self.checks if not c.passed]

    def summary(self) -> str:
        if self.accepted:
            return f"All {len(self.checks)} acceptance criteria passed."
        failed = self.failed_checks
        return f"{len(failed)} of {len(self.checks)} criteria failed: " + ", ".join(
            c.name for c in failed
        )


class FailureDiagnosis(BaseModel):
    """Why the candidate failed, in a form the Strategist can act on."""

    label: DiagnosisLabel
    explanation: str
    evidence: dict[str, float] = Field(default_factory=dict)
    suggested_actions: list[str] = Field(default_factory=list)


class PlanRevision(BaseModel):
    """The Strategist's proposed change for the next iteration."""

    stop: bool = False
    stop_reason: str = ""
    features: FeatureRecipe | None = None
    model: ModelPlan | None = None
    rationale: str = ""
    addresses_diagnosis: str = ""
    web_search_used: bool = False

    def to_plan(self, previous: IterationPlan) -> IterationPlan:
        return IterationPlan(
            features=self.features or previous.features,
            model=self.model or previous.model,
            rationale=self.rationale,
        )


class IterationRecord(BaseModel):
    """One full pass of the loop, kept for history and reporting."""

    index: int
    plan: IterationPlan
    validation: ValidationReport | None = None
    verdict: AcceptanceVerdict | None = None
    diagnosis: FailureDiagnosis | None = None
    revision: PlanRevision | None = None
    elapsed_seconds: float = 0.0
    error: str | None = None
    artifact_dir: str = ""

    def primary_score(self) -> float | None:
        if self.validation is None:
            return None
        return self.validation.test_metrics.get(self.validation.primary_metric)


class AgenticRunReport(BaseModel):
    """Final, honest account of the whole run."""

    run_id: str
    dataset_path: str
    task: TaskType
    task_detection: TaskDetection
    criteria: dict[str, Any]
    n_compounds: int
    n_train: int
    n_test: int
    split_method: SplitMethod
    iterations_run: int
    max_iterations: int
    accepted: bool
    stop_reason: str
    best_iteration_index: int | None = None
    best_validation: ValidationReport | None = None
    best_verdict: AcceptanceVerdict | None = None
    best_plan: IterationPlan | None = None
    iterations: list[IterationRecord] = Field(default_factory=list)
    llm_provider: str = ""
    llm_model: str = ""
    web_searches: int = 0
    total_elapsed_seconds: float = 0.0
    warnings: list[str] = Field(default_factory=list)
    artifact_paths: dict[str, str] = Field(default_factory=dict)
