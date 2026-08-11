"""Schemas for read-only post-external-test audit."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


PrimaryAuditOutcome = Literal[
    "external_validation_passed",
    "external_validation_failed",
]

DiagnosticFlag = Literal[
    "external_distribution_shift",
    "external_metric_unstable",
    "applicability_domain_failure",
    "internal_validation_mismatch",
    "small_external_test",
    "influential_compounds_detected",
]


class PostTestAuditCriteria(BaseModel):
    """Frozen before external unlock; must not change after seeing test metrics."""

    maximum_cv_test_r2_gap: float = 0.20
    minimum_external_r2: float = 0.50
    minimum_ad_coverage: float = 0.70
    minimum_bootstrap_n: int = 20
    minimum_subgroup_n: int = 5
    residual_outlier_z: float = 3.0
    bootstrap_samples: int = 1000
    bootstrap_seed: int = 42
    confidence_level: float = 0.95


class BootstrapCI(BaseModel):
    metric: str
    estimate: float | None = None
    lower: float | None = None
    upper: float | None = None
    n_samples: int = 0
    n_bootstrap: int = 0
    available: bool = True
    warning: str | None = None


class MetricBlock(BaseModel):
    mean_r2: float | None = None
    rmse: float | None = None
    mae: float | None = None
    n: int | None = None
    source: str = ""


class SubgroupMetrics(BaseModel):
    label: str
    n: int
    r2: float | None = None
    rmse: float | None = None
    mae: float | None = None
    reliable: bool = True
    warning: str | None = None


class PostTestAuditResult(BaseModel):
    primary_outcome: PrimaryAuditOutcome
    diagnostic_flags: list[DiagnosticFlag] = Field(default_factory=list)
    criteria: PostTestAuditCriteria
    criteria_snapshot_path: str = ""
    train_metrics: MetricBlock = Field(default_factory=MetricBlock)
    cv_metrics: MetricBlock = Field(default_factory=MetricBlock)
    agent_val_metrics: MetricBlock = Field(default_factory=MetricBlock)
    external_metrics: MetricBlock = Field(default_factory=MetricBlock)
    train_cv_r2_gap: float | None = None
    cv_test_r2_gap: float | None = None
    bootstrap_cis: list[BootstrapCI] = Field(default_factory=list)
    external_n: int = 0
    external_activity_min: float | None = None
    external_activity_max: float | None = None
    external_activity_variance: float | None = None
    residual_outlier_ids: list[str] = Field(default_factory=list)
    influential_compound_ids: list[str] = Field(default_factory=list)
    ad_coverage: float | None = None
    in_domain_metrics: SubgroupMetrics | None = None
    out_of_domain_metrics: SubgroupMetrics | None = None
    random_vs_grouped_cv: dict[str, Any] = Field(
        default_factory=lambda: {
            "available": False,
            "status": "unavailable",
            "recommendation": (
                "Grouped/cluster-aware CV was not generated before model lock. "
                "Recommend generating it in a future development lineage before locking."
            ),
        }
    )
    evidence_paths: dict[str, str] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
    recommendations: list[str] = Field(default_factory=list)
    failed_criteria: list[str] = Field(default_factory=list)
    remediation_allowed: bool = False
    explanation: str = ""
