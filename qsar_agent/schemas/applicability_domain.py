"""Applicability domain schemas."""

from __future__ import annotations

from pydantic import BaseModel, Field


class ApplicabilityDomainSummary(BaseModel):
    train_in_domain_count: int
    train_in_domain_pct: float
    test_in_domain_count: int
    test_in_domain_pct: float
    warning_leverage: float
    residual_threshold: float = 3.0
    high_leverage_ids: list[str] = Field(default_factory=list)
    response_outlier_ids: list[str] = Field(default_factory=list)


class ApplicabilityDomainResult(BaseModel):
    summary: ApplicabilityDomainSummary
    classifications_path: str
    report_path: str
    williams_png_path: str
    williams_svg_path: str
