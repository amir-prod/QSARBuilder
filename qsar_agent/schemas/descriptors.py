"""Mordred descriptor schemas."""

from __future__ import annotations

from pydantic import BaseModel, Field


class MordredCalculationResult(BaseModel):
    compound_count: int
    descriptor_count: int
    descriptors_with_missing: int
    failed_descriptor_values: int
    mordred_version: str
    rdkit_version: str
    enable_3d: bool
    raw_descriptors_path: str
    calculation_report_path: str
    failed_values_path: str | None = None
    warnings: list[str] = Field(default_factory=list)
