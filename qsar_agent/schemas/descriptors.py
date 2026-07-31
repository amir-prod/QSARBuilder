"""Descriptor calculation schemas."""

from __future__ import annotations

from pydantic import BaseModel, Field


class BackendInfo(BaseModel):
    name: str
    available: bool = True
    supports_3d: bool = False
    used: bool = True


class DescriptorCalculationResult(BaseModel):
    compound_count: int
    descriptor_count: int
    generated_descriptor_count: int
    external_descriptor_count: int = 0
    descriptors_with_missing: int = 0
    backends: list[str] = Field(default_factory=list)
    backends_detail: list[BackendInfo] = Field(default_factory=list)
    run_geometry_optimization: bool = False
    geometry_source: str = "rdkit_light_sdf_no_xtb"
    three_d_geometries_used: bool = False
    three_d_descriptors_included: bool = False
    generated_descriptor_columns: list[str] = Field(default_factory=list)
    rdkit_version: str = ""
    descjocky_version: str = ""
    raw_descriptors_path: str
    generated_descriptors_path: str = ""
    calculation_report_path: str
    calculation_report_md_path: str = ""
    external_descriptors_path: str | None = None
    warnings: list[str] = Field(default_factory=list)


# Backward-compatible alias for older imports/tests during migration.
MordredCalculationResult = DescriptorCalculationResult
