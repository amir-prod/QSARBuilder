"""Dataset validation schemas."""

from __future__ import annotations

from pydantic import BaseModel, Field


class ActivityStats(BaseModel):
    min: float | None = None
    max: float | None = None
    mean: float | None = None
    median: float | None = None
    std: float | None = None


class DatasetValidationResult(BaseModel):
    original_row_count: int
    valid_compound_count: int
    invalid_smiles_count: int
    missing_or_invalid_activity_count: int
    duplicate_compound_count: int
    activity_stats: ActivityStats
    cleaned_dataset_path: str
    invalid_rows_path: str | None = None
    duplicate_compounds_path: str | None = None
    validation_report_path: str
    warnings: list[str] = Field(default_factory=list)
