"""Preprocessing schemas."""

from __future__ import annotations

from pydantic import BaseModel, Field


class PreprocessingResult(BaseModel):
    initial_descriptor_count: int
    removed_nonnumeric: int
    removed_missing: int
    removed_constant: int
    removed_near_constant: int
    removed_correlated: int
    final_descriptor_count: int
    missing_value_threshold: float
    near_constant_threshold: float
    correlation_threshold: float
    preprocessed_train_path: str
    preprocessed_test_path: str
    preprocessor_path: str
    preprocessing_report_path: str
    removed_descriptors_path: str
    retained_descriptors_path: str
    warnings: list[str] = Field(default_factory=list)
