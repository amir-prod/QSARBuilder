"""Workflow state schemas."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from qsar_agent.schemas.applicability_domain import ApplicabilityDomainSummary
from qsar_agent.schemas.modeling import Metrics


class StageStatus(str, Enum):
    PENDING = "Pending"
    RUNNING = "Running"
    COMPLETED = "Completed"
    FAILED = "Failed"
    CANCELLED = "Cancelled"


WORKFLOW_STAGES = [
    "dataset_validation",
    "mordred_descriptors",
    "umap_split",
    "descriptor_preprocessing",
    "sequential_feature_selection",
    "feature_count_selection",
    "genetic_algorithm",
    "final_model",
    "applicability_domain",
]


class StageInfo(BaseModel):
    stage: str
    status: StageStatus = StageStatus.PENDING
    message: str = ""


class AgentFinalReport(BaseModel):
    run_id: str
    dataset_size: int
    valid_compounds: int
    train_size: int
    test_size: int
    initial_mordred_descriptors: int
    final_preprocessed_descriptors: int
    selected_descriptor_count: int
    ga_selected_descriptors: list[str]
    train_metrics: Metrics
    test_metrics: Metrics
    applicability_domain_summary: ApplicabilityDomainSummary
    warnings: list[str] = Field(default_factory=list)
    artifact_paths: dict[str, str] = Field(default_factory=dict)
    agent_explanation: str = ""


class WorkflowState(BaseModel):
    run_id: str
    stages: list[StageInfo] = Field(default_factory=list)
    logs: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    artifact_paths: dict[str, str] = Field(default_factory=dict)
    final_report: AgentFinalReport | None = None
    config_snapshot: dict[str, Any] = Field(default_factory=dict)
    zip_path: str | None = None

    @classmethod
    def create(cls, run_id: str, config_snapshot: dict[str, Any]) -> "WorkflowState":
        stages = [StageInfo(stage=s) for s in WORKFLOW_STAGES]
        return cls(run_id=run_id, stages=stages, config_snapshot=config_snapshot)

    def set_stage_status(self, stage: str, status: StageStatus, message: str = "") -> None:
        for s in self.stages:
            if s.stage == stage:
                s.status = status
                s.message = message
                break
