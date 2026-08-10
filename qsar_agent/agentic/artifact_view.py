"""Allowlisted agent-visible artifact view (no raw filesystem access for agents)."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from qsar_agent.schemas.agentic import AgentVisibleSummary


# Paths/fields agents must never receive
DENIED_ARTIFACT_BASENAMES = frozenset(
    {
        "predictions.csv",
        "model_metrics.json",
        "prediction_scatter.png",
        "prediction_scatter.svg",
        "williams_plot.png",
        "williams_plot.svg",
        "applicability_domain.csv",
        "applicability_domain.json",
        "branch_external_artifacts.json",
        "preprocessed_test_descriptors.csv",
        "test_set_raw_descriptors.csv",
    }
)

DENIED_FIELD_NAMES = frozenset(
    {
        "test_r2",
        "test_rmse",
        "test_mae",
        "external_test_r2",
        "external_r2",
        "test_predictions",
        "test_residuals",
        "ad_test_outliers",
        "branch_external_artifacts",
    }
)


class AgentArtifactView(BaseModel):
    """Compact allowlisted view passed to agents instead of a run directory path."""

    experiment_id: str
    summary: AgentVisibleSummary
    approved_artifact_paths: dict[str, str] = Field(default_factory=dict)
    ledger_digest: list[dict[str, Any]] = Field(default_factory=list)
    acceptance_criteria: dict[str, Any] = Field(default_factory=dict)
    budget: dict[str, Any] = Field(default_factory=dict)
    notes: list[str] = Field(default_factory=list)

    def to_agent_payload(self) -> dict[str, Any]:
        payload = self.model_dump()
        assert_no_external_test_fields(payload)
        return payload


def is_denied_path(path: str) -> bool:
    name = path.replace("\\", "/").split("/")[-1].lower()
    return name in {n.lower() for n in DENIED_ARTIFACT_BASENAMES}


_ALLOWED_EXTERNAL_TEST_META_KEYS = frozenset(
    {
        "external_test_unavailable",
        "external_test_statement",
        "external_test_locked",
        "prevent_external_test_access",
        "external_test_accessed",
        "external_test_access_attempted",
        "external_eval_before_lock",
        "deterministic_hard_failure_flags",
    }
)


def assert_no_external_test_fields(payload: Any, path: str = "root") -> None:
    """Raise if external-test metric/result fields appear in an agent payload."""
    if isinstance(payload, dict):
        for key, value in payload.items():
            key_l = str(key).lower()
            if key_l in _ALLOWED_EXTERNAL_TEST_META_KEYS:
                assert_no_external_test_fields(value, f"{path}.{key}")
                continue
            if key_l in DENIED_FIELD_NAMES or key_l in (
                "test_r2",
                "test_rmse",
                "test_mae",
                "external_test_r2",
                "external_r2",
                "test_predictions",
                "test_residuals",
            ):
                raise ValueError(f"Denied external-test field in agent payload at {path}.{key}")
            if isinstance(value, str) and is_denied_path(value):
                raise ValueError(f"Denied external-test artifact path at {path}.{key}: {value}")
            assert_no_external_test_fields(value, f"{path}.{key}")
    elif isinstance(payload, list):
        for i, item in enumerate(payload):
            assert_no_external_test_fields(item, f"{path}[{i}]")


def filter_approved_paths(paths: dict[str, str]) -> dict[str, str]:
    return {k: v for k, v in paths.items() if not is_denied_path(v)}
