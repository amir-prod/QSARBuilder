"""Seal the external test set from the modeling-improvement agent."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from qsar_agent.schemas.agentic import PipelinePhase
from qsar_agent.schemas.handoff import (
    ErrorAnalysis,
    ExperimentRecord,
    ExternalTestMetrics,
    HandoffPackage,
    LargestErrorCompound,
    PlotReference,
    WinnerADResults,
)

TEST_PATH_MARKERS = (
    "test_set",
    "test_predictions",
    "preprocessed_test",
    "sealed_test",
    "_test_predictions",
)
TEST_SPLIT_NAMES = {"test", "external_test", "sealed_test"}


class SealedTestAccessError(PermissionError):
    """Raised when development-phase code tries to read sealed-test artifacts."""


def phase_value(phase: PipelinePhase | str | None) -> str:
    """Return the phase token. ``str(StrEnum)`` is not the value on Python 3.11."""
    if phase is None:
        return ""
    value = getattr(phase, "value", phase)
    return str(value)


def assert_development_phase(phase: PipelinePhase | str, *, sealed_test_result: Any = None) -> None:
    if phase_value(phase) != PipelinePhase.DEVELOPMENT.value:
        raise SealedTestAccessError(
            f"Development node refused to run in phase {phase}."
        )
    if sealed_test_result:
        raise SealedTestAccessError(
            "Development node refused state that already contains sealed-test results."
        )


def path_looks_like_test(path: str | Path | None) -> bool:
    if not path:
        return False
    lowered = str(path).replace("\\", "/").lower()
    if "/agent_results/sealed_test/" in lowered:
        return True
    return any(marker in lowered for marker in TEST_PATH_MARKERS)


def assert_no_test_paths(payload: Any, *, label: str = "payload") -> None:
    """Reject nested structures that reference sealed-test files or keys."""
    if payload is None:
        return
    if isinstance(payload, (str, Path)):
        text = str(payload)
        lowered = text.lower()
        if path_looks_like_test(text):
            raise SealedTestAccessError(f"{label} references a sealed-test path: {text}")
        if lowered in TEST_SPLIT_NAMES and "train" not in lowered:
            pass
        return
    if isinstance(payload, dict):
        for key, value in payload.items():
            key_l = str(key).lower()
            if key_l in {
                "external_test",
                "test_metrics",
                "test_r2",
                "test_rmse",
                "test_mae",
                "sealed_test_result",
                "external_test_r2",
            }:
                if isinstance(value, dict) and not any(
                    v not in (None, {}, [], "") for v in value.values()
                ):
                    continue
                if value not in (None, {}, [], ""):
                    raise SealedTestAccessError(f"{label} contains sealed-test field {key!r}.")
            assert_no_test_paths(value, label=f"{label}.{key}")
        return
    if isinstance(payload, (list, tuple)):
        for i, item in enumerate(payload):
            assert_no_test_paths(item, label=f"{label}[{i}]")


def _blank_plot_if_test(plot: PlotReference) -> PlotReference:
    if plot.relative_path and path_looks_like_test(plot.relative_path):
        return PlotReference(
            name=plot.name,
            status="unavailable",
            relative_path=None,
            reason="Sealed from the modeling-improvement agent (contains external-test observations).",
        )
    return plot


def development_view(package: HandoffPackage) -> HandoffPackage:
    """Return a copy of the handoff with all sealed-test outcomes removed."""
    experiments: list[ExperimentRecord] = []
    for exp in package.experiments:
        artifacts = exp.artifacts.model_copy(
            update={
                "test_predictions": None,
                "observed_vs_predicted": _blank_plot_if_test(exp.artifacts.observed_vs_predicted),
                "williams": PlotReference(
                    name="williams",
                    status="unavailable",
                    reason="Williams plot includes observed external-test values; sealed during development.",
                ),
                "residuals": PlotReference(
                    name="residuals",
                    status="unavailable",
                    reason="Residual plot includes observed external-test values; sealed during development.",
                ),
            }
        )
        ad = exp.applicability_domain
        if ad is not None:
            partitions = {
                k: v
                for k, v in (ad.outliers_by_partition or {}).items()
                if k not in TEST_SPLIT_NAMES
            }
            ad = ad.model_copy(update={"outliers_by_partition": partitions})
        experiments.append(
            exp.model_copy(
                update={
                    "external_test": ExternalTestMetrics(reported_after_selection=True),
                    "artifacts": artifacts,
                    "applicability_domain": ad,
                }
            )
        )

    err = package.error_analysis
    kept_errors: list[LargestErrorCompound] = [
        row for row in err.largest_error_compounds if row.split not in TEST_SPLIT_NAMES
    ]
    error_analysis = ErrorAnalysis(
        winner_run_id=err.winner_run_id,
        largest_error_compounds=kept_errors,
        target_range_performance=list(err.target_range_performance),
        inside_domain=err.inside_domain,
        outside_domain=err.outside_domain,
        residual_diagnostics=err.residual_diagnostics,
    )

    ad_winner = package.applicability_domain
    sealed_ad = WinnerADResults(
        winner_run_id=ad_winner.winner_run_id,
        method=ad_winner.method,
        warning_leverage=ad_winner.warning_leverage,
        residual_threshold=ad_winner.residual_threshold,
        structural_outlier_count=ad_winner.structural_outlier_count,
        response_outlier_count=ad_winner.response_outlier_count,
        structural_outlier_ids=list(ad_winner.structural_outlier_ids),
        response_outlier_ids=list(ad_winner.response_outlier_ids),
        outliers_by_partition={
            k: v
            for k, v in (ad_winner.outliers_by_partition or {}).items()
            if k not in TEST_SPLIT_NAMES
        },
        handling_decision=ad_winner.handling_decision,
        handling_justification=ad_winner.handling_justification,
    )

    return package.model_copy(
        update={
            "experiments": experiments,
            "error_analysis": error_analysis,
            "applicability_domain": sealed_ad,
        }
    )


def development_view_dict(package: HandoffPackage) -> dict[str, Any]:
    view = development_view(package)
    dumped = view.model_dump(mode="json")
    for exp in dumped.get("experiments", []):
        artifacts = exp.get("artifacts") or {}
        artifacts["test_predictions"] = None
        exp["artifacts"] = artifacts
        exp.pop("external_test", None)
    return dumped
