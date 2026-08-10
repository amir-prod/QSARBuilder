"""Build compact agent-visible summaries from allowlisted artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from qsar_agent.agentic.artifact_view import filter_approved_paths
from qsar_agent.agentic.ledger import experiment_dir, save_json
from qsar_agent.models.registry import model_simplicity_score
from qsar_agent.schemas.agentic import AgentVisibleSummary, ExperimentKind


def build_agent_visible_summary(
    *,
    experiment_id: str,
    run_dir: Path,
    internal_metrics: dict[str, Any],
    source_paths: dict[str, str] | None = None,
    estimator: str | None = None,
    selected_features: list[str] | None = None,
    experiment_kind: ExperimentKind | None = None,
    cv_folds_hash: str | None = None,
    warnings: list[str] | None = None,
    extra: dict[str, Any] | None = None,
) -> AgentVisibleSummary:
    """Deterministically assemble a compact summary; never includes external-test fields."""
    source_paths = filter_approved_paths(source_paths or {})
    features = list(selected_features or internal_metrics.get("selected_features") or [])
    feature_count = internal_metrics.get("feature_count", len(features) or None)
    agent_dev_size = internal_metrics.get("agent_dev_size")
    ratio = None
    if agent_dev_size and feature_count:
        ratio = float(agent_dev_size) / float(feature_count)

    complexity = None
    params = internal_metrics.get("best_parameters") or {}
    if estimator:
        try:
            complexity = f"simplicity_score={model_simplicity_score(estimator, params):.3f}"
        except Exception:
            complexity = estimator

    extra = extra or {}
    summary = AgentVisibleSummary(
        experiment_id=experiment_id,
        dataset_size=internal_metrics.get("dataset_size"),
        validation_counts=internal_metrics.get("validation_counts") or {},
        development_split_size=internal_metrics.get("development_split_size"),
        agent_dev_size=agent_dev_size,
        agent_val_size=internal_metrics.get("agent_val_size"),
        descriptor_count_before=internal_metrics.get("descriptor_count_before"),
        descriptor_count_after=internal_metrics.get("descriptor_count_after"),
        feature_count=int(feature_count) if feature_count is not None else None,
        selected_feature_names=features,
        samples_per_feature_ratio=ratio,
        mean_train_r2=_f(internal_metrics.get("mean_train_r2")),
        mean_cv_r2=_f(internal_metrics.get("mean_cv_r2")),
        cv_r2_std=_f(internal_metrics.get("cv_r2_std")),
        train_cv_gap=_f(internal_metrics.get("train_cv_gap")),
        mean_cv_rmse=_f(internal_metrics.get("mean_cv_rmse")),
        mean_cv_mae=_f(internal_metrics.get("mean_cv_mae")),
        agent_val_r2=_f(internal_metrics.get("agent_val_r2")),
        overfitting_status=internal_metrics.get("overfitting_status"),
        overfitting_acceptable=internal_metrics.get("overfitting_acceptable"),
        hpo_rounds=internal_metrics.get("hpo_rounds"),
        best_parameters=params if isinstance(params, dict) else {},
        estimator=estimator or internal_metrics.get("estimator"),
        model_complexity_summary=complexity,
        sfs_summary=internal_metrics.get("sfs_summary") or {},
        ga_summary=internal_metrics.get("ga_summary") or {},
        warnings=list(warnings or internal_metrics.get("warnings") or []),
        source_artifact_paths=source_paths,
        external_test_unavailable=True,
        cv_folds_hash=cv_folds_hash or internal_metrics.get("cv_folds_hash"),
        experiment_kind=experiment_kind,
    )
    # Merge non-conflicting extras into warnings/notes only
    if extra.get("descriptor_removal_fraction") is not None:
        summary.warnings.append(
            f"descriptor_removal_fraction={extra['descriptor_removal_fraction']}"
        )

    out = experiment_dir(run_dir, experiment_id) / "agent_visible_summary.json"
    save_json(out, summary.model_dump())
    return summary


def load_agent_visible_summary(run_dir: Path, experiment_id: str) -> AgentVisibleSummary | None:
    path = experiment_dir(run_dir, experiment_id) / "agent_visible_summary.json"
    if not path.exists():
        return None
    return AgentVisibleSummary.model_validate_json(path.read_text(encoding="utf-8"))


def metrics_from_final_selection(final_selection: Any) -> dict[str, Any]:
    """Extract internal metrics from a FinalModelSelection-like object/dict."""
    if final_selection is None:
        return {}
    if hasattr(final_selection, "model_dump"):
        data = final_selection.model_dump()
    elif isinstance(final_selection, dict):
        data = final_selection
    else:
        return {}
    cv = data.get("cv_summary") or {}
    assessment = data.get("assessment") or {}
    return {
        "mean_train_r2": cv.get("mean_train_r2"),
        "mean_cv_r2": cv.get("mean_cv_r2"),
        "cv_r2_std": cv.get("std_cv_r2"),
        "train_cv_gap": cv.get("train_cv_r2_gap"),
        "mean_cv_rmse": cv.get("mean_cv_rmse"),
        "mean_cv_mae": cv.get("mean_cv_mae"),
        "mean_train_rmse": cv.get("mean_train_rmse"),
        "mean_train_mae": cv.get("mean_train_mae"),
        "n_folds": cv.get("n_folds"),
        "overfitting_status": assessment.get("status"),
        "overfitting_acceptable": assessment.get("is_acceptable"),
        "best_parameters": data.get("params") or data.get("best_params") or {},
    }


def _f(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def read_json_if_exists(path: str | Path) -> dict[str, Any] | None:
    p = Path(path)
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))
