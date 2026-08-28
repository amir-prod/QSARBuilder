"""Deterministic acceptance-requirement evaluator (LLM must not pass/fail numbers)."""

from __future__ import annotations

from typing import Any

from qsar_agent.config import ModelingRequirements, WorkflowConfig
from qsar_agent.schemas.agentic import FailedRequirement, RequirementEvaluation
from qsar_agent.schemas.handoff import ExperimentMetrics


TRAIN_CV_GAP_DEFINITION = "mean_inner_fold_train_r2 - mean_inner_fold_validation_r2"


def requirements_from_config(config: WorkflowConfig) -> ModelingRequirements:
    req = config.agentic_improvement.requirements.model_copy()
    hpo = config.hpo
    if req.minimum_cv_r2 is None:
        req.minimum_cv_r2 = hpo.minimum_cv_r2
    if req.maximum_train_cv_gap is None:
        req.maximum_train_cv_gap = hpo.overfit_gap_threshold
    if req.maximum_cv_r2_std is None:
        req.maximum_cv_r2_std = hpo.cv_std_threshold
    req.allow_latent_components = bool(
        req.allow_latent_components or config.agentic_improvement.allow_latent_components
    )
    return req


def _gap_from_metrics(metrics: dict[str, Any]) -> tuple[float | None, float | None]:
    fold_train = metrics.get("mean_cv_fold_train_r2")
    oof = metrics.get("oof_cv_r2", metrics.get("cv_r2"))
    gap = metrics.get("cv_fold_train_val_gap", metrics.get("train_cv_r2_gap"))
    if gap is None and fold_train is not None and oof is not None:
        gap = float(fold_train) - float(oof)
    refit_gap = metrics.get("refit_train_cv_gap")
    if refit_gap is None:
        refit = metrics.get("refit_train_r2", metrics.get("train_r2"))
        if refit is not None and oof is not None:
            refit_gap = float(refit) - float(oof)
    return (
        None if gap is None else float(gap),
        None if refit_gap is None else float(refit_gap),
    )


def metrics_from_experiment(record: dict[str, Any] | ExperimentMetrics) -> dict[str, Any]:
    if isinstance(record, ExperimentMetrics):
        return record.model_dump()
    if "metrics" in record and isinstance(record["metrics"], dict):
        return dict(record["metrics"])
    return dict(record)


def evaluate_requirements(
    metrics: dict[str, Any] | ExperimentMetrics,
    requirements: ModelingRequirements,
    *,
    feature_count: int | None = None,
    ad_coverage: float | None = None,
    runtime_seconds: float | None = None,
    model_complexity: float | None = None,
) -> RequirementEvaluation:
    """Compare observed development metrics against configured gates.

    ``train_cv_gap`` is mean inner-fold train R² minus mean inner-fold validation R².
    ``refit_train_cv_gap`` is reported separately and is not mixed into that definition.
    Unavailable values stay ``None`` (never coerced to zero).
    """
    raw = metrics_from_experiment(metrics)
    gap, refit_gap = _gap_from_metrics(raw)
    cv_r2 = raw.get("oof_cv_r2", raw.get("cv_r2"))
    cv_std = raw.get("cv_r2_std")
    rmse = raw.get("cv_rmse")
    mae = raw.get("cv_mae")
    n_feat = feature_count if feature_count is not None else raw.get("feature_count")

    passed: list[str] = []
    failed: list[FailedRequirement] = []

    def _check(name: str, observed: float | int | None, required: float | int | None, cmp: str) -> None:
        if required is None:
            return
        if observed is None:
            failed.append(
                FailedRequirement(
                    name=name,
                    observed=None,
                    required=required,
                    message="Metric unavailable (null); not treated as zero.",
                )
            )
            return
        ok = float(observed) >= float(required) if cmp == "ge" else float(observed) <= float(required)
        if ok:
            passed.append(name)
        else:
            failed.append(FailedRequirement(name=name, observed=observed, required=required))

    _check("minimum_cv_r2", None if cv_r2 is None else float(cv_r2), requirements.minimum_cv_r2, "ge")
    _check("maximum_train_cv_gap", gap, requirements.maximum_train_cv_gap, "le")
    _check("maximum_cv_r2_std", None if cv_std is None else float(cv_std), requirements.maximum_cv_r2_std, "le")
    _check("maximum_rmse", None if rmse is None else float(rmse), requirements.maximum_rmse, "le")
    _check("maximum_mae", None if mae is None else float(mae), requirements.maximum_mae, "le")
    _check(
        "maximum_feature_count",
        None if n_feat is None else int(n_feat),
        requirements.maximum_feature_count,
        "le",
    )
    _check("minimum_ad_coverage", ad_coverage, requirements.minimum_ad_coverage, "ge")
    _check("maximum_runtime", runtime_seconds, requirements.maximum_runtime_seconds, "le")
    _check("maximum_model_complexity", model_complexity, requirements.maximum_model_complexity, "le")

    return RequirementEvaluation(
        acceptance_status="passed" if not failed else "failed",
        passed_requirements=passed,
        failed_requirements=failed,
        train_cv_gap=gap,
        train_cv_gap_definition=TRAIN_CV_GAP_DEFINITION,
        refit_train_cv_gap=refit_gap,
        metrics={
            "cv_r2": None if cv_r2 is None else float(cv_r2),
            "cv_r2_std": None if cv_std is None else float(cv_std),
            "cv_rmse": None if rmse is None else float(rmse),
            "cv_mae": None if mae is None else float(mae),
            "train_cv_gap": gap,
            "refit_train_cv_gap": refit_gap,
        },
    )
