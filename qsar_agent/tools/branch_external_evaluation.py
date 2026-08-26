"""External-test evaluation (scatter + Williams) for model branches."""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path
from typing import Any, Callable

from qsar_agent.config import ModelConfig
from qsar_agent.models.registry import normalize_estimator_name
from qsar_agent.schemas.applicability_domain import (
    ApplicabilityDomainResult,
    ApplicabilityDomainSummary,
)
from qsar_agent.schemas.model_fallback import BranchExternalArtifacts, ModelBranchResult
from qsar_agent.schemas.modeling import Metrics, ModelingResult
from qsar_agent.services.artifact_manager import save_json
from qsar_agent.services.plotting import plot_residuals
from qsar_agent.tools.applicability_domain import calculate_applicability_domain
from qsar_agent.tools.final_model import train_and_evaluate_final_model

_PROMOTE_FILES = (
    "prediction_scatter.png",
    "prediction_scatter.svg",
    "williams_plot.png",
    "williams_plot.svg",
    "predictions.csv",
    "final_model.joblib",
    "model_metrics.json",
    "applicability_domain.csv",
    "applicability_domain_report.json",
    "run_manifest.json",
)


def branch_display_label(branch: ModelBranchResult) -> str:
    if branch.is_expansion and branch.expansion_label:
        return f"{branch.estimator} ({branch.expansion_label})"
    return branch.estimator


def flatten_branches(*roots: ModelBranchResult | None) -> list[ModelBranchResult]:
    """Collect unique branch results (parent + SFS subset + expansion) by branch_dir."""
    out: list[ModelBranchResult] = []
    seen: set[str] = set()
    for root in roots:
        if root is None:
            continue
        for branch in (root, root.sfs_subset, root.sfs_subset_hpo, root.expansion):
            if branch is None:
                continue
            if branch.hpo_result.final_selection is None:
                continue
            key = str(Path(branch.branch_dir).resolve()) if branch.branch_dir else id(branch)
            key_s = str(key)
            if key_s in seen:
                continue
            seen.add(key_s)
            out.append(branch)
    return out


def evaluate_branch_on_external_test(
    branch: ModelBranchResult,
    *,
    train_path: str | Path,
    test_path: str | Path,
    activity_label: str = "activity",
    dataset_hash: str = "",
    config_snapshot: dict[str, Any] | None = None,
    hpo_metadata: dict[str, Any] | None = None,
    val_path: str | Path | None = None,
) -> tuple[BranchExternalArtifacts, ModelingResult, ApplicabilityDomainResult]:
    """Fit branch on train, evaluate on external test; write scatter + Williams into branch_dir."""
    if not branch.branch_dir:
        raise ValueError(f"Branch {branch_display_label(branch)} has empty branch_dir.")
    started = time.perf_counter()
    branch_dir = Path(branch.branch_dir)
    branch_dir.mkdir(parents=True, exist_ok=True)

    model_config = ModelConfig(**branch.model_config_snapshot) if branch.model_config_snapshot else ModelConfig(
        estimator=branch.estimator
    )
    base_estimator = normalize_estimator_name(model_config.estimator or branch.estimator)
    if base_estimator and base_estimator != model_config.estimator:
        model_config = model_config.model_copy(update={"estimator": base_estimator})
    meta = dict(hpo_metadata or {})
    if branch.hpo_result.final_selection is not None:
        meta.setdefault("enabled", branch.hpo_result.enabled)
        meta.setdefault("rounds_completed", branch.hpo_result.rounds_completed)
        meta.setdefault("final_model_source", branch.hpo_result.final_selection.source)
        meta.setdefault("final_params", branch.hpo_result.final_selection.params)

    modeling = train_and_evaluate_final_model(
        train_path,
        test_path,
        branch_dir,
        branch.ga.selected_features,
        model_config,
        activity_label,
        dataset_hash,
        config_snapshot,
        hpo_metadata=meta,
        val_path=val_path,
    )
    ad = calculate_applicability_domain(
        train_path,
        test_path,
        modeling.predictions_path,
        branch_dir,
        branch.ga.selected_features,
        val_path=val_path,
    )
    residual_png = branch_dir / "residual_plot.png"
    residual_svg = branch_dir / "residual_plot.svg"
    try:
        import pandas as pd

        pred_df = pd.read_csv(modeling.predictions_path)
        plot_residuals(
            pred_df["predicted_activity"].values,
            pred_df["residual"].values,
            pred_df["split"].values,
            residual_png,
            residual_svg,
        )
        residual_png_path = str(residual_png)
        residual_svg_path = str(residual_svg)
    except Exception:
        residual_png_path = ""
        residual_svg_path = ""
    artifacts = BranchExternalArtifacts(
        estimator=branch.estimator,
        label=branch_display_label(branch),
        is_expansion=branch.is_expansion,
        expansion_label=branch.expansion_label,
        branch_dir=str(branch_dir),
        selected_features=list(branch.ga.selected_features),
        predictions_path=modeling.predictions_path,
        model_path=modeling.model_path,
        metrics_path=modeling.metrics_path,
        scatter_png_path=modeling.scatter_png_path,
        scatter_svg_path=modeling.scatter_svg_path,
        williams_png_path=ad.williams_png_path,
        williams_svg_path=ad.williams_svg_path,
        residual_png_path=residual_png_path,
        residual_svg_path=residual_svg_path,
        ad_report_path=ad.report_path,
        ad_classifications_path=ad.classifications_path,
        train_r2=modeling.train_metrics.r2,
        val_r2=None if modeling.val_metrics is None else modeling.val_metrics.r2,
        test_r2=modeling.test_metrics.r2,
        runtime_seconds=time.perf_counter() - started,
    )
    return artifacts, modeling, ad


def evaluate_branches_on_external_test(
    branches: list[ModelBranchResult],
    *,
    train_path: str | Path,
    test_path: str | Path,
    activity_label: str = "activity",
    dataset_hash: str = "",
    config_snapshot: dict[str, Any] | None = None,
    log_callback: Callable[[str], None] | None = None,
    val_path: str | Path | None = None,
) -> list[tuple[BranchExternalArtifacts, ModelingResult, ApplicabilityDomainResult]]:
    """Evaluate each unique branch dir; return artifacts + modeling + AD per branch."""
    results = []
    for branch in branches:
        label = branch_display_label(branch)
        if log_callback:
            log_callback(f"External evaluation for branch: {label}")
        results.append(
            evaluate_branch_on_external_test(
                branch,
                train_path=train_path,
                test_path=test_path,
                activity_label=activity_label,
                dataset_hash=dataset_hash,
                config_snapshot=config_snapshot,
                val_path=val_path,
            )
        )
        if log_callback:
            art = results[-1][0]
            val_txt = f", val R²={art.val_r2:.3f}" if art.val_r2 is not None else ""
            log_callback(
                f"External evaluation complete for {label}: "
                f"train R²={art.train_r2:.3f}{val_txt}, test R²={art.test_r2:.3f}."
            )
    return results


def persist_branch_external_artifacts(
    run_dir: Path, artifacts: list[BranchExternalArtifacts]
) -> None:
    save_json(run_dir / "branch_external_artifacts.json", [a.model_dump() for a in artifacts])


def append_external_eval(
    artifacts: list[BranchExternalArtifacts],
    *branches: ModelBranchResult | None,
    train_path: str | Path,
    test_path: str | Path | None,
    activity_label: str = "activity",
    dataset_hash: str = "",
    config_snapshot: dict[str, Any] | None = None,
    log_callback: Callable[[str], None] | None = None,
    val_path: str | Path | None = None,
    run_dir: Path | None = None,
) -> list[BranchExternalArtifacts]:
    """Fit and plot scatter + Williams for each new branch as soon as it is ready."""
    if test_path is None:
        return artifacts
    seen = {str(Path(a.branch_dir).resolve()) for a in artifacts if a.branch_dir}
    for branch in branches:
        if branch is None or not branch.branch_dir:
            continue
        if branch.hpo_result.final_selection is None:
            continue
        key = str(Path(branch.branch_dir).resolve())
        if key in seen:
            continue
        label = branch_display_label(branch)
        if log_callback:
            log_callback(f"External evaluation for branch: {label}")
        art, _modeling, _ad = evaluate_branch_on_external_test(
            branch,
            train_path=train_path,
            test_path=test_path,
            activity_label=activity_label,
            dataset_hash=dataset_hash,
            config_snapshot=config_snapshot,
            val_path=val_path,
        )
        artifacts.append(art)
        seen.add(key)
        if log_callback:
            val_txt = f", val R²={art.val_r2:.3f}" if art.val_r2 is not None else ""
            log_callback(
                f"External evaluation complete for {label}: "
                f"train R²={art.train_r2:.3f}{val_txt}, test R²={art.test_r2:.3f}."
            )
        if run_dir is not None:
            persist_branch_external_artifacts(run_dir, artifacts)
    return artifacts


def promote_branch_artifacts_to_run_dir(branch_dir: str | Path, run_dir: Path) -> dict[str, str]:
    """
    Copy external-eval artifacts from a branch directory into the run root.

    No-op (returns existing paths) when branch_dir is already the run root.
    """
    src = Path(branch_dir).resolve()
    dst = Path(run_dir).resolve()
    dst.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}
    if src == dst:
        for name in _PROMOTE_FILES:
            p = dst / name
            if p.exists():
                paths[name] = str(p)
        return paths

    for name in _PROMOTE_FILES:
        src_file = src / name
        if not src_file.exists():
            continue
        dest_file = dst / name
        shutil.copy2(src_file, dest_file)
        paths[name] = str(dest_file)
    return paths


def find_winning_branch(
    branches: list[ModelBranchResult],
    *,
    winning_estimator: str,
    selected_features: list[str],
    winner_is_expansion: bool = False,
    winner_expansion_label: str = "",
) -> ModelBranchResult | None:
    """Locate the branch that matches cross-model selection metadata."""
    feat_set = list(selected_features)
    for branch in branches:
        label = branch_display_label(branch)
        if label != winning_estimator and branch.estimator != winning_estimator:
            continue
        if bool(branch.is_expansion) != bool(winner_is_expansion):
            continue
        if winner_is_expansion and branch.expansion_label != winner_expansion_label:
            continue
        if list(branch.ga.selected_features) != feat_set:
            continue
        return branch
    # Fallback: match by features + expansion flags only.
    for branch in branches:
        if bool(branch.is_expansion) != bool(winner_is_expansion):
            continue
        if winner_is_expansion and branch.expansion_label != winner_expansion_label:
            continue
        if list(branch.ga.selected_features) == feat_set:
            return branch
    return None


def modeling_result_with_run_dir_paths(
    modeling: ModelingResult,
    ad: ApplicabilityDomainResult,
    run_dir: Path,
) -> tuple[ModelingResult, ApplicabilityDomainResult]:
    """Rewrite modeling/AD path fields to run-root locations after promotion."""
    run_dir = Path(run_dir)
    modeling = modeling.model_copy(
        update={
            "predictions_path": str(run_dir / "predictions.csv"),
            "metrics_path": str(run_dir / "model_metrics.json"),
            "model_path": str(run_dir / "final_model.joblib"),
            "scatter_png_path": str(run_dir / "prediction_scatter.png"),
            "scatter_svg_path": str(run_dir / "prediction_scatter.svg"),
            "manifest_path": str(run_dir / "run_manifest.json"),
        }
    )
    ad = ad.model_copy(
        update={
            "classifications_path": str(run_dir / "applicability_domain.csv"),
            "report_path": str(run_dir / "applicability_domain_report.json"),
            "williams_png_path": str(run_dir / "williams_plot.png"),
            "williams_svg_path": str(run_dir / "williams_plot.svg"),
        }
    )
    return modeling, ad


def load_modeling_and_ad_from_artifacts(
    art: BranchExternalArtifacts,
    *,
    hpo_metadata: dict[str, Any] | None = None,
) -> tuple[ModelingResult, ApplicabilityDomainResult]:
    """Rebuild ModelingResult / ApplicabilityDomainResult from branch artifact files."""
    hpo_meta = hpo_metadata or {}
    metrics = json.loads(Path(art.metrics_path).read_text(encoding="utf-8"))
    val_metrics = Metrics(**metrics["val"]) if "val" in metrics else None
    modeling = ModelingResult(
        train_metrics=Metrics(**metrics["train"]),
        val_metrics=val_metrics,
        test_metrics=Metrics(**metrics["test"]),
        selected_features=art.selected_features,
        predictions_path=art.predictions_path,
        metrics_path=art.metrics_path,
        model_path=art.model_path,
        scatter_png_path=art.scatter_png_path,
        scatter_svg_path=art.scatter_svg_path,
        manifest_path=str(Path(art.branch_dir) / "run_manifest.json"),
        hpo_enabled=bool(hpo_meta.get("enabled", False)),
        hpo_rounds_completed=int(hpo_meta.get("rounds_completed", 0)),
        final_model_source=str(hpo_meta.get("final_model_source", "baseline")),
    )
    ad_report = json.loads(Path(art.ad_report_path).read_text(encoding="utf-8"))
    summary_keys = (
        "train_in_domain_count",
        "train_in_domain_pct",
        "val_in_domain_count",
        "val_in_domain_pct",
        "test_in_domain_count",
        "test_in_domain_pct",
        "warning_leverage",
        "residual_threshold",
        "high_leverage_ids",
        "response_outlier_ids",
    )
    summary_data = {k: ad_report[k] for k in summary_keys if k in ad_report}
    ad = ApplicabilityDomainResult(
        summary=ApplicabilityDomainSummary(**summary_data),
        classifications_path=art.ad_classifications_path,
        report_path=art.ad_report_path,
        williams_png_path=art.williams_png_path,
        williams_svg_path=art.williams_svg_path,
    )
    return modeling, ad
