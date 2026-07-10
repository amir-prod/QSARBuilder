"""Multi-model fallback orchestration after failed RF HPO."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import pandas as pd

from qsar_agent.config import WorkflowConfig
from qsar_agent.models.registry import DEFAULT_FALLBACK_ESTIMATORS, estimator_slug, get_default_model_config
from qsar_agent.schemas.hyperparameter_optimization import AgentGridProposal, HPOConfig
from qsar_agent.schemas.model_fallback import CrossModelSelection, ModelBranchResult, ModelFallbackResult
from qsar_agent.services.artifact_manager import save_json
from qsar_agent.tools.hyperparameter_optimization import select_best_across_models
from qsar_agent.tools.model_branch import run_model_branch


def run_model_fallback_if_needed(
    rf_branch: ModelBranchResult,
    *,
    train_path: str | Path,
    run_dir: Path,
    workflow_config: WorkflowConfig,
    hpo_config: HPOConfig,
    grid_proposer: Callable[..., AgentGridProposal] | None = None,
    log_callback: Callable[[str], None] | None = None,
) -> ModelFallbackResult:
    """
    Try fallback estimators when RF HPO did not find an acceptable model.

    Compares RF + all fallback branches and returns the globally best candidate.
    """
    rf_acceptable = (
        rf_branch.hpo_result.final_selection is not None
        and rf_branch.hpo_result.final_selection.assessment.is_acceptable
    )

    fallback_settings = workflow_config.model_fallback
    if rf_acceptable or not fallback_settings.enabled:
        reason = "RF acceptable" if rf_acceptable else "Model fallback disabled"
        if log_callback:
            log_callback(f"Model fallback skipped: {reason}.")
        cross = _cross_selection_from_branch(rf_branch, reason)
        return ModelFallbackResult(
            triggered=False,
            rf_branch=rf_branch,
            cross_model_selection=cross,
        )

    estimators = fallback_settings.estimators or list(DEFAULT_FALLBACK_ESTIMATORS)
    fallback_branches: list[ModelBranchResult] = []

    for estimator in estimators:
        if log_callback:
            log_callback(f"Starting fallback model branch: {estimator}")
        model_cfg = get_default_model_config(
            estimator,
            random_state=workflow_config.random_seed,
            n_jobs=workflow_config.hpo.n_jobs,
        )
        branch_dir = run_dir / "fallback_models" / estimator_slug(estimator)
        branch = run_model_branch(
            train_path=train_path,
            run_dir=run_dir,
            model_config=model_cfg,
            sfs_config=workflow_config.sfs,
            ga_config=workflow_config.ga,
            hpo_config=hpo_config,
            output_subdir=branch_dir,
            grid_proposer=grid_proposer,
            log_callback=log_callback,
            explain_feature_count=False,
        )
        fallback_branches.append(branch)
        if log_callback and branch.hpo_result.final_selection:
            fs = branch.hpo_result.final_selection
            log_callback(
                f"Fallback {estimator} complete: CV R²={fs.cv_summary.mean_cv_r2:.3f}, "
                f"status={fs.assessment.status}."
            )

    candidates = [_branch_to_candidate(rf_branch), *[_branch_to_candidate(b) for b in fallback_branches]]
    best = select_best_across_models(candidates)

    cross = CrossModelSelection(
        winning_estimator=best["winning_estimator"],
        selected_features=best["selected_features"],
        final_model_config=best["final_model_config"],
        final_selection=best["final_selection"],
        selection_rationale=best["selection_rationale"],
        warning=best.get("warning", ""),
        compared_models=best["compared_models"],
    )

    comparison_json = run_dir / "model_comparison.json"
    comparison_md = run_dir / "model_comparison_summary.md"
    comparison_csv = run_dir / "model_comparison.csv"

    save_json(comparison_json, cross.model_dump())
    pd.DataFrame(cross.compared_models).to_csv(comparison_csv, index=False)

    md_lines = [
        "# Model Comparison (RF + Fallbacks)\n",
        f"**Winner:** {cross.winning_estimator} ({cross.final_selection.source})\n",
        f"{cross.selection_rationale}\n",
    ]
    if cross.warning:
        md_lines.append(f"**Warning:** {cross.warning}\n")
    md_lines.append("\n## All candidates\n")
    for row in cross.compared_models:
        md_lines.append(
            f"- {row['estimator']} ({row['source']}): CV R²={row['mean_cv_r2']:.4f}, "
            f"gap={row['train_cv_r2_gap']:.4f}, status={row['status']}, "
            f"acceptable={row['acceptable']}, n_features={row['n_features']}"
        )
    comparison_md.write_text("\n".join(md_lines), encoding="utf-8")

    if log_callback:
        log_callback(
            f"Model fallback complete. Winner: {cross.winning_estimator} "
            f"(CV R²={cross.final_selection.cv_summary.mean_cv_r2:.3f})."
        )

    return ModelFallbackResult(
        triggered=True,
        fallback_models_tried=estimators,
        rf_branch=rf_branch,
        fallback_branches=fallback_branches,
        cross_model_selection=cross,
        comparison_json_path=str(comparison_json),
        comparison_md_path=str(comparison_md),
        comparison_csv_path=str(comparison_csv),
    )


def _branch_to_candidate(branch: ModelBranchResult) -> dict:
    from qsar_agent.config import ModelConfig

    return {
        "estimator": branch.estimator,
        "selected_features": branch.ga.selected_features,
        "final_selection": branch.hpo_result.final_selection,
        "model_config": ModelConfig(**branch.model_config_snapshot),
    }


def _cross_selection_from_branch(branch: ModelBranchResult, reason: str) -> CrossModelSelection:
    fs = branch.hpo_result.final_selection
    if fs is None:
        raise ValueError("RF branch missing final_selection.")
    return CrossModelSelection(
        winning_estimator=branch.estimator,
        selected_features=branch.ga.selected_features,
        final_model_config=branch.model_config_snapshot,
        final_selection=fs,
        selection_rationale=f"Using RF branch only: {reason}.",
        warning=fs.warning,
        compared_models=[
            {
                "estimator": branch.estimator,
                "source": fs.source,
                "mean_cv_r2": fs.cv_summary.mean_cv_r2,
                "train_cv_r2_gap": fs.cv_summary.train_cv_r2_gap,
                "status": fs.assessment.status,
                "acceptable": fs.assessment.is_acceptable,
                "n_features": len(branch.ga.selected_features),
            }
        ],
    )
