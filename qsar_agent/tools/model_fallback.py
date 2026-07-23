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
from qsar_agent.tools.sfs_fixed_ga_expansion import run_sfs_fixed_ga_expansion


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

    Compares RF + all fallback branches (and any SFS-fixed GA expansions) and
    returns the globally best candidate.
    """
    rf_acceptable = (
        rf_branch.hpo_result.final_selection is not None
        and rf_branch.hpo_result.final_selection.assessment.is_acceptable
    )

    fallback_settings = workflow_config.model_fallback
    expansion_settings = workflow_config.sfs_fixed_ga_expansion

    # Ensure RF expansion exists when RF is not acceptable (also when fallback disabled).
    if not rf_acceptable and rf_branch.expansion is None:
        from qsar_agent.config import ModelConfig

        if rf_branch.model_config_snapshot:
            expansion_model = ModelConfig(**rf_branch.model_config_snapshot)
        else:
            expansion_model = workflow_config.model

        expansion = run_sfs_fixed_ga_expansion(
            rf_branch,
            train_path=train_path,
            run_dir=run_dir,
            model_config=expansion_model,
            ga_config=workflow_config.ga,
            hpo_config=hpo_config,
            expansion_settings=expansion_settings,
            grid_proposer=grid_proposer,
            log_callback=log_callback,
        )
        if expansion is not None:
            rf_branch = rf_branch.model_copy(update={"expansion": expansion})

    if rf_acceptable or not fallback_settings.enabled:
        reason = "RF acceptable" if rf_acceptable else "Model fallback disabled"
        if log_callback:
            log_callback(f"Model fallback skipped: {reason}.")
        candidates = _collect_candidates(rf_branch)
        if len(candidates) > 1:
            best = select_best_across_models(candidates)
            cross = _cross_from_best(best)
            _write_comparison(run_dir, cross)
            return ModelFallbackResult(
                triggered=False,
                rf_branch=rf_branch,
                cross_model_selection=cross,
                comparison_json_path=str(run_dir / "model_comparison.json"),
                comparison_md_path=str(run_dir / "model_comparison_summary.md"),
                comparison_csv_path=str(run_dir / "model_comparison.csv"),
            )
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
            expansion_settings=expansion_settings,
        )
        fallback_branches.append(branch)
        if log_callback and branch.hpo_result.final_selection:
            fs = branch.hpo_result.final_selection
            log_callback(
                f"Fallback {estimator} complete: CV R²={fs.cv_summary.mean_cv_r2:.3f}, "
                f"status={fs.assessment.status}."
            )

    candidates = _collect_candidates(rf_branch)
    for b in fallback_branches:
        candidates.extend(_collect_candidates(b))
    best = select_best_across_models(candidates)
    cross = _cross_from_best(best)

    _write_comparison(run_dir, cross)

    if log_callback:
        log_callback(
            f"Model fallback complete. Winner: {cross.winning_estimator}"
            f"{' [' + cross.winner_expansion_label + ']' if cross.winner_is_expansion else ''} "
            f"(CV R²={cross.final_selection.cv_summary.mean_cv_r2:.3f})."
        )

    return ModelFallbackResult(
        triggered=True,
        fallback_models_tried=estimators,
        rf_branch=rf_branch,
        fallback_branches=fallback_branches,
        cross_model_selection=cross,
        comparison_json_path=str(run_dir / "model_comparison.json"),
        comparison_md_path=str(run_dir / "model_comparison_summary.md"),
        comparison_csv_path=str(run_dir / "model_comparison.csv"),
    )


def _collect_candidates(branch: ModelBranchResult) -> list[dict]:
    cands = [_branch_to_candidate(branch)]
    if branch.expansion is not None and branch.expansion.hpo_result.final_selection is not None:
        cands.append(_branch_to_candidate(branch.expansion))
    return cands


def _branch_to_candidate(branch: ModelBranchResult) -> dict:
    from qsar_agent.config import ModelConfig

    estimator_label = branch.estimator
    if branch.is_expansion and branch.expansion_label:
        estimator_label = f"{branch.estimator} ({branch.expansion_label})"

    return {
        "estimator": estimator_label,
        "base_estimator": branch.estimator,
        "selected_features": branch.ga.selected_features,
        "final_selection": branch.hpo_result.final_selection,
        "model_config": ModelConfig(**branch.model_config_snapshot),
        "is_expansion": branch.is_expansion,
        "expansion_label": branch.expansion_label,
    }


def _cross_from_best(best: dict) -> CrossModelSelection:
    return CrossModelSelection(
        winning_estimator=best["winning_estimator"],
        selected_features=best["selected_features"],
        final_model_config=best["final_model_config"],
        final_selection=best["final_selection"],
        selection_rationale=best["selection_rationale"],
        warning=best.get("warning", ""),
        compared_models=best["compared_models"],
        winner_is_expansion=bool(best.get("winner_is_expansion", False)),
        winner_expansion_label=str(best.get("winner_expansion_label", "")),
    )


def _write_comparison(run_dir: Path, cross: CrossModelSelection) -> None:
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
    if cross.winner_is_expansion:
        md_lines.append(
            f"**Winner source:** SFS-fixed GA expansion (`{cross.winner_expansion_label}`)\n"
        )
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
                "is_expansion": False,
            }
        ],
        winner_is_expansion=False,
        winner_expansion_label="",
    )
