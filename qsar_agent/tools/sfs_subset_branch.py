"""Compete with the one-SE SFS feature subset (default params and HPO)."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import pandas as pd

from qsar_agent.config import ModelConfig, SFSSubsetBranchSettings
from qsar_agent.models.registry import baseline_params_from_config, params_to_model_config
from qsar_agent.schemas.feature_selection import GAResult
from qsar_agent.schemas.hyperparameter_optimization import (
    AgentGridProposal,
    FinalModelSelection,
    HPOConfig,
    HPOResult,
)
from qsar_agent.schemas.model_fallback import ModelBranchResult
from qsar_agent.services.artifact_manager import save_json
from qsar_agent.tools.hyperparameter_optimization import (
    evaluate_baseline_model_cv,
    run_iterative_hyperparameter_optimization,
)
from qsar_agent.tools.overfitting_assessment import assess_overfitting
from qsar_agent.tools.sfs_fixed_ga_expansion import _sfs_features_at_count


def _same_feature_set(left: list[str], right: list[str]) -> bool:
    return set(left) == set(right)


def _synthetic_ga_result(features: list[str], features_path: Path, best_fitness: float) -> GAResult:
    save_json(features_path, {"selected_features": features, "source": "sfs_subset"})
    return GAResult(
        selected_features=list(features),
        best_fitness=best_fitness,
        history_csv_path="",
        selected_features_path=str(features_path),
        configuration_path="",
        convergence_png_path="",
        convergence_svg_path="",
    )


def _baseline_selection_from_hpo(
    hpo_result: HPOResult,
    model_config: ModelConfig,
) -> FinalModelSelection | None:
    if hpo_result.baseline_cv is not None and hpo_result.baseline_assessment is not None:
        compared = []
        if hpo_result.final_selection is not None:
            compared = list(hpo_result.final_selection.compared_candidates)
        return FinalModelSelection(
            source="baseline",
            params=baseline_params_from_config(model_config),
            cv_summary=hpo_result.baseline_cv.summary,
            assessment=hpo_result.baseline_assessment,
            selection_rationale="SFS subset with default hyperparameters.",
            compared_candidates=compared,
        )
    if hpo_result.final_selection is not None and hpo_result.final_selection.source == "baseline":
        return hpo_result.final_selection
    return None


def _hpo_disabled_result(
    train_path: str | Path,
    selected_features: list[str],
    model_config: ModelConfig,
    hpo_config: HPOConfig,
    subset_dir: Path,
    val_path: str | Path | None,
) -> HPOResult:
    from qsar_agent.tools.hyperparameter_optimization import _score_holdout_val

    baseline_cv = evaluate_baseline_model_cv(
        train_path,
        selected_features,
        model_config,
        hpo_config.cv_folds,
        hpo_config.random_seed,
        subset_dir,
    )
    val_r2 = _score_holdout_val(train_path, val_path, selected_features, model_config)
    if val_r2 is not None:
        baseline_cv = baseline_cv.model_copy(
            update={
                "summary": baseline_cv.summary.model_copy(
                    update={"holdout_val_r2": val_r2}
                )
            }
        )
    assessment = assess_overfitting(baseline_cv.summary, hpo_config.thresholds)
    params = baseline_params_from_config(model_config)
    final_config = params_to_model_config(params, model_config).model_dump()
    selection = FinalModelSelection(
        source="baseline",
        params=params,
        cv_summary=baseline_cv.summary,
        assessment=assessment,
        selection_rationale="SFS subset with default hyperparameters (HPO disabled).",
    )
    return HPOResult(
        enabled=False,
        rounds_completed=0,
        max_rounds=hpo_config.max_hpo_rounds,
        baseline_cv=baseline_cv,
        baseline_assessment=assessment,
        final_assessment=assessment,
        final_selection=selection,
        final_model_config=final_config,
    )


def run_sfs_subset_branch(
    branch: ModelBranchResult,
    *,
    train_path: str | Path,
    run_dir: Path,
    model_config: ModelConfig,
    hpo_config: HPOConfig,
    settings: SFSSubsetBranchSettings | None = None,
    grid_proposer: Callable[..., AgentGridProposal] | None = None,
    log_callback: Callable[[str], None] | None = None,
    val_path: str | Path | None = None,
) -> tuple[ModelBranchResult | None, ModelBranchResult | None]:
    """
    Evaluate the one-SE SFS subset as competing model(s).

    Returns ``(sfs_subset, sfs_subset_hpo)``. The HPO child is only created when
    HPO selects a non-baseline configuration. Returns ``(None, None)`` when
    skipped (disabled, missing SFS row, or features identical to GA).
    """
    cfg = settings or SFSSubsetBranchSettings()
    if not cfg.enabled:
        if log_callback:
            log_callback("SFS-subset branch skipped: disabled.")
        return None, None

    sfs_features = _sfs_features_at_count(branch)
    if _same_feature_set(sfs_features, list(branch.ga.selected_features)):
        if log_callback:
            log_callback(
                "SFS-subset branch skipped: SFS features match GA-selected features."
            )
        return None, None

    parent_dir = Path(branch.branch_dir) if branch.branch_dir else Path(run_dir)
    subset_dir = parent_dir / cfg.output_subdir
    subset_dir.mkdir(parents=True, exist_ok=True)

    if log_callback:
        log_callback(
            f"SFS-subset branch for {branch.estimator}: "
            f"{len(sfs_features)} one-SE feature(s), running HPO."
        )

    train_df = pd.read_csv(train_path)
    n_train = len(train_df)
    n_features = len(sfs_features)

    if hpo_config.enabled:
        hpo_result = run_iterative_hyperparameter_optimization(
            train_path,
            sfs_features,
            model_config,
            hpo_config,
            run_dir=run_dir,
            output_subdir=subset_dir,
            grid_proposer=grid_proposer,
            log_callback=log_callback,
            n_features=n_features,
            n_train_samples=n_train,
            val_path=val_path,
        )
    else:
        hpo_result = _hpo_disabled_result(
            train_path, sfs_features, model_config, hpo_config, subset_dir, val_path
        )

    baseline_fs = _baseline_selection_from_hpo(hpo_result, model_config)
    sfs_row = next(
        (r for r in branch.sfs.results if r.n_features == branch.feature_count.selected_feature_count),
        None,
    )
    fitness = (
        float(sfs_row.combined_r2)
        if sfs_row is not None and sfs_row.combined_r2 is not None
        else float(sfs_row.mean_cv_r2)
        if sfs_row is not None
        else 0.0
    )
    synthetic_ga = _synthetic_ga_result(
        sfs_features, subset_dir / "sfs_subset_features.json", fitness
    )

    baseline_branch = None
    if baseline_fs is not None:
        baseline_config = params_to_model_config(baseline_fs.params, model_config).model_dump()
        baseline_hpo = hpo_result.model_copy(
            update={
                "final_selection": baseline_fs,
                "final_model_config": baseline_config,
            }
        )
        baseline_branch = ModelBranchResult(
            estimator=branch.estimator,
            model_config_snapshot=baseline_config,
            branch_dir=str(subset_dir),
            sfs=branch.sfs,
            feature_count=branch.feature_count,
            ga=synthetic_ga,
            hpo_result=baseline_hpo,
            is_expansion=True,
            expansion_label=cfg.output_subdir,
        )

    hpo_branch = None
    final = hpo_result.final_selection
    if final is not None and final.source != "baseline":
        hpo_dir = parent_dir / cfg.hpo_output_subdir
        hpo_dir.mkdir(parents=True, exist_ok=True)
        hpo_branch = ModelBranchResult(
            estimator=branch.estimator,
            model_config_snapshot=hpo_result.final_model_config or params_to_model_config(
                final.params, model_config
            ).model_dump(),
            branch_dir=str(hpo_dir),
            sfs=branch.sfs,
            feature_count=branch.feature_count,
            ga=synthetic_ga,
            hpo_result=hpo_result,
            is_expansion=True,
            expansion_label=cfg.hpo_output_subdir,
        )

    if baseline_branch is None and hpo_branch is None:
        if log_callback:
            log_callback("SFS-subset branch skipped: no HPO selection available.")
        return None, None

    summary = {
        "estimator": branch.estimator,
        "selected_features": sfs_features,
        "n_features": len(sfs_features),
        "baseline_source": (
            baseline_branch.hpo_result.final_selection.source
            if baseline_branch and baseline_branch.hpo_result.final_selection
            else ""
        ),
        "hpo_source": final.source if final is not None else "",
        "mean_cv_r2_baseline": (
            baseline_branch.hpo_result.final_selection.cv_summary.mean_cv_r2
            if baseline_branch and baseline_branch.hpo_result.final_selection
            else None
        ),
        "mean_cv_r2_hpo": final.cv_summary.mean_cv_r2 if final is not None else None,
        "subset_dir": str(subset_dir),
    }
    save_json(subset_dir / "sfs_subset_summary.json", summary)

    if log_callback:
        parts = []
        if baseline_branch and baseline_branch.hpo_result.final_selection:
            bfs = baseline_branch.hpo_result.final_selection
            parts.append(f"default CV R²={bfs.cv_summary.mean_cv_r2:.3f}")
        if hpo_branch and final is not None:
            parts.append(f"HPO ({final.source}) CV R²={final.cv_summary.mean_cv_r2:.3f}")
        log_callback(
            f"SFS-subset branch complete for {branch.estimator}: " + ", ".join(parts)
        )

    return baseline_branch, hpo_branch


def attach_sfs_subset_branches(
    branch: ModelBranchResult,
    *,
    train_path: str | Path,
    run_dir: Path,
    model_config: ModelConfig,
    hpo_config: HPOConfig,
    settings: SFSSubsetBranchSettings | None = None,
    grid_proposer: Callable[..., AgentGridProposal] | None = None,
    log_callback: Callable[[str], None] | None = None,
    val_path: str | Path | None = None,
) -> ModelBranchResult:
    """Run SFS-subset HPO and attach child branches onto ``branch``."""
    baseline, hpo = run_sfs_subset_branch(
        branch,
        train_path=train_path,
        run_dir=run_dir,
        model_config=model_config,
        hpo_config=hpo_config,
        settings=settings,
        grid_proposer=grid_proposer,
        log_callback=log_callback,
        val_path=val_path,
    )
    if baseline is None and hpo is None:
        return branch
    return branch.model_copy(update={"sfs_subset": baseline, "sfs_subset_hpo": hpo})
