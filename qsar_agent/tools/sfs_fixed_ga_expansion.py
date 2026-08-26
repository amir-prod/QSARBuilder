"""SFS-fixed GA expansion recovery when a model branch is not acceptable."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Callable

import pandas as pd

from qsar_agent.config import (
    GAConfig,
    ModelConfig,
    SFSFixedGAExpansionSettings,
)
from qsar_agent.schemas.hyperparameter_optimization import AgentGridProposal, HPOConfig
from qsar_agent.schemas.model_fallback import ModelBranchResult
from qsar_agent.services.artifact_manager import save_json
from qsar_agent.tools.genetic_algorithm import run_genetic_algorithm
from qsar_agent.tools.hyperparameter_optimization import run_iterative_hyperparameter_optimization
from qsar_agent.tools.descriptor_calculation import META_COLUMNS


def _sfs_features_at_count(branch: ModelBranchResult) -> list[str]:
    k = branch.feature_count.selected_feature_count
    for row in branch.sfs.results:
        if row.n_features == k:
            return list(row.selected_features)
    raise ValueError(
        f"SFS results do not contain a subset for selected_feature_count={k}."
    )


def _remaining_descriptor_count(train_path: str | Path, fixed_features: list[str]) -> int:
    df = pd.read_csv(train_path, nrows=1)
    feature_cols = [c for c in df.columns if c not in META_COLUMNS]
    return len([c for c in feature_cols if c not in set(fixed_features)])


def run_sfs_fixed_ga_expansion(
    branch: ModelBranchResult,
    *,
    train_path: str | Path,
    run_dir: Path,
    model_config: ModelConfig,
    ga_config: GAConfig,
    hpo_config: HPOConfig,
    expansion_settings: SFSFixedGAExpansionSettings | None = None,
    grid_proposer: Callable[..., AgentGridProposal] | None = None,
    log_callback: Callable[[str], None] | None = None,
    val_path: str | Path | None = None,
) -> ModelBranchResult | None:
    """
    If the branch HPO conclusion is not acceptable, freeze SFS features, GA-add
    extras, re-run HPO in a separate subfolder, and return an expansion branch.

    Returns None when expansion is skipped (acceptable model, disabled, or
    insufficient remaining descriptors).
    """
    settings = expansion_settings or SFSFixedGAExpansionSettings()
    if not settings.enabled:
        if log_callback:
            log_callback("SFS-fixed GA expansion skipped: disabled.")
        return None

    fs = branch.hpo_result.final_selection
    if fs is None:
        if log_callback:
            log_callback("SFS-fixed GA expansion skipped: no final HPO selection.")
        return None
    if fs.assessment.is_acceptable:
        if log_callback:
            log_callback("SFS-fixed GA expansion skipped: model is acceptable.")
        return None

    fixed_features = _sfs_features_at_count(branch)
    extra_n = int(settings.extra_features)
    remaining = _remaining_descriptor_count(train_path, fixed_features)
    if remaining < extra_n:
        if log_callback:
            log_callback(
                f"SFS-fixed GA expansion skipped: only {remaining} remaining "
                f"descriptor(s), need {extra_n}."
            )
        return None

    branch_dir = Path(branch.branch_dir) if branch.branch_dir else Path(run_dir)
    expansion_dir = branch_dir / settings.output_subdir
    expansion_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    if log_callback:
        log_callback(
            f"SFS-fixed GA expansion for {branch.estimator}: "
            f"freeze {len(fixed_features)} SFS feature(s), GA-add {extra_n}."
        )

    ga = run_genetic_algorithm(
        train_path,
        expansion_dir,
        extra_n,
        ga_config,
        model_config,
        fixed_features=fixed_features,
        val_path=val_path,
    )
    extra_features = [f for f in ga.selected_features if f not in set(fixed_features)]

    train_df = pd.read_csv(train_path)
    n_train = len(train_df)
    n_features = len(ga.selected_features)

    hpo_result = run_iterative_hyperparameter_optimization(
        train_path,
        ga.selected_features,
        model_config,
        hpo_config,
        run_dir=run_dir,
        output_subdir=expansion_dir,
        grid_proposer=grid_proposer,
        log_callback=log_callback,
        n_features=n_features,
        n_train_samples=n_train,
        val_path=val_path,
    )

    acceptable_after = (
        hpo_result.final_selection is not None
        and hpo_result.final_selection.assessment.is_acceptable
    )
    label = settings.output_subdir
    summary = {
        "estimator": branch.estimator,
        "fixed_features": fixed_features,
        "extra_features": extra_features,
        "selected_features": ga.selected_features,
        "n_fixed": len(fixed_features),
        "n_extra": len(extra_features),
        "ga_best_fitness": ga.best_fitness,
        "acceptable_after_hpo": acceptable_after,
        "final_source": (
            hpo_result.final_selection.source if hpo_result.final_selection else ""
        ),
        "mean_cv_r2": (
            hpo_result.final_selection.cv_summary.mean_cv_r2
            if hpo_result.final_selection
            else None
        ),
        "status": (
            hpo_result.final_selection.assessment.status
            if hpo_result.final_selection
            else ""
        ),
        "expansion_dir": str(expansion_dir),
    }
    save_json(expansion_dir / "expansion_summary.json", summary)

    if log_callback and hpo_result.final_selection:
        fs2 = hpo_result.final_selection
        log_callback(
            f"SFS-fixed GA expansion complete for {branch.estimator}: "
            f"CV R²={fs2.cv_summary.mean_cv_r2:.3f}, status={fs2.assessment.status}, "
            f"n_features={len(ga.selected_features)}."
        )

    return ModelBranchResult(
        estimator=branch.estimator,
        model_config_snapshot=hpo_result.final_model_config,
        branch_dir=str(expansion_dir),
        runtime_seconds=time.perf_counter() - started,
        sfs=branch.sfs,
        feature_count=branch.feature_count,
        ga=ga,
        hpo_result=hpo_result,
        is_expansion=True,
        expansion_label=label,
    )
