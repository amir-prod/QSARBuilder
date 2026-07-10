"""Per-model feature selection and HPO branch."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from qsar_agent.config import GAConfig, ModelConfig, SFSConfig
from qsar_agent.schemas.hyperparameter_optimization import AgentGridProposal, HPOConfig
from qsar_agent.schemas.model_fallback import ModelBranchResult
from qsar_agent.services.plotting import plot_sfs_r2
from qsar_agent.tools.feature_count_selection import (
    save_feature_count_selection,
    select_feature_count_one_se_rule,
)
from qsar_agent.tools.genetic_algorithm import run_genetic_algorithm
from qsar_agent.tools.hyperparameter_optimization import run_iterative_hyperparameter_optimization
from qsar_agent.tools.sequential_feature_selection import run_sequential_feature_selection


def run_model_branch(
    *,
    train_path: str | Path,
    run_dir: Path,
    model_config: ModelConfig,
    sfs_config: SFSConfig,
    ga_config: GAConfig,
    hpo_config: HPOConfig,
    output_subdir: Path | None = None,
    grid_proposer: Callable[..., AgentGridProposal] | None = None,
    log_callback: Callable[[str], None] | None = None,
    explain_feature_count: bool = True,
) -> ModelBranchResult:
    """Run SFS → feature count → GA → HPO for a single estimator."""
    branch_dir = output_subdir if output_subdir is not None else run_dir
    branch_dir.mkdir(parents=True, exist_ok=True)

    sfs = run_sequential_feature_selection(
        train_path,
        branch_dir,
        sfs_config.max_features,
        sfs_config.cv_folds,
        model_config,
        sfs_config.random_seed,
        sfs_config.n_jobs,
    )

    feature_count = select_feature_count_one_se_rule(sfs)
    feature_count = save_feature_count_selection(feature_count, branch_dir)

    import pandas as pd

    plot_sfs_r2(
        pd.read_csv(sfs.results_csv_path),
        Path(sfs.plot_png_path),
        Path(sfs.plot_svg_path),
        feature_count.selected_feature_count,
    )

    if explain_feature_count:
        from qsar_agent.agents.qsar_agent import run_agent_feature_count_selection

        feature_count = run_agent_feature_count_selection(sfs, branch_dir)

    ga = run_genetic_algorithm(
        train_path,
        branch_dir,
        feature_count.selected_feature_count,
        ga_config,
        model_config,
    )

    train_df = pd.read_csv(train_path)
    n_train = len(train_df)
    n_features = len(ga.selected_features)

    hpo_result = run_iterative_hyperparameter_optimization(
        train_path,
        ga.selected_features,
        model_config,
        hpo_config,
        run_dir=run_dir,
        output_subdir=branch_dir if output_subdir is not None else None,
        grid_proposer=grid_proposer,
        log_callback=log_callback,
        n_features=n_features,
        n_train_samples=n_train,
    )

    return ModelBranchResult(
        estimator=model_config.estimator,
        model_config_snapshot=hpo_result.final_model_config,
        branch_dir=str(branch_dir),
        sfs=sfs,
        feature_count=feature_count,
        ga=ga,
        hpo_result=hpo_result,
    )
