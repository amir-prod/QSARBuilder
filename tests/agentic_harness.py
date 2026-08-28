"""Shared helpers for modeling-improvement agent tests."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import pandas as pd

from qsar_agent.config import (
    AgenticImprovementSettings,
    GAConfig,
    HPOSettings,
    MultiSeedGASettings,
    SFSConfig,
    WorkflowConfig,
)
from qsar_agent.schemas.agentic import AgentDecision, PipelinePhase, ToolResult
from qsar_agent.services.artifact_manager import save_json
from tests.test_handoff import _experiment, _package, _write_views

PASSING_METRICS = {
    "cv_r2": 0.80,
    "oof_cv_r2": 0.80,
    "cv_r2_std": 0.02,
    "cv_rmse": 0.20,
    "cv_mae": 0.15,
    "mean_cv_fold_train_r2": 0.85,
    "cv_fold_train_val_gap": 0.05,
    "train_cv_r2_gap": 0.05,
    "feature_count": 5,
}

FAILING_METRICS = {
    "cv_r2": 0.20,
    "oof_cv_r2": 0.20,
    "cv_r2_std": 0.25,
    "cv_rmse": 0.80,
    "cv_mae": 0.60,
    "mean_cv_fold_train_r2": 0.60,
    "cv_fold_train_val_gap": 0.40,
    "train_cv_r2_gap": 0.40,
    "feature_count": 5,
}


def write_agent_run(
    tmp_path: Path,
    *,
    passing: bool = True,
    config: WorkflowConfig | None = None,
    run_id: str = "run001",
) -> Path:
    """Create ``tmp_path/<run_id>/final_report`` plus a workflow config."""
    run_dir = tmp_path / run_id
    run_dir.mkdir()
    if passing:
        exp = _experiment("winner1", cv_r2=0.72, gap=0.09, std=0.04)
    else:
        exp = _experiment(
            "winner1",
            train_r2=0.90,
            cv_r2=0.20,
            val_r2=0.18,
            gap=0.40,
            std=0.25,
        )
    package = _package(experiments=[exp])
    _write_views(run_dir, package)
    cfg = config or default_agent_config()
    configs = run_dir / "final_report" / "configs"
    configs.mkdir(exist_ok=True)
    save_json(configs / "workflow_config.json", cfg.model_dump(mode="json"))
    return run_dir


def default_agent_config(**agentic_updates: Any) -> WorkflowConfig:
    settings = AgenticImprovementSettings(
        enabled=True,
        ga=MultiSeedGASettings(seeds=[1, 2], population_size=8, generations=2),
        **agentic_updates,
    )
    return WorkflowConfig(
        sfs=SFSConfig(n_jobs=1, cv_folds=3, max_features=5),
        ga=GAConfig(n_jobs=1, population_size=8, n_generations=2, cv_folds=3),
        hpo=HPOSettings(n_jobs=1, cv_folds=3),
        agentic_improvement=settings,
    )


def make_decision(
    tool_name: str,
    arguments: dict[str, Any] | None = None,
    *,
    diagnosis: str = "underfitting",
) -> AgentDecision:
    return AgentDecision.model_validate(
        {
            "diagnosis": diagnosis,
            "evidence": [
                {
                    "observation": "cv_r2",
                    "value": 0.2,
                    "interpretation": "below the configured minimum",
                }
            ],
            "hypothesis": "An allowlisted experiment may improve development metrics.",
            "action": {"tool_name": tool_name, "arguments": arguments or {}},
            "expected_effect": {"cv_r2": "maintain_or_improve", "train_cv_gap": "decrease"},
            "success_conditions": {"minimum_cv_r2": 0.5, "maximum_train_cv_gap": 0.15},
            "reason_existing_results_are_insufficient": "Hard requirements are still failed.",
            "confidence": "low",
        }
    )


def recording_execute(
    calls: list[str],
    *,
    metrics: dict[str, Any] | None = None,
    experiment_id: str = "exp-stub",
) -> Callable[..., ToolResult]:
    chosen = metrics if metrics is not None else PASSING_METRICS

    def _execute(tool: str, args: dict[str, Any], state: Any) -> ToolResult:
        calls.append(tool)
        extra: dict[str, Any] = {"arguments": args}
        if tool == "request_new_capability":
            extra.update(args)
        result_metrics = dict(chosen)
        if tool == "evaluate_sealed_test":
            result_metrics = {"test_r2": 0.55, "test_rmse": 0.4, "test_mae": 0.3}
        return ToolResult(
            experiment_id=f"{experiment_id}-{len(calls)}",
            tool_name=tool,
            arguments=args,
            metrics=result_metrics,
            selected_features=["a", "b", "c", "d", "e"],
            extra=extra,
        )

    return _execute


def write_development_tables(
    run_dir: Path,
    *,
    n_train: int = 24,
    n_val: int = 6,
    n_test: int = 6,
    n_features: int = 8,
    seed: int = 0,
) -> list[str]:
    """Write preprocessed train/val/test CSVs the agent tools expect."""
    import numpy as np

    rng = np.random.RandomState(seed)
    n = n_train + n_val + n_test
    names = [f"feat_{i}" for i in range(n_features)]
    X = rng.randn(n, n_features)
    y = 1.5 * X[:, 0] + 0.8 * X[:, 1] - 0.4 * X[:, 2] + rng.randn(n) * 0.05
    frame = pd.DataFrame(X, columns=names)
    frame.insert(0, "compound_id", [f"C{i:03d}" for i in range(n)])
    frame["activity"] = y
    train = frame.iloc[:n_train]
    val = frame.iloc[n_train : n_train + n_val]
    test = frame.iloc[n_train + n_val :]
    train.to_csv(run_dir / "preprocessed_train_descriptors.csv", index=False)
    val.to_csv(run_dir / "preprocessed_val_descriptors.csv", index=False)
    test.to_csv(run_dir / "preprocessed_test_descriptors.csv", index=False)
    splits = pd.DataFrame(
        {
            "compound_id": frame["compound_id"],
            "split": ["train"] * n_train + ["val"] * n_val + ["test"] * n_test,
        }
    )
    splits.to_csv(run_dir / "split_assignments.csv", index=False)
    return names


def is_development(phase: Any) -> bool:
    return str(getattr(phase, "value", phase)) == PipelinePhase.DEVELOPMENT.value
