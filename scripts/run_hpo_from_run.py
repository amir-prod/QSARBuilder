#!/usr/bin/env python3
"""
Run hyperparameter optimization on an existing QSAR Agent run.

Use this when a workflow completed through genetic feature selection (or later)
but HPO was not run, or you want to re-test HPO without redoing SFS/GA.

Required artifacts in the run directory:
  - preprocessed_train_descriptors.csv
  - ga_selected_features.json  (or selected_features in run_manifest.json)

Optional:
  - preprocessed_val_descriptors.csv (used with CV for HPO selection when present)
  - preprocessed_test_descriptors.csv (only if --run-final-model)

Example:
  python scripts/run_hpo_from_run.py outputs/acb0edb8584e
  python scripts/run_hpo_from_run.py --run-id acb0edb8584e --no-openai -v
  python scripts/run_hpo_from_run.py outputs/acb0edb8584e --run-final-model
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

# Allow running from project root without installing the package.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from qsar_agent.agents.qsar_agent import propose_hyperparameter_grid
from qsar_agent.config import ModelConfig
from qsar_agent.schemas.hyperparameter_optimization import AgentGridProposal, HPOConfig, OverfittingThresholds
from qsar_agent.tools.hyperparameter_optimization import (
    count_grid_combinations,
    run_iterative_hyperparameter_optimization,
)


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def _resolve_run_dir(run_dir: str | None, run_id: str | None, output_root: str) -> Path:
    if run_dir:
        path = Path(run_dir)
    elif run_id:
        path = Path(output_root) / run_id
    else:
        raise SystemExit("Provide --run-dir or --run-id.")
    if not path.is_dir():
        raise SystemExit(f"Run directory not found: {path}")
    return path.resolve()


def _load_selected_features(run_dir: Path) -> list[str]:
    ga_path = run_dir / "ga_selected_features.json"
    if ga_path.exists():
        data = json.loads(ga_path.read_text(encoding="utf-8"))
        features = data.get("selected_features") or data.get("features")
        if features:
            return list(features)

    manifest_path = run_dir / "run_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        features = manifest.get("selected_features")
        if features:
            return list(features)

    raise SystemExit(
        f"No selected features found in {ga_path} or {manifest_path}."
    )


def _load_model_config(run_dir: Path) -> ModelConfig:
    manifest_path = run_dir / "run_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        cfg = manifest.get("model_config") or manifest.get("workflow_config", {}).get("model")
        if cfg:
            return ModelConfig(**{**ModelConfig().model_dump(), **cfg})

    logging.warning("run_manifest.json missing; using default ModelConfig.")
    return ModelConfig()


def _load_hpo_defaults(run_dir: Path) -> HPOConfig:
    manifest_path = run_dir / "run_manifest.json"
    hpo_cfg = HPOConfig()
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        wf = manifest.get("workflow_config", {})
        hpo = wf.get("hpo")
        random_seed = wf.get("random_seed", 42)
        cv_folds = wf.get("sfs", {}).get("cv_folds", 5)
        if hpo:
            return HPOConfig(
                **{
                    **hpo_cfg.model_dump(),
                    **hpo,
                    "random_seed": random_seed,
                    "cv_folds": hpo.get("cv_folds") or cv_folds,
                    "thresholds": OverfittingThresholds(
                        overfit_gap_threshold=hpo.get("overfit_gap_threshold", 0.15),
                        severe_overfit_gap_threshold=hpo.get("severe_overfit_gap_threshold", 0.25),
                        minimum_cv_r2=hpo.get("minimum_cv_r2", 0.50),
                        cv_std_threshold=hpo.get("cv_std_threshold", 0.15),
                        minimum_train_r2=hpo.get("minimum_train_r2", 0.40),
                    ),
                }
            )
        return HPOConfig(cv_folds=cv_folds, random_seed=random_seed)
    return hpo_cfg


def _print_agent_grid_proposal(proposal: AgentGridProposal) -> None:
    """Print hyperparameter grid returned by OpenAI or fallback proposer."""
    source = (
        "OpenAI agent"
        if proposal.search_strategy != "fallback"
        else "deterministic fallback (no OpenAI)"
    )
    n_combos = count_grid_combinations(proposal.proposed_grid)
    print(f"\n{'=' * 60}")
    print(f"HPO round {proposal.round_index}: proposed grid ({source})")
    print("=" * 60)
    print(f"Search strategy: {proposal.search_strategy}")
    print(f"Reasoning:       {proposal.reasoning_summary}")
    print(f"Combinations:    ~{n_combos}")
    if proposal.warnings:
        print(f"Warnings:        {'; '.join(proposal.warnings)}")
    print("\nProposed hyperparameters to search:")
    print(json.dumps(proposal.proposed_grid, indent=2, default=str))
    if proposal.expected_effect_on_overfitting:
        print(f"\nExpected effect on overfitting: {proposal.expected_effect_on_overfitting}")
    if proposal.expected_effect_on_underfitting:
        print(f"Expected effect on underfitting: {proposal.expected_effect_on_underfitting}")


def _make_logging_grid_proposer():
    """Wrap OpenAI grid proposer to print proposals as they arrive."""

    def proposer(**kwargs) -> AgentGridProposal:
        proposal = propose_hyperparameter_grid(**kwargs)
        _print_agent_grid_proposal(proposal)
        return proposal

    return proposer


def _print_round_proposals(result, *, skip_agent_proposal: bool = False) -> None:
    """Print agent proposals and post-sanitization grids for all HPO rounds."""
    if not result.rounds:
        return
    print("\n" + "=" * 60)
    print("HPO round proposals (summary)")
    print("=" * 60)
    for rr in result.rounds:
        if rr.agent_proposal:
            if skip_agent_proposal:
                print(
                    f"\nHPO round {rr.round_index}: OpenAI proposal printed above "
                    f"({rr.agent_proposal.search_strategy})"
                )
            else:
                _print_agent_grid_proposal(rr.agent_proposal)
        if rr.sanitization and rr.sanitization.sanitized_grid:
            sanitized = rr.sanitization.sanitized_grid
            if sanitized != (rr.agent_proposal.proposed_grid if rr.agent_proposal else {}):
                print("\nSanitized grid (after validation/shrinking):")
                print(json.dumps(sanitized, indent=2, default=str))
                print(f"Candidates searched: {rr.candidates_searched}")
        if rr.best_params:
            print("\nBest parameters from this round:")
            print(json.dumps(rr.best_params, indent=2, default=str))


def _print_summary(run_dir: Path, result) -> None:
    print("\n" + "=" * 60)
    print("HPO finished")
    print("=" * 60)
    print(f"Run directory:     {run_dir}")
    print(f"Rounds completed:  {result.rounds_completed} / {result.max_rounds}")
    if result.baseline_assessment:
        b = result.baseline_assessment
        print(f"Baseline status:   {b.status} (gap={b.train_cv_r2_gap:.3f}, CV R²={b.mean_cv_r2:.3f})")
    if result.final_selection:
        s = result.final_selection
        print(f"Final selection:   {s.source}")
        print(f"Final params:      {json.dumps(s.params, indent=2)}")
        if s.warning:
            print(f"Warning:           {s.warning}")
    print("\nKey artifacts:")
    for label, path in [
        ("Baseline CV metrics", run_dir / "baseline_cv_metrics.csv"),
        ("Baseline assessment", run_dir / "baseline_overfitting_assessment.json"),
        ("Iteration log (md)", run_dir / "hpo_iteration_log.md"),
        ("Iteration log (json)", run_dir / "hpo_iteration_log.json"),
        ("Final selection", run_dir / "hpo_final_selection.json"),
        ("Summary plot", run_dir / "hpo_summary.png"),
    ]:
        print(f"  {label}: {path} {'[ok]' if path.exists() else '[missing]'}")

    log_md = run_dir / "hpo_iteration_log.md"
    if log_md.exists():
        print("\n--- hpo_iteration_log.md ---")
        print(log_md.read_text(encoding="utf-8"))
        print("--- end log ---")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run HPO on an existing QSAR Agent run (post-GA artifacts)."
    )
    parser.add_argument(
        "run_dir",
        nargs="?",
        help="Path to existing run directory (e.g. outputs/acb0edb8584e)",
    )
    parser.add_argument("--run-id", help="Run ID under --output-root (alternative to run_dir)")
    parser.add_argument("--output-root", default="outputs", help="Output root when using --run-id")
    parser.add_argument("--no-openai", action="store_true", help="Use deterministic fallback grids only")
    parser.add_argument("--disable-hpo", action="store_true", help="Baseline CV only; skip search rounds")
    parser.add_argument("--max-rounds", type=int, default=3, help="Max HPO rounds (1-3)")
    parser.add_argument("--max-candidates", type=int, default=120, help="Max grid candidates per round")
    parser.add_argument("--cv-folds", type=int, default=None, help="CV folds (default: from manifest or 5)")
    parser.add_argument("--n-jobs", type=int, default=-1, help="Parallel jobs for search")
    parser.add_argument("--run-final-model", action="store_true", help="Retrain and evaluate on test set after HPO")
    parser.add_argument("-v", "--verbose", action="store_true", help="Debug logging")
    args = parser.parse_args()

    _setup_logging(args.verbose)
    run_dir = _resolve_run_dir(args.run_dir, args.run_id, args.output_root)

    train_path = run_dir / "preprocessed_train_descriptors.csv"
    val_path = run_dir / "preprocessed_val_descriptors.csv"
    test_path = run_dir / "preprocessed_test_descriptors.csv"
    if not train_path.exists():
        raise SystemExit(f"Missing training data: {train_path}")
    val_arg = val_path if val_path.exists() else None

    features = _load_selected_features(run_dir)
    model_cfg = _load_model_config(run_dir)
    hpo_cfg = _load_hpo_defaults(run_dir)

    hpo_cfg = hpo_cfg.model_copy(
        update={
            "enabled": not args.disable_hpo,
            "max_hpo_rounds": min(max(args.max_rounds, 1), 3),
            "max_candidates_per_round": args.max_candidates,
            "n_jobs": args.n_jobs,
            **({"cv_folds": args.cv_folds} if args.cv_folds is not None else {}),
        }
    )

    import pandas as pd

    train_df = pd.read_csv(train_path)
    n_train = len(train_df)
    n_features = len(features)

    logging.info("Run directory: %s", run_dir)
    logging.info("Training compounds: %d", n_train)
    logging.info("Selected features: %d", n_features)
    logging.info("HPO enabled: %s, max rounds: %d", hpo_cfg.enabled, hpo_cfg.max_hpo_rounds)

    def log_callback(msg: str) -> None:
        logging.info(msg)

    grid_proposer = None
    use_openai_proposer = hpo_cfg.enabled and not args.no_openai
    if use_openai_proposer:
        grid_proposer = _make_logging_grid_proposer()
    elif hpo_cfg.enabled:
        logging.info("OpenAI disabled; using deterministic fallback grids.")

    result = run_iterative_hyperparameter_optimization(
        train_path,
        features,
        model_cfg,
        hpo_cfg,
        run_dir,
        grid_proposer=grid_proposer,
        log_callback=log_callback,
        n_features=n_features,
        n_train_samples=n_train,
        val_path=val_arg,
    )

    _print_round_proposals(result, skip_agent_proposal=use_openai_proposer)
    _print_summary(run_dir, result)

    if args.run_final_model:
        if not test_path.exists():
            raise SystemExit(f"--run-final-model requires {test_path}")

        from qsar_agent.tools.final_model import train_and_evaluate_final_model

        final_cfg = ModelConfig(**result.final_model_config)
        hpo_metadata = {
            "enabled": result.enabled,
            "max_rounds": result.max_rounds,
            "rounds_completed": result.rounds_completed,
            "final_model_source": result.final_selection.source if result.final_selection else "baseline",
            "final_params": result.final_selection.params if result.final_selection else model_cfg.model_dump(),
            "baseline_assessment": (
                result.baseline_assessment.model_dump() if result.baseline_assessment else {}
            ),
            "final_assessment": result.final_assessment.model_dump() if result.final_assessment else {},
        }
        logging.info("Training final model and evaluating on external test set...")
        modeling = train_and_evaluate_final_model(
            train_path,
            test_path,
            run_dir,
            features,
            final_cfg,
            hpo_metadata=hpo_metadata,
            val_path=val_arg,
        )
        print(f"\nFinal model saved: {modeling.model_path}")
        print(f"Train R²: {modeling.train_metrics.r2:.4f}")
        if modeling.val_metrics is not None:
            print(f"Val R²:   {modeling.val_metrics.r2:.4f}")
        print(f"Test R²:  {modeling.test_metrics.r2:.4f}")


if __name__ == "__main__":
    main()
