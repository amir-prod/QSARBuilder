"""Allowlisted deterministic tools dispatched by the LangGraph execute node."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from qsar_agent.agentic.ids import make_experiment_id
from qsar_agent.agentic.ledger import append_ledger_row, existing_experiment_ids
from qsar_agent.agentic.ranking import rank_candidates
from qsar_agent.agentic.requirements import evaluate_requirements, requirements_from_config
from qsar_agent.agentic.sealing import (
    SealedTestAccessError,
    assert_development_phase,
    assert_no_test_paths,
    path_looks_like_test,
    phase_value,
)
from qsar_agent.config import GAConfig, ModelConfig, WorkflowConfig
from qsar_agent.models.registry import SUPPORTED_ESTIMATORS, params_to_model_config, sanitize_param_grid
from qsar_agent.schemas.agentic import (
    APPROVED_TOOL_NAMES,
    CapabilityRequest,
    PipelinePhase,
    ToolResult,
)
from qsar_agent.schemas.hyperparameter_optimization import HPOConfig
from qsar_agent.services.artifact_manager import save_json
from qsar_agent.tools.applicability_domain import calculate_applicability_domain
from qsar_agent.tools.descriptor_calculation import META_COLUMNS
from qsar_agent.tools.development_eval import evaluate_feature_subset, oof_predictions
from qsar_agent.tools.error_analysis import analyze_errors
from qsar_agent.tools.feature_selection_methods import select_features
from qsar_agent.tools.feature_stability import consensus_subset, summarize_stability
from qsar_agent.tools.final_model import train_and_evaluate_final_model
from qsar_agent.tools.genetic_algorithm import run_genetic_algorithm
from qsar_agent.tools.hyperparameter_optimization import (
    evaluate_baseline_model_cv,
    run_hyperparameter_search,
)
from qsar_agent.tools.outlier_persistence import (
    mask_compounds,
    persistent_outliers_from_oof,
    proposal_from_report,
)
from qsar_agent.tools.sequential_feature_selection import run_sequential_feature_selection


def agent_dir(run_dir: str | Path) -> Path:
    path = Path(run_dir) / "agent_results"
    path.mkdir(parents=True, exist_ok=True)
    return path


def experiment_dir(run_dir: str | Path, experiment_id: str) -> Path:
    path = agent_dir(run_dir) / "experiments" / experiment_id
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_workflow_config(run_dir: str | Path) -> WorkflowConfig:
    candidates = [
        Path(run_dir) / "final_report" / "configs" / "workflow_config.json",
        Path(run_dir) / "workflow_config.json",
    ]
    for path in candidates:
        if path.is_file():
            return WorkflowConfig.model_validate(json.loads(path.read_text(encoding="utf-8")))
    return WorkflowConfig()


def development_paths(run_dir: Path) -> dict[str, Path]:
    return {
        "train": run_dir / "preprocessed_train_descriptors.csv",
        "val": run_dir / "preprocessed_val_descriptors.csv",
        "test": run_dir / "preprocessed_test_descriptors.csv",
        "split": run_dir / "split_assignments.csv",
    }


def _model_config(config: WorkflowConfig, args: dict[str, Any]) -> ModelConfig:
    base = config.model.model_copy()
    estimator = args.get("estimator") or args.get("model")
    if estimator:
        base = base.model_copy(update={"estimator": estimator})
    params = args.get("params") or args.get("hyperparameters")
    if params:
        base = params_to_model_config(params, base)
    return base


def _pareto_pick(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    """Smallest subset within one SE of the best eligible CV R²."""
    if not candidates:
        raise ValueError("No GA candidates to pick from.")
    best = max(candidates, key=lambda c: (c["metrics"].get("cv_r2") is not None, c["metrics"].get("cv_r2") or float("-inf")))
    best_r2 = best["metrics"].get("cv_r2") or 0.0
    best_std = best["metrics"].get("cv_r2_std") or 0.0
    threshold = best_r2 - best_std
    eligible = [c for c in candidates if (c["metrics"].get("cv_r2") or float("-inf")) >= threshold - 1e-12]
    return min(eligible, key=lambda c: (c["metrics"].get("feature_count") or 10**9, -(c["metrics"].get("cv_r2") or 0)))


def _tool_result_from_eval(
    *,
    experiment_id: str,
    tool_name: str,
    arguments: dict[str, Any],
    eval_payload: dict[str, Any],
    extra: dict[str, Any] | None = None,
    runtime: float | None = None,
    parent: str | None = None,
) -> ToolResult:
    metrics = eval_payload.get("metrics") or {}
    features = eval_payload.get("selected_features") or []
    return ToolResult(
        experiment_id=experiment_id,
        tool_name=tool_name,
        arguments=arguments,
        metrics={k: (None if v is None else float(v)) for k, v in metrics.items() if isinstance(v, (int, float)) or v is None},
        selected_features=list(features),
        artifact_paths={
            "oof_predictions": str(eval_payload.get("oof_predictions_path") or ""),
        },
        runtime_seconds=runtime,
        parent_experiment_id=parent,
        extra=extra or {},
    )


def run_model_search(ctx: dict[str, Any], args: dict[str, Any], experiment_id: str) -> ToolResult:
    config: WorkflowConfig = ctx["config"]
    run_dir: Path = ctx["run_dir"]
    paths = development_paths(run_dir)
    out = experiment_dir(run_dir, experiment_id)
    model_cfg = _model_config(config, args)
    features = args.get("selected_features") or ctx.get("selected_features")
    if not features:
        raise ValueError("run_model_search requires selected_features.")
    started = time.perf_counter()
    grid = args.get("param_grid")
    if grid:
        sanitization = sanitize_param_grid(
            model_cfg.estimator,
            grid,
            max_candidates=config.hpo.max_candidates_per_round,
        )
        if not sanitization.sanitized_grid:
            raise ValueError("No allowlisted hyperparameters remained after sanitization.")
        save_json(out / "grid_sanitization.json", sanitization.model_dump())
        hpo_cfg = HPOConfig(
            cv_folds=int(args.get("cv_folds") or config.hpo.cv_folds or 5),
            max_candidates_per_round=config.hpo.max_candidates_per_round,
            random_seed=config.random_seed,
            n_jobs=config.hpo.n_jobs,
        )
        run_hyperparameter_search(
            paths["train"],
            features,
            sanitization.sanitized_grid,
            hpo_cfg,
            model_config=model_cfg,
            run_dir=out,
        )
    payload = evaluate_feature_subset(
        paths["train"],
        paths["val"] if paths["val"].is_file() else None,
        list(features),
        out,
        model_config=model_cfg,
        cv_folds=int(args.get("cv_folds") or config.hpo.cv_folds or 5),
        random_seed=config.random_seed,
    )
    return _tool_result_from_eval(
        experiment_id=experiment_id,
        tool_name="run_model_search",
        arguments=args,
        eval_payload=payload,
        extra={"estimator": model_cfg.estimator},
        runtime=time.perf_counter() - started,
        parent=ctx.get("parent_id"),
    )


def run_feature_selection_search(ctx: dict[str, Any], args: dict[str, Any], experiment_id: str) -> ToolResult:
    config: WorkflowConfig = ctx["config"]
    run_dir: Path = ctx["run_dir"]
    paths = development_paths(run_dir)
    out = experiment_dir(run_dir, experiment_id)
    method = str(args.get("method") or "genetic_algorithm")
    n_features = int(args.get("n_features") or args.get("k") or 5)
    model_cfg = _model_config(config, args)
    started = time.perf_counter()
    extra: dict[str, Any] = {"method": method}
    selected: list[str]

    if method in {"sequential_forward", "sequential_backward"}:
        sfs_dir = out / "sfs"
        sfs_dir.mkdir(exist_ok=True)
        result = run_sequential_feature_selection(
            paths["train"],
            sfs_dir,
            max_features=n_features,
            cv_folds=int(args.get("cv_folds") or config.sfs.cv_folds),
            model_config=model_cfg,
            random_seed=config.sfs.random_seed,
            n_jobs=config.sfs.n_jobs,
            val_path=None,
            forward=method == "sequential_forward",
        )
        row = next(r for r in result.results if r.n_features == min(n_features, result.max_features_evaluated))
        selected = list(row.selected_features)
    elif method == "genetic_algorithm":
        ga_settings = config.agentic_improvement.ga
        seeds = list(args.get("seeds") or ga_settings.seeds)
        lo, hi = ga_settings.feature_count_range
        n_features = max(lo, min(hi, n_features))
        pop = int(args.get("population_size") or ga_settings.population_size)
        gens = int(args.get("generations") or ga_settings.generations)
        objective = args.get("objective")
        per_seed: list[dict[str, Any]] = []
        subsets: list[list[str]] = []
        for seed in seeds:
            seed_dir = out / f"ga_seed_{seed}"
            seed_dir.mkdir(exist_ok=True)
            ga_cfg = GAConfig(
                population_size=min(pop, 40),
                n_generations=min(gens, 20),
                cv_folds=int(args.get("cv_folds") or config.ga.cv_folds),
                n_jobs=config.ga.n_jobs,
                random_seed=int(seed),
            )
            ga = run_genetic_algorithm(
                paths["train"],
                seed_dir,
                number_of_features=n_features,
                ga_config=ga_cfg,
                model_config=model_cfg,
                val_path=None,
                objective=objective if isinstance(objective, dict) else ({"name": objective} if objective else None),
            )
            eval_seed = evaluate_feature_subset(
                paths["train"],
                paths["val"] if paths["val"].is_file() else None,
                ga.selected_features,
                seed_dir / "outer_eval",
                model_config=model_cfg,
                cv_folds=ga_cfg.cv_folds,
                random_seed=int(seed),
            )
            per_seed.append(
                {
                    "seed": seed,
                    "selected_features": ga.selected_features,
                    "metrics": eval_seed["metrics"],
                    "inner_fitness": ga.best_fitness,
                }
            )
            subsets.append(list(ga.selected_features))
        stability = summarize_stability(subsets)
        consensus = consensus_subset(subsets, min_frequency=0.5, max_size=n_features)
        if not consensus:
            consensus = subsets[0]
        cons_eval = evaluate_feature_subset(
            paths["train"],
            paths["val"] if paths["val"].is_file() else None,
            consensus,
            out / "consensus_eval",
            model_config=model_cfg,
            cv_folds=int(args.get("cv_folds") or config.ga.cv_folds),
            random_seed=config.random_seed,
        )
        pool = per_seed + [
            {"seed": "consensus", "selected_features": consensus, "metrics": cons_eval["metrics"]}
        ]
        picked = _pareto_pick(pool)
        selected = list(picked["selected_features"])
        extra.update(
            {
                "per_seed": per_seed,
                "stability": stability.model_dump(),
                "consensus_features": consensus,
                "consensus_metrics": cons_eval["metrics"],
                "selected_by": picked.get("seed"),
                "ga_converged_stable": stability.stability_status == "stable",
            }
        )
        save_json(out / "ga_multi_seed.json", extra)
        payload = evaluate_feature_subset(
            paths["train"],
            paths["val"] if paths["val"].is_file() else None,
            selected,
            out / "outer_eval",
            model_config=model_cfg,
            cv_folds=int(args.get("cv_folds") or config.ga.cv_folds),
            random_seed=config.random_seed,
        )
        result = _tool_result_from_eval(
            experiment_id=experiment_id,
            tool_name="run_feature_selection_search",
            arguments=args,
            eval_payload=payload,
            extra=extra,
            runtime=time.perf_counter() - started,
            parent=ctx.get("parent_id"),
        )
        return result
    else:
        allow_latent = bool(
            config.agentic_improvement.allow_latent_components
            or config.agentic_improvement.requirements.allow_latent_components
        )
        sel = select_features(
            paths["train"],
            method=method,  # type: ignore[arg-type]
            n_features=n_features,
            out_dir=out / "selector",
            model_config=model_cfg,
            random_seed=config.random_seed,
            allow_latent=allow_latent,
        )
        selected = list(sel["selected_features"])
        extra.update({k: v for k, v in sel.items() if k != "selected_features"})

    if method in {"pca", "pls"}:
        raise ValueError(
            "PCA/PLS components require a latent-representation pipeline; "
            "request_new_capability if interpretability allows transforming the matrix."
        )

    payload = evaluate_feature_subset(
        paths["train"],
        paths["val"] if paths["val"].is_file() else None,
        selected,
        out / "outer_eval",
        model_config=model_cfg,
        cv_folds=int(args.get("cv_folds") or config.sfs.cv_folds),
        random_seed=config.random_seed,
    )
    extra["selected_features"] = selected
    return _tool_result_from_eval(
        experiment_id=experiment_id,
        tool_name="run_feature_selection_search",
        arguments=args,
        eval_payload=payload,
        extra=extra,
        runtime=time.perf_counter() - started,
        parent=ctx.get("parent_id"),
    )


def run_representation_experiment(ctx: dict[str, Any], args: dict[str, Any], experiment_id: str) -> ToolResult:
    config: WorkflowConfig = ctx["config"]
    run_dir: Path = ctx["run_dir"]
    paths = development_paths(run_dir)
    out = experiment_dir(run_dir, experiment_id)
    df = pd.read_csv(paths["train"])
    names = [c for c in df.columns if c not in META_COLUMNS]
    prefix = args.get("column_prefix")
    backend = str(args.get("backend") or args.get("representation") or "").lower()
    if prefix:
        selected = [c for c in names if str(c).startswith(str(prefix))]
    elif backend == "rdkit":
        selected = [c for c in names if "mordred" not in str(c).lower()]
    elif backend == "mordred":
        selected = [c for c in names if "mordred" in str(c).lower()] or names
    else:
        stride = max(1, int(args.get("stride") or 2))
        selected = names[::stride]
    if not selected:
        selected = names[: min(10, len(names))]
    features = selected[: int(args.get("max_features") or min(20, len(selected)))]
    started = time.perf_counter()
    payload = evaluate_feature_subset(
        paths["train"],
        paths["val"] if paths["val"].is_file() else None,
        features,
        out,
        model_config=_model_config(config, args),
        cv_folds=int(args.get("cv_folds") or config.sfs.cv_folds),
        random_seed=config.random_seed,
    )
    return _tool_result_from_eval(
        experiment_id=experiment_id,
        tool_name="run_representation_experiment",
        arguments=args,
        eval_payload=payload,
        extra={"representation": backend or prefix or "strided"},
        runtime=time.perf_counter() - started,
        parent=ctx.get("parent_id"),
    )


def run_robustness_analysis(ctx: dict[str, Any], args: dict[str, Any], experiment_id: str) -> ToolResult:
    config: WorkflowConfig = ctx["config"]
    run_dir: Path = ctx["run_dir"]
    paths = development_paths(run_dir)
    out = experiment_dir(run_dir, experiment_id)
    features = args.get("selected_features") or ctx.get("selected_features")
    if not features:
        raise ValueError("run_robustness_analysis requires selected_features.")
    repeats = int(args.get("repeats") or 3)
    started = time.perf_counter()
    summaries = []
    for i in range(repeats):
        seed = int(config.random_seed) + 17 * (i + 1)
        payload = evaluate_feature_subset(
            paths["train"],
            paths["val"] if paths["val"].is_file() else None,
            list(features),
            out / f"repeat_{i+1}",
            model_config=_model_config(config, args),
            cv_folds=int(args.get("cv_folds") or 3),
            random_seed=seed,
        )
        summaries.append(payload["metrics"])
    mean_cv = float(sum(s["cv_r2"] for s in summaries) / len(summaries))
    payload = {
        "metrics": {
            "cv_r2": mean_cv,
            "cv_r2_std": float(pd.Series([s["cv_r2"] for s in summaries]).std(ddof=0)),
            "cv_rmse": float(sum(s["cv_rmse"] for s in summaries) / len(summaries)),
            "cv_mae": float(sum(s["cv_mae"] for s in summaries) / len(summaries)),
            "mean_cv_fold_train_r2": float(sum(s["mean_cv_fold_train_r2"] for s in summaries) / len(summaries)),
            "oof_cv_r2": mean_cv,
            "cv_fold_train_val_gap": float(sum(s["cv_fold_train_val_gap"] for s in summaries) / len(summaries)),
            "val_r2": summaries[-1].get("val_r2"),
            "feature_count": len(features),
        },
        "selected_features": list(features),
        "oof_predictions_path": "",
    }
    save_json(out / "robustness_repeats.json", summaries)
    return _tool_result_from_eval(
        experiment_id=experiment_id,
        tool_name="run_robustness_analysis",
        arguments=args,
        eval_payload=payload,
        extra={"repeats": repeats},
        runtime=time.perf_counter() - started,
        parent=ctx.get("parent_id"),
    )


def run_residual_analysis(ctx: dict[str, Any], args: dict[str, Any], experiment_id: str) -> ToolResult:
    run_dir: Path = ctx["run_dir"]
    out = experiment_dir(run_dir, experiment_id)
    oof_path = args.get("oof_predictions_path") or ctx.get("oof_predictions_path")
    if not oof_path:
        features = args.get("selected_features") or ctx.get("selected_features")
        paths = development_paths(run_dir)
        oof = oof_predictions(paths["train"], list(features), _model_config(ctx["config"], args))
        oof_path = out / "oof_predictions.csv"
        oof.to_csv(oof_path, index=False)
    df = pd.read_csv(oof_path)
    work = df.copy()
    if "split" not in work.columns:
        work["split"] = "oof"
    if "predicted_activity" not in work.columns:
        work["predicted_activity"] = work["activity"] - work["residual"]
    tmp_pred = out / "dev_predictions.csv"
    work.to_csv(tmp_pred, index=False)
    analysis = analyze_errors(tmp_pred, None, experiment_id)
    dumped = analysis.model_dump()
    dumped["largest_error_compounds"] = [
        row for row in dumped["largest_error_compounds"] if row.get("split") != "test"
    ]
    save_json(out / "residual_analysis.json", dumped)
    return ToolResult(
        experiment_id=experiment_id,
        tool_name="run_residual_analysis",
        arguments=args,
        extra={"error_analysis": dumped},
        artifact_paths={"residual_analysis": str(out / "residual_analysis.json")},
    )


def run_applicability_domain_analysis(ctx: dict[str, Any], args: dict[str, Any], experiment_id: str) -> ToolResult:
    config: WorkflowConfig = ctx["config"]
    run_dir: Path = ctx["run_dir"]
    paths = development_paths(run_dir)
    out = experiment_dir(run_dir, experiment_id)
    features = args.get("selected_features") or ctx.get("selected_features")
    if not features:
        raise ValueError("run_applicability_domain_analysis requires selected_features.")
    payload = evaluate_feature_subset(
        paths["train"],
        paths["val"] if paths["val"].is_file() else None,
        list(features),
        out / "fit",
        model_config=_model_config(config, args),
        cv_folds=int(args.get("cv_folds") or 3),
        random_seed=config.random_seed,
    )
    pred_path = out / "fit" / "oof_predictions.csv"
    pred = pd.read_csv(pred_path)
    if "split" not in pred.columns:
        pred["split"] = "train"
    else:
        pred["split"] = pred["split"].replace({"oof": "train"})
    pred_path_ad = out / "ad_predictions.csv"
    pred.to_csv(pred_path_ad, index=False)
    ad = calculate_applicability_domain(
        paths["train"],
        None,
        pred_path_ad,
        out,
        list(features),
        val_path=paths["val"] if paths["val"].is_file() else None,
    )
    coverage = float(ad.summary.train_in_domain_pct)
    metrics = dict(payload["metrics"])
    return ToolResult(
        experiment_id=experiment_id,
        tool_name="run_applicability_domain_analysis",
        arguments=args,
        metrics=metrics,
        selected_features=list(features),
        extra={"ad_coverage": coverage, "ad_summary": ad.summary.model_dump()},
        artifact_paths={"ad_report": ad.report_path, "williams": ad.williams_png_path},
    )


def detect_persistent_outliers(ctx: dict[str, Any], args: dict[str, Any], experiment_id: str) -> ToolResult:
    run_dir: Path = ctx["run_dir"]
    out = experiment_dir(run_dir, experiment_id)
    features = args.get("selected_features") or ctx.get("selected_features")
    paths = development_paths(run_dir)
    config: WorkflowConfig = ctx["config"]
    families = args.get("model_families") or [config.model.estimator, "SVR"]
    tables = []
    structural: dict[str, float] = {}
    for fam in families[:4]:
        cfg = config.model.model_copy(update={"estimator": fam}) if fam in SUPPORTED_ESTIMATORS else config.model
        oof = oof_predictions(paths["train"], list(features), cfg, cv_folds=3, random_seed=config.random_seed)
        tables.append(oof)
    reports = persistent_outliers_from_oof(tables, structural_flags=structural, model_family=str(families[0]))
    save_json(out / "persistent_outliers.json", [r.model_dump() for r in reports])
    return ToolResult(
        experiment_id=experiment_id,
        tool_name="detect_persistent_outliers",
        arguments=args,
        extra={"outliers": [r.model_dump() for r in reports[:20]]},
        artifact_paths={"persistent_outliers": str(out / "persistent_outliers.json")},
    )


def audit_compound(ctx: dict[str, Any], args: dict[str, Any], experiment_id: str) -> ToolResult:
    run_dir: Path = ctx["run_dir"]
    out = experiment_dir(run_dir, experiment_id)
    cid = str(args.get("compound_id") or "")
    paths = development_paths(run_dir)
    train = pd.read_csv(paths["train"])
    row = train[train["compound_id"].astype(str) == cid]
    payload = {
        "compound_id": cid,
        "in_train": bool(len(row)),
        "activity": None if row.empty else float(row.iloc[0]["activity"]),
    }
    save_json(out / "audit.json", payload)
    return ToolResult(experiment_id=experiment_id, tool_name="audit_compound", arguments=args, extra=payload)


def run_exclusion_sensitivity_analysis(ctx: dict[str, Any], args: dict[str, Any], experiment_id: str) -> ToolResult:
    config: WorkflowConfig = ctx["config"]
    run_dir: Path = ctx["run_dir"]
    out = experiment_dir(run_dir, experiment_id)
    if not args.get("approved") and not ctx.get("exclusion_approved"):
        raise PermissionError("Exclusion sensitivity analysis requires human approval.")
    compound_ids = list(args.get("compound_ids") or ([args["compound_id"]] if args.get("compound_id") else []))
    if not compound_ids:
        raise ValueError("No compounds specified for exclusion sensitivity.")
    features = args.get("selected_features") or ctx.get("selected_features")
    paths = development_paths(run_dir)
    train = pd.read_csv(paths["train"])
    reduced = mask_compounds(train, compound_ids)
    if len(reduced) >= len(train):
        raise ValueError("Exclusion did not remove any development rows.")
    tmp_train = out / "train_without_exclusions.csv"
    reduced.to_csv(tmp_train, index=False)
    val_path = paths["val"] if paths["val"].is_file() else None
    full = evaluate_feature_subset(paths["train"], val_path, list(features), out / "full", config.model, cv_folds=3, random_seed=config.random_seed)
    excl = evaluate_feature_subset(tmp_train, val_path, list(features), out / "excluded", config.model, cv_folds=3, random_seed=config.random_seed)
    delta = {
        "cv_r2": (excl["metrics"]["cv_r2"] or 0) - (full["metrics"]["cv_r2"] or 0),
        "train_cv_gap": (excl["metrics"]["cv_fold_train_val_gap"] or 0) - (full["metrics"]["cv_fold_train_val_gap"] or 0),
        "n_full": int(len(train)),
        "n_excluded": int(len(reduced)),
    }
    save_json(out / "exclusion_sensitivity.json", {"full": full["metrics"], "excluded": excl["metrics"], "delta": delta})
    robust = abs(delta["cv_r2"]) < 0.5  # always report; caller decides
    extra = {"delta": delta, "robust": robust, "compound_ids": compound_ids}
    return _tool_result_from_eval(
        experiment_id=experiment_id,
        tool_name="run_exclusion_sensitivity_analysis",
        arguments=args,
        eval_payload=excl,
        extra=extra,
        parent=ctx.get("parent_id"),
    )


def compare_experiments(ctx: dict[str, Any], args: dict[str, Any], experiment_id: str) -> ToolResult:
    config: WorkflowConfig = ctx["config"]
    experiments = args.get("experiments") or ctx.get("completed_experiments") or []
    rankings = rank_candidates(experiments, requirements_from_config(config))
    return ToolResult(
        experiment_id=experiment_id,
        tool_name="compare_experiments",
        arguments=args,
        extra={"rankings": [r.model_dump() for r in rankings]},
    )


def request_new_capability(ctx: dict[str, Any], args: dict[str, Any], experiment_id: str) -> ToolResult:
    req = CapabilityRequest(
        capability=str(args.get("capability") or "unspecified"),
        scientific_reason=str(args.get("scientific_reason") or args.get("reason") or ""),
        evidence_from_current_results=list(args.get("evidence_from_current_results") or []),
        required_inputs=list(args.get("required_inputs") or []),
        expected_outputs=list(args.get("expected_outputs") or []),
        existing_tools_considered=list(args.get("existing_tools_considered") or list(APPROVED_TOOL_NAMES)),
        why_existing_tools_are_insufficient=str(args.get("why_existing_tools_are_insufficient") or ""),
        leakage_risks=list(args.get("leakage_risks") or []),
        reproducibility_risks=list(args.get("reproducibility_risks") or []),
        compute_risks=list(args.get("compute_risks") or []),
        suggested_deterministic_implementation=str(args.get("suggested_deterministic_implementation") or ""),
        approval_required=True,
    )
    cap_dir = agent_dir(ctx["run_dir"]) / "capability_requests"
    cap_dir.mkdir(parents=True, exist_ok=True)
    path = cap_dir / f"{experiment_id}.json"
    save_json(path, req.model_dump())
    return ToolResult(
        experiment_id=experiment_id,
        tool_name="request_new_capability",
        arguments=args,
        extra=req.model_dump(),
        artifact_paths={"capability_request": str(path)},
    )


def freeze_pipeline(ctx: dict[str, Any], args: dict[str, Any], experiment_id: str) -> ToolResult:
    run_dir: Path = ctx["run_dir"]
    frozen = agent_dir(run_dir) / "frozen_pipeline"
    frozen.mkdir(parents=True, exist_ok=True)
    candidate = args.get("candidate") or ctx.get("current_best_candidate") or {}
    features = candidate.get("selected_features") or args.get("selected_features") or ctx.get("selected_features")
    save_json(frozen / "frozen_config.json", {"candidate": candidate, "selected_features": features})
    paths = development_paths(run_dir)
    if features:
        payload = evaluate_feature_subset(
            paths["train"],
            paths["val"] if paths["val"].is_file() else None,
            list(features),
            frozen / "refit",
            model_config=_model_config(ctx["config"], args),
            cv_folds=3,
            random_seed=ctx["config"].random_seed,
        )
        save_json(frozen / "refit_metrics.json", payload["metrics"])
    return ToolResult(
        experiment_id=experiment_id,
        tool_name="freeze_pipeline",
        arguments=args,
        selected_features=list(features or []),
        extra={"frozen_dir": str(frozen)},
        artifact_paths={"frozen_pipeline": str(frozen)},
    )


def evaluate_sealed_test(ctx: dict[str, Any], args: dict[str, Any], experiment_id: str) -> ToolResult:
    if phase_value(ctx.get("phase")) != PipelinePhase.FROZEN.value:
        raise SealedTestAccessError("evaluate_sealed_test requires phase FROZEN.")
    run_dir: Path = ctx["run_dir"]
    paths = development_paths(run_dir)
    if path_looks_like_test(paths["test"]) is False:
        pass
    out = agent_dir(run_dir) / "sealed_test"
    out.mkdir(parents=True, exist_ok=True)
    features = args.get("selected_features") or ctx.get("selected_features")
    if not features:
        raise ValueError("Frozen pipeline has no selected features.")
    result = train_and_evaluate_final_model(
        paths["train"],
        paths["test"],
        out,
        list(features),
        model_config=_model_config(ctx["config"], args),
        dataset_hash=ctx.get("dataset_hash") or "",
        val_path=paths["val"] if paths["val"].is_file() else None,
    )
    metrics = {
        "test_r2": result.test_metrics.r2,
        "test_rmse": result.test_metrics.rmse,
        "test_mae": result.test_metrics.mae,
        "train_r2": result.train_metrics.r2,
        "val_r2": None if result.val_metrics is None else result.val_metrics.r2,
    }
    save_json(out / "sealed_test_metrics.json", metrics)
    return ToolResult(
        experiment_id=experiment_id,
        tool_name="evaluate_sealed_test",
        arguments=args,
        metrics=metrics,
        selected_features=list(features),
        artifact_paths={"sealed_test": str(out)},
        extra={"confirmatory": True},
    )


TOOL_IMPLEMENTATIONS: dict[str, Callable[..., ToolResult]] = {
    "run_model_search": run_model_search,
    "run_feature_selection_search": run_feature_selection_search,
    "run_representation_experiment": run_representation_experiment,
    "run_robustness_analysis": run_robustness_analysis,
    "run_residual_analysis": run_residual_analysis,
    "run_applicability_domain_analysis": run_applicability_domain_analysis,
    "detect_persistent_outliers": detect_persistent_outliers,
    "audit_compound": audit_compound,
    "run_exclusion_sensitivity_analysis": run_exclusion_sensitivity_analysis,
    "compare_experiments": compare_experiments,
    "request_new_capability": request_new_capability,
    "freeze_pipeline": freeze_pipeline,
    "evaluate_sealed_test": evaluate_sealed_test,
}


def execute_tool(
    tool_name: str,
    arguments: dict[str, Any],
    *,
    run_dir: str | Path,
    state_phase: PipelinePhase | str,
    dataset_hash: str,
    development_split_hash: str,
    parent_id: str | None = None,
    selected_features: list[str] | None = None,
    completed_experiments: list[dict[str, Any]] | None = None,
    current_best: dict[str, Any] | None = None,
    exclusion_approved: bool = False,
    sealed_test_result: Any = None,
    oof_predictions_path: str | None = None,
) -> ToolResult:
    if tool_name not in APPROVED_TOOL_NAMES:
        raise ValueError(f"Tool {tool_name!r} is not allowlisted.")
    if tool_name not in ("evaluate_sealed_test", "freeze_pipeline", "request_new_capability"):
        assert_development_phase(state_phase, sealed_test_result=sealed_test_result)
        assert_no_test_paths(arguments, label="tool.arguments")
    run_dir = Path(run_dir)
    config = load_workflow_config(run_dir)
    experiment_id = make_experiment_id(
        tool_name=tool_name,
        arguments=arguments or {},
        dataset_hash=dataset_hash,
        development_split_hash=development_split_hash,
        parent_experiment_id=parent_id,
    )
    stored = agent_dir(run_dir) / "experiments" / experiment_id / "tool_result.json"
    if stored.is_file():
        return ToolResult.model_validate(json.loads(stored.read_text(encoding="utf-8")))
    if experiment_id in existing_experiment_ids(agent_dir(run_dir)) and stored.is_file():
        return ToolResult.model_validate(json.loads(stored.read_text(encoding="utf-8")))

    impl = TOOL_IMPLEMENTATIONS.get(tool_name)
    if impl is None:
        raise ValueError(f"No implementation for {tool_name}.")
    ctx = {
        "run_dir": run_dir,
        "config": config,
        "phase": state_phase,
        "parent_id": parent_id,
        "selected_features": selected_features,
        "completed_experiments": completed_experiments,
        "current_best_candidate": current_best,
        "exclusion_approved": exclusion_approved,
        "dataset_hash": dataset_hash,
        "oof_predictions_path": oof_predictions_path,
    }
    result = impl(ctx, arguments or {}, experiment_id)
    save_json(experiment_dir(run_dir, experiment_id) / "tool_result.json", result.model_dump(mode="json"))
    append_ledger_row(
        agent_dir(run_dir),
        {
            "experiment_id": result.experiment_id,
            "parent_experiment": parent_id or "",
            "tool_name": tool_name,
            "arguments": arguments,
            "dataset_hash": dataset_hash,
            "development_split_hash": development_split_hash,
            "selected_features": result.selected_features,
            "cv_r2": (result.metrics or {}).get("cv_r2"),
            "cv_r2_std": (result.metrics or {}).get("cv_r2_std"),
            "train_cv_gap": (result.metrics or {}).get("cv_fold_train_val_gap")
            or (result.metrics or {}).get("train_cv_r2_gap"),
            "refit_train_cv_gap": (result.metrics or {}).get("refit_train_cv_gap"),
            "cv_rmse": (result.metrics or {}).get("cv_rmse"),
            "cv_mae": (result.metrics or {}).get("cv_mae"),
            "val_r2": (result.metrics or {}).get("val_r2"),
            "feature_count": len(result.selected_features),
            "runtime_seconds": result.runtime_seconds,
            "artifact_paths": result.artifact_paths,
            "decision": "continue",
        },
    )
    return result
