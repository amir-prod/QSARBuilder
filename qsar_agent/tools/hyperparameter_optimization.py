"""Agent-guided hyperparameter optimization (training set only)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold, RandomizedSearchCV

from qsar_agent.config import ModelConfig
from qsar_agent.models.registry import (
    baseline_params_from_config,
    count_grid_combinations,
    get_fallback_grid,
    model_simplicity_score,
    params_to_model_config,
    sanitize_param_grid,
)
from qsar_agent.schemas.hyperparameter_optimization import (
    AgentGridProposal,
    BaselineCVResult,
    CandidateResult,
    CVSummary,
    FinalModelSelection,
    FoldMetrics,
    GridSanitizationResult,
    HPOConfig,
    HPORoundResult,
    HPOResult,
    ModelSource,
    OverfittingThresholds,
)
from qsar_agent.services import build_estimator
from qsar_agent.services.artifact_manager import save_json
from qsar_agent.services.plotting import plot_hpo_round_performance, plot_hpo_summary
from qsar_agent.tools.combined_score import combined_r2
from qsar_agent.tools.overfitting_assessment import assess_overfitting

# Re-export registry helpers used by tests and agents.
__all__ = [
    "count_grid_combinations",
    "get_fallback_grid",
    "sanitize_param_grid",
    "select_final_model_config",
    "select_best_across_models",
    "run_iterative_hyperparameter_optimization",
    "run_hyperparameter_search",
    "evaluate_baseline_model_cv",
]


def _load_xy(train_path: str | Path, selected_features: list[str]) -> tuple[pd.DataFrame, pd.Series]:
    train_df = pd.read_csv(train_path)
    for feat in selected_features:
        if feat not in train_df.columns:
            raise ValueError(f"Selected feature not in training data: {feat}")
    return train_df[selected_features], train_df["activity"]


def _score_holdout_val(
    train_path: str | Path,
    val_path: str | Path | None,
    selected_features: list[str],
    model_config: ModelConfig | None,
) -> float | None:
    """Fit on full train and score the held-out validation set. Never uses test."""
    if val_path is None:
        return None
    X_train, y_train = _load_xy(train_path, selected_features)
    X_val, y_val = _load_xy(val_path, selected_features)
    model = build_estimator(model_config)
    model.fit(X_train, y_train)
    return float(r2_score(y_val, model.predict(X_val)))


def _combined_from_summary(summary: CVSummary) -> float:
    return combined_r2(summary.mean_cv_r2, summary.holdout_val_r2)


def _summary_from_folds(folds: list[FoldMetrics]) -> CVSummary:
    return CVSummary(
        mean_train_r2=float(np.mean([f.train_r2 for f in folds])),
        mean_cv_r2=float(np.mean([f.val_r2 for f in folds])),
        std_cv_r2=float(np.std([f.val_r2 for f in folds])),
        mean_train_rmse=float(np.mean([f.train_rmse for f in folds])),
        mean_cv_rmse=float(np.mean([f.val_rmse for f in folds])),
        mean_train_mae=float(np.mean([f.train_mae for f in folds])),
        mean_cv_mae=float(np.mean([f.val_mae for f in folds])),
        train_cv_r2_gap=float(
            np.mean([f.train_r2 for f in folds]) - np.mean([f.val_r2 for f in folds])
        ),
        n_folds=len(folds),
    )


def evaluate_baseline_model_cv(
    train_path: str | Path,
    selected_features: list[str],
    model_config: ModelConfig | None = None,
    cv_folds: int = 5,
    random_seed: int = 42,
    run_dir: Path | None = None,
) -> BaselineCVResult:
    """K-fold CV on training data only using GA-selected features."""
    X, y = _load_xy(train_path, selected_features)
    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=random_seed)

    fold_metrics: list[FoldMetrics] = []
    for fold_idx, (tr_idx, val_idx) in enumerate(kf.split(X)):
        model = build_estimator(model_config)
        X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]
        model.fit(X_tr, y_tr)
        y_tr_pred = model.predict(X_tr)
        y_val_pred = model.predict(X_val)
        fold_metrics.append(
            FoldMetrics(
                fold=fold_idx + 1,
                train_r2=float(r2_score(y_tr, y_tr_pred)),
                val_r2=float(r2_score(y_val, y_val_pred)),
                train_rmse=float(np.sqrt(mean_squared_error(y_tr, y_tr_pred))),
                val_rmse=float(np.sqrt(mean_squared_error(y_val, y_val_pred))),
                train_mae=float(mean_absolute_error(y_tr, y_tr_pred)),
                val_mae=float(mean_absolute_error(y_val, y_val_pred)),
            )
        )

    summary = _summary_from_folds(fold_metrics)
    fold_path = ""
    summary_path = ""
    if run_dir is not None:
        fold_path = str(run_dir / "baseline_cv_metrics.csv")
        pd.DataFrame([f.model_dump() for f in fold_metrics]).to_csv(fold_path, index=False)
        summary_path = str(run_dir / "baseline_cv_summary.json")
        save_json(run_dir / "baseline_cv_summary.json", summary.model_dump())

    return BaselineCVResult(
        fold_metrics=fold_metrics,
        summary=summary,
        fold_metrics_path=fold_path,
        summary_path=summary_path,
    )


def _cv_results_to_candidates(cv_results: dict, sanitized_grid: dict) -> list[CandidateResult]:
    candidates: list[CandidateResult] = []
    n = len(cv_results["params"])
    mean_test = cv_results["mean_test_score"]
    mean_train = cv_results["mean_train_score"]
    std_test = cv_results["std_test_score"]

    for i in range(n):
        gap = float(mean_train[i] - mean_test[i])
        candidates.append(
            CandidateResult(
                rank=0,
                params=cv_results["params"][i],
                mean_train_r2=float(mean_train[i]),
                mean_cv_r2=float(mean_test[i]),
                std_cv_r2=float(std_test[i]),
                mean_train_rmse=0.0,
                mean_cv_rmse=0.0,
                mean_train_mae=0.0,
                mean_cv_mae=0.0,
                train_cv_r2_gap=gap,
            )
        )

    candidates.sort(key=lambda c: c.mean_cv_r2, reverse=True)
    for rank, cand in enumerate(candidates, start=1):
        cand.rank = rank
    if candidates:
        candidates[0].is_best = True
    return candidates


def run_hyperparameter_search(
    train_path: str | Path,
    selected_features: list[str],
    param_grid: dict[str, list[Any]],
    hpo_config: HPOConfig,
    model_config: ModelConfig | None = None,
    run_dir: Path | None = None,
    round_index: int = 1,
) -> tuple[list[CandidateResult], dict[str, Any], CVSummary, GridSanitizationResult]:
    """Grid or randomized search on training data only."""
    X, y = _load_xy(train_path, selected_features)
    base_cfg = model_config or ModelConfig()
    sanitization = sanitize_param_grid(
        base_cfg.estimator,
        param_grid,
        max_candidates=hpo_config.max_candidates_per_round,
        random_seed=hpo_config.random_seed,
        n_features=len(selected_features),
        n_train_samples=len(X),
    )

    estimator = build_estimator(
        ModelConfig(
            estimator=base_cfg.estimator,
            random_state=base_cfg.random_state,
            n_jobs=base_cfg.n_jobs,
        )
    )
    cv = KFold(
        n_splits=hpo_config.cv_folds,
        shuffle=True,
        random_state=hpo_config.random_seed,
    )

    grid = sanitization.sanitized_grid
    if sanitization.used_randomized_search and hpo_config.use_randomized_search_fallback:
        search = RandomizedSearchCV(
            estimator,
            grid,
            n_iter=hpo_config.max_candidates_per_round,
            scoring="r2",
            cv=cv,
            n_jobs=hpo_config.n_jobs,
            random_state=hpo_config.random_seed,
            return_train_score=True,
            error_score="raise",
        )
    else:
        search = GridSearchCV(
            estimator,
            grid,
            scoring="r2",
            cv=cv,
            n_jobs=hpo_config.n_jobs,
            return_train_score=True,
            error_score="raise",
        )

    search.fit(X, y)
    candidates = _cv_results_to_candidates(search.cv_results_, grid)
    best_params = search.best_params_
    best = candidates[0] if candidates else None
    best_summary = CVSummary(
        mean_train_r2=best.mean_train_r2 if best else 0.0,
        mean_cv_r2=best.mean_cv_r2 if best else 0.0,
        std_cv_r2=best.std_cv_r2 if best else 0.0,
        mean_train_rmse=0.0,
        mean_cv_rmse=0.0,
        mean_train_mae=0.0,
        mean_cv_mae=0.0,
        train_cv_r2_gap=best.train_cv_r2_gap if best else 0.0,
        n_folds=hpo_config.cv_folds,
    )

    if run_dir is not None:
        prefix = f"hpo_round_{round_index}"
        pd.DataFrame([c.model_dump() for c in candidates]).to_csv(
            run_dir / f"{prefix}_search_results.csv", index=False
        )
        save_json(run_dir / f"{prefix}_best_params.json", best_params)
        save_json(run_dir / f"{prefix}_cv_summary.json", best_summary.model_dump())
        save_json(run_dir / f"{prefix}_grid_sanitization.json", sanitization.model_dump())
        plot_hpo_round_performance(
            candidates,
            run_dir / f"{prefix}_performance.png",
            run_dir / f"{prefix}_performance.svg",
            round_index,
        )

    return candidates, best_params, best_summary, sanitization


def select_final_model_config(
    baseline_summary: CVSummary,
    baseline_params: dict[str, Any],
    baseline_assessment,
    round_results: list[HPORoundResult],
    thresholds: OverfittingThresholds,
    estimator: str = "RandomForestRegressor",
) -> FinalModelSelection:
    """Choose final configuration using combined CV + holdout-validation R²."""
    candidates: list[dict[str, Any]] = [
        {
            "source": "baseline",
            "params": baseline_params,
            "summary": baseline_summary,
            "assessment": baseline_assessment,
        }
    ]
    for rr in round_results:
        candidates.append(
            {
                "source": f"hpo_round_{rr.round_index}",
                "params": rr.best_params,
                "summary": rr.best_cv_summary,
                "assessment": rr.assessment,
            }
        )

    acceptable = [c for c in candidates if c["assessment"].is_acceptable]
    pool = acceptable if acceptable else candidates
    best = max(pool, key=lambda c: _combined_from_summary(c["summary"]))
    best_combo = _combined_from_summary(best["summary"])
    se_threshold = best_combo - best["summary"].std_cv_r2

    within_se = [
        c for c in pool if _combined_from_summary(c["summary"]) >= se_threshold - 1e-9
    ]
    chosen = min(
        within_se,
        key=lambda c: model_simplicity_score(estimator, c["params"]),
    )
    chosen_combo = _combined_from_summary(chosen["summary"])

    warning = ""
    if not acceptable:
        warning = (
            "No acceptable model found after HPO; selected highest combined R² candidate. "
            "Final model may still be overfit, unstable, or poor-performing."
        )
    elif chosen not in acceptable:
        warning = "Selected model from acceptable pool with one-SE simplicity rule."

    val_txt = (
        f", holdout val R²={chosen['summary'].holdout_val_r2:.4f}"
        if chosen["summary"].holdout_val_r2 is not None
        else ""
    )
    rationale = (
        f"Selected {chosen['source']} with combined R²={chosen_combo:.4f} "
        f"(mean CV R²={chosen['summary'].mean_cv_r2:.4f}{val_txt}), "
        f"train-CV gap={chosen['summary'].train_cv_r2_gap:.4f}, "
        f"status={chosen['assessment'].status}. "
    )
    if acceptable:
        rationale += (
            f"Preferred acceptable models ({len(acceptable)}); applied one-SE rule "
            f"on combined R² (threshold >= {se_threshold:.4f}) with simplicity tie-break."
        )
    else:
        rationale += "No acceptable models; chose best combined R² with warning."

    source: ModelSource = chosen["source"]  # type: ignore[assignment]

    return FinalModelSelection(
        source=source,
        params=chosen["params"],
        cv_summary=chosen["summary"],
        assessment=chosen["assessment"],
        selection_rationale=rationale,
        warning=warning,
        compared_candidates=[
            {
                "source": c["source"],
                "mean_cv_r2": c["summary"].mean_cv_r2,
                "holdout_val_r2": c["summary"].holdout_val_r2,
                "combined_r2": _combined_from_summary(c["summary"]),
                "status": c["assessment"].status,
                "acceptable": c["assessment"].is_acceptable,
            }
            for c in candidates
        ],
    )


def select_best_across_models(
    candidates: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Pick the globally best model across RF and fallback branches.

    Each candidate dict must include:
      estimator, selected_features, final_selection (FinalModelSelection), model_config (ModelConfig)

    Optional: base_estimator (for simplicity scoring), is_expansion, expansion_label.
    """
    pool_items: list[dict[str, Any]] = []
    for cand in candidates:
        fs = cand["final_selection"]
        if fs is None:
            continue
        base_est = cand.get("base_estimator") or cand["estimator"]
        pool_items.append(
            {
                "estimator": cand["estimator"],
                "base_estimator": base_est,
                "selected_features": cand["selected_features"],
                "model_config": cand["model_config"],
                "source": fs.source,
                "params": fs.params,
                "summary": fs.cv_summary,
                "assessment": fs.assessment,
                "final_selection": fs,
                "is_expansion": bool(cand.get("is_expansion", False)),
                "expansion_label": str(cand.get("expansion_label", "")),
            }
        )

    if not pool_items:
        raise ValueError("No model candidates available for cross-model selection.")

    acceptable = [c for c in pool_items if c["assessment"].is_acceptable]
    pool = acceptable if acceptable else pool_items
    best = max(pool, key=lambda c: _combined_from_summary(c["summary"]))
    best_combo = _combined_from_summary(best["summary"])
    se_threshold = best_combo - best["summary"].std_cv_r2

    within_se = [
        c for c in pool if _combined_from_summary(c["summary"]) >= se_threshold - 1e-9
    ]
    chosen = min(
        within_se,
        key=lambda c: model_simplicity_score(c["base_estimator"], c["params"]),
    )
    chosen_combo = _combined_from_summary(chosen["summary"])

    warning = ""
    if not acceptable:
        warning = (
            "No acceptable model found across estimators; selected highest combined R² candidate. "
            "Final model may still be overfit, unstable, or poor-performing."
        )

    val_txt = (
        f", holdout val R²={chosen['summary'].holdout_val_r2:.4f}"
        if chosen["summary"].holdout_val_r2 is not None
        else ""
    )
    rationale = (
        f"Selected {chosen['estimator']} ({chosen['source']}) with "
        f"combined R²={chosen_combo:.4f} "
        f"(mean CV R²={chosen['summary'].mean_cv_r2:.4f}{val_txt}), "
        f"train-CV gap={chosen['summary'].train_cv_r2_gap:.4f}, "
        f"status={chosen['assessment'].status}. "
    )
    if acceptable:
        rationale += (
            f"Compared {len(pool_items)} model branch(es); "
            f"{len(acceptable)} acceptable; applied one-SE rule "
            f"on combined R² (threshold >= {se_threshold:.4f}) with simplicity tie-break."
        )
    else:
        rationale += (
            f"Compared {len(pool_items)} model branch(es); "
            "no acceptable models; chose best combined R² with warning."
        )

    compared_models = [
        {
            "estimator": c["estimator"],
            "source": c["source"],
            "mean_cv_r2": c["summary"].mean_cv_r2,
            "holdout_val_r2": c["summary"].holdout_val_r2,
            "combined_r2": _combined_from_summary(c["summary"]),
            "train_cv_r2_gap": c["summary"].train_cv_r2_gap,
            "status": c["assessment"].status,
            "acceptable": c["assessment"].is_acceptable,
            "n_features": len(c["selected_features"]),
            "is_expansion": c["is_expansion"],
        }
        for c in pool_items
    ]

    final_config = params_to_model_config(chosen["params"], chosen["model_config"]).model_dump()

    return {
        "winning_estimator": chosen["estimator"],
        "selected_features": chosen["selected_features"],
        "final_model_config": final_config,
        "final_selection": chosen["final_selection"],
        "selection_rationale": rationale,
        "warning": warning,
        "compared_models": compared_models,
        "winner_is_expansion": chosen["is_expansion"],
        "winner_expansion_label": chosen["expansion_label"],
    }


def _should_stop_hpo(
    assessment,
    baseline_cv: float,
    current_cv: float,
    hpo_config: HPOConfig,
) -> tuple[bool, str]:
    th = hpo_config.thresholds
    if assessment.is_acceptable:
        return True, "Model is acceptable after cross-validation assessment."
    improved = current_cv - baseline_cv >= hpo_config.min_cv_improvement
    gap_ok = assessment.train_cv_r2_gap <= th.overfit_gap_threshold
    std_ok = assessment.cv_r2_std <= th.cv_std_threshold
    if improved and gap_ok and std_ok and not assessment.is_overfit:
        return True, (
            f"CV R² improved by {current_cv - baseline_cv:.3f}, gap and variability within limits."
        )
    return False, ""


def run_iterative_hyperparameter_optimization(
    train_path: str | Path,
    selected_features: list[str],
    baseline_model_config: ModelConfig | None = None,
    hpo_config: HPOConfig | None = None,
    run_dir: Path | None = None,
    output_subdir: Path | None = None,
    grid_proposer: Callable[..., AgentGridProposal] | None = None,
    log_callback: Callable[[str], None] | None = None,
    n_features: int | None = None,
    n_train_samples: int | None = None,
    val_path: str | Path | None = None,
) -> HPOResult:
    """Full HPO controller: baseline CV, up to 3 agent-guided rounds, final selection."""
    cfg = hpo_config or HPOConfig()
    base = baseline_model_config or ModelConfig()
    parent_run_dir = Path(run_dir) if run_dir else Path(".")
    run_dir = Path(output_subdir) if output_subdir else parent_run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    logs: list[str] = []
    iteration_records: list[dict[str, Any]] = []
    fallback_events: list[dict[str, Any]] = []

    def log(msg: str) -> None:
        logs.append(msg)
        if log_callback:
            log_callback(msg)

    if not cfg.enabled:
        log("Hyperparameter optimization disabled; using baseline model configuration.")
        params = base.model_dump()
        return HPOResult(
            enabled=False,
            rounds_completed=0,
            max_rounds=cfg.max_hpo_rounds,
            final_model_config=params,
        )

    log("Baseline CV diagnostics started.")
    baseline_cv = evaluate_baseline_model_cv(
        train_path,
        selected_features,
        base,
        cfg.cv_folds,
        cfg.random_seed,
        run_dir,
    )
    log("Baseline CV diagnostics completed.")

    baseline_val_r2 = _score_holdout_val(
        train_path, val_path, selected_features, base
    )
    if baseline_val_r2 is not None:
        baseline_cv = baseline_cv.model_copy(
            update={
                "summary": baseline_cv.summary.model_copy(
                    update={"holdout_val_r2": baseline_val_r2}
                )
            }
        )
        log(f"Baseline holdout validation R² = {baseline_val_r2:.3f}.")

    # Always provide dataset size to the LLM grid proposer.
    if n_features is None:
        n_features = len(selected_features)
    if n_train_samples is None:
        n_train_samples = int(pd.read_csv(train_path).shape[0])
    log(
        f"HPO dataset context for grid proposals: "
        f"n_train_samples={n_train_samples}, n_features={n_features}."
    )

    baseline_params = baseline_params_from_config(base)

    baseline_assessment = assess_overfitting(baseline_cv.summary, cfg.thresholds)
    save_json(run_dir / "baseline_overfitting_assessment.json", baseline_assessment.model_dump())
    log(
        f"Overfitting assessment: {baseline_assessment.status}. "
        f"Train-CV R² gap = {baseline_assessment.train_cv_r2_gap:.3f}."
    )

    round_results: list[HPORoundResult] = []
    hpo_triggered = not baseline_assessment.is_acceptable
    trigger_reason = baseline_assessment.explanation if hpo_triggered else "Baseline acceptable."

    if baseline_assessment.is_acceptable:
        log("Baseline model acceptable. Skipping HPO rounds.")
        final_selection = select_final_model_config(
            baseline_cv.summary,
            baseline_params,
            baseline_assessment,
            [],
            cfg.thresholds,
            estimator=base.estimator,
        )
        final_assessment = baseline_assessment
    else:
        log(f"Model appears {baseline_assessment.status}. Starting HPO.")
        previous_results: list[HPORoundResult] = []
        last_assessment = baseline_assessment
        last_cv = baseline_cv.summary.mean_cv_r2
        stop = False

        for round_idx in range(1, cfg.max_hpo_rounds + 1):
            log(f"HPO round {round_idx}/{cfg.max_hpo_rounds} started.")
            iteration_records.append(
                {"round": round_idx, "phase": "start", "baseline_cv_r2": baseline_cv.summary.mean_cv_r2}
            )

            if grid_proposer:
                try:
                    proposal = grid_proposer(
                        round_index=round_idx,
                        model_type=base.estimator,
                        baseline_assessment=baseline_assessment,
                        previous_hpo_results=previous_results,
                        constraints={
                            "max_candidates": cfg.max_candidates_per_round,
                            "n_features": n_features,
                            "n_train_samples": n_train_samples,
                        },
                    )
                except Exception as exc:
                    fallback_events.append(
                        {"round": round_idx, "reason": str(exc), "type": "agent_exception"}
                    )
                    proposal = None
            else:
                proposal = None

            if proposal is None:
                status = last_assessment.status
                grid = get_fallback_grid(base.estimator, status)
                proposal = AgentGridProposal(
                    round_index=round_idx,
                    reasoning_summary=f"Deterministic fallback grid for status={status}.",
                    search_strategy="fallback",
                    proposed_grid=grid,
                    expected_effect_on_overfitting="Regularization per fallback template.",
                    expected_effect_on_underfitting="Capacity per fallback template.",
                    computational_cost_estimate=str(count_grid_combinations(grid)),
                    warnings=["OpenAI unavailable or proposal failed; using fallback grid."],
                )
                fallback_events.append(
                    {"round": round_idx, "reason": "fallback_grid", "status": status}
                )

            agent_grid_path = run_dir / f"hpo_round_{round_idx}_agent_grid.json"
            save_json(agent_grid_path, proposal.model_dump())
            explanation_md = (
                f"# HPO Round {round_idx} Agent Grid Proposal\n\n"
                f"**Strategy:** {proposal.search_strategy}\n\n"
                f"{proposal.reasoning_summary}\n\n"
                f"**Expected overfitting effect:** {proposal.expected_effect_on_overfitting}\n\n"
                f"**Expected underfitting effect:** {proposal.expected_effect_on_underfitting}\n\n"
                f"**Cost estimate:** {proposal.computational_cost_estimate}\n"
            )
            agent_explanation_path = run_dir / f"hpo_round_{round_idx}_agent_explanation.md"
            agent_explanation_path.write_text(explanation_md, encoding="utf-8")

            candidates, best_params, best_summary, sanitization = run_hyperparameter_search(
                train_path,
                selected_features,
                proposal.proposed_grid,
                cfg,
                base,
                run_dir,
                round_idx,
            )
            round_model = params_to_model_config(best_params, base)
            round_val_r2 = _score_holdout_val(
                train_path, val_path, selected_features, round_model
            )
            if round_val_r2 is not None:
                best_summary = best_summary.model_copy(
                    update={"holdout_val_r2": round_val_r2}
                )
                log(f"HPO round {round_idx} holdout validation R² = {round_val_r2:.3f}.")

            round_assessment = assess_overfitting(best_summary, cfg.thresholds)
            save_json(
                run_dir / f"hpo_round_{round_idx}_overfitting_assessment.json",
                round_assessment.model_dump(),
            )

            rr = HPORoundResult(
                round_index=round_idx,
                agent_proposal=proposal,
                sanitization=sanitization,
                candidates=candidates,
                best_params=best_params,
                best_cv_summary=best_summary,
                assessment=round_assessment,
                candidates_searched=len(candidates),
                agent_grid_path=str(agent_grid_path),
                agent_explanation_path=str(agent_explanation_path),
                search_results_path=str(run_dir / f"hpo_round_{round_idx}_search_results.csv"),
                best_params_path=str(run_dir / f"hpo_round_{round_idx}_best_params.json"),
                cv_summary_path=str(run_dir / f"hpo_round_{round_idx}_cv_summary.json"),
                assessment_path=str(
                    run_dir / f"hpo_round_{round_idx}_overfitting_assessment.json"
                ),
                performance_png_path=str(run_dir / f"hpo_round_{round_idx}_performance.png"),
                performance_svg_path=str(run_dir / f"hpo_round_{round_idx}_performance.svg"),
                grid_sanitization_path=str(
                    run_dir / f"hpo_round_{round_idx}_grid_sanitization.json"
                ),
            )
            round_results.append(rr)
            previous_results.append(rr)

            log(
                f"HPO round {round_idx}/{cfg.max_hpo_rounds} completed. "
                f"Best CV R² = {best_summary.mean_cv_r2:.3f}. "
                f"Gap = {round_assessment.train_cv_r2_gap:.3f}."
            )
            iteration_records.append(
                {
                    "round": round_idx,
                    "candidates": len(candidates),
                    "best_cv_r2": best_summary.mean_cv_r2,
                    "gap": round_assessment.train_cv_r2_gap,
                    "status": round_assessment.status,
                    "best_params": best_params,
                }
            )

            should_stop, stop_reason = _should_stop_hpo(
                round_assessment,
                baseline_cv.summary.mean_cv_r2,
                best_summary.mean_cv_r2,
                cfg,
            )
            if should_stop:
                log(f"Model acceptable after HPO round {round_idx}. Stopping. {stop_reason}")
                stop = True
                break

            last_assessment = round_assessment
            last_cv = best_summary.mean_cv_r2

            if round_idx == cfg.max_hpo_rounds:
                log(
                    f"HPO round {round_idx}/{cfg.max_hpo_rounds} completed. "
                    "Maximum HPO rounds reached."
                )

        final_selection = select_final_model_config(
            baseline_cv.summary,
            baseline_params,
            baseline_assessment,
            round_results,
            cfg.thresholds,
            estimator=base.estimator,
        )
        if round_results:
            final_assessment = round_results[-1].assessment
            for rr in round_results:
                if f"hpo_round_{rr.round_index}" == final_selection.source:
                    final_assessment = rr.assessment
                    break
        else:
            final_assessment = baseline_assessment

    save_json(run_dir / "final_overfitting_assessment.json", final_assessment.model_dump())

    final_selection_path = run_dir / "hpo_final_selection.json"
    save_json(final_selection_path, final_selection.model_dump())
    selection_md = (
        f"# Final HPO Model Selection\n\n"
        f"**Source:** {final_selection.source}\n\n"
        f"{final_selection.selection_rationale}\n\n"
    )
    if final_selection.warning:
        selection_md += f"**Warning:** {final_selection.warning}\n"
    selection_explanation_path = run_dir / "hpo_final_selection_explanation.md"
    selection_explanation_path.write_text(selection_md, encoding="utf-8")

    summary_rows = [
        {
            "source": "baseline",
            "mean_train_r2": baseline_cv.summary.mean_train_r2,
            "mean_cv_r2": baseline_cv.summary.mean_cv_r2,
            "std_cv_r2": baseline_cv.summary.std_cv_r2,
            "train_cv_r2_gap": baseline_cv.summary.train_cv_r2_gap,
            "selected": final_selection.source == "baseline",
        }
    ]
    for rr in round_results:
        summary_rows.append(
            {
                "source": f"hpo_round_{rr.round_index}",
                "mean_train_r2": rr.best_cv_summary.mean_train_r2,
                "mean_cv_r2": rr.best_cv_summary.mean_cv_r2,
                "std_cv_r2": rr.best_cv_summary.std_cv_r2,
                "train_cv_r2_gap": rr.best_cv_summary.train_cv_r2_gap,
                "selected": final_selection.source == f"hpo_round_{rr.round_index}",
            }
        )
    summary_csv_path = run_dir / "hpo_all_rounds_summary.csv"
    pd.DataFrame(summary_rows).to_csv(summary_csv_path, index=False)
    pd.DataFrame(summary_rows).to_csv(run_dir / "hpo_summary.csv", index=False)
    plot_hpo_summary(
        summary_rows,
        run_dir / "hpo_summary.png",
        run_dir / "hpo_summary.svg",
        final_selection.source,
    )

    md_lines = [
        "# Hyperparameter Optimization Log\n",
        "HPO round 0/3: Baseline model evaluated.",
        f"Decision: {trigger_reason}",
    ]
    for rr in round_results:
        md_lines.append(
            f"\nHPO round {rr.round_index}/{cfg.max_hpo_rounds}: "
            f"{rr.agent_proposal.search_strategy if rr.agent_proposal else 'search'}."
        )
        md_lines.append(f"Candidates searched: {rr.candidates_searched}.")
        md_lines.append(f"Best CV R²: {rr.best_cv_summary.mean_cv_r2:.2f}.")
        md_lines.append(f"Train-CV R² gap: {rr.assessment.train_cv_r2_gap:.2f}.")
        md_lines.append(f"Assessment: {rr.assessment.status}.")
    md_lines.append(f"\nFinal selected model: {final_selection.source}.")
    if final_selection.warning:
        md_lines.append(f"Warning: {final_selection.warning}")
    iteration_md = "\n".join(md_lines)
    iteration_md_path = run_dir / "hpo_iteration_log.md"
    iteration_md_path.write_text(iteration_md, encoding="utf-8")

    iteration_json = {
        "max_hpo_rounds": cfg.max_hpo_rounds,
        "rounds_completed": len(round_results),
        "baseline_acceptable": baseline_assessment.is_acceptable,
        "hpo_triggered": hpo_triggered,
        "hpo_trigger_reason": trigger_reason,
        "iteration_records": iteration_records,
        "logs": logs,
        "final_source": final_selection.source,
    }
    iteration_json_path = run_dir / "hpo_iteration_log.json"
    save_json(iteration_json_path, iteration_json)

    fallback_path = ""
    if fallback_events:
        fb_path = run_dir / "hpo_agent_fallback_log.json"
        save_json(fb_path, {"events": fallback_events})
        fallback_path = str(fb_path)

    final_config = params_to_model_config(final_selection.params, base).model_dump()
    log(f"Final selected model: {final_selection.source}.")

    return HPOResult(
        enabled=True,
        rounds_completed=len(round_results),
        max_rounds=cfg.max_hpo_rounds,
        baseline_cv=baseline_cv,
        baseline_assessment=baseline_assessment,
        baseline_assessment_path=str(run_dir / "baseline_overfitting_assessment.json"),
        final_assessment=final_assessment,
        final_assessment_path=str(run_dir / "final_overfitting_assessment.json"),
        rounds=round_results,
        final_selection=final_selection,
        hpo_triggered=hpo_triggered,
        hpo_trigger_reason=trigger_reason,
        iteration_log_json_path=str(iteration_json_path),
        iteration_log_md_path=str(iteration_md_path),
        final_selection_json_path=str(final_selection_path),
        final_selection_explanation_path=str(selection_explanation_path),
        all_rounds_summary_path=str(summary_csv_path),
        summary_plot_png_path=str(run_dir / "hpo_summary.png"),
        summary_plot_svg_path=str(run_dir / "hpo_summary.svg"),
        summary_csv_path=str(summary_csv_path),
        agent_fallback_log_path=fallback_path,
        final_model_config=final_config,
    )
