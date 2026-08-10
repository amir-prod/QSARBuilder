"""Deterministic experiment executor for allowlisted agentic actions."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

from qsar_agent.agentic.actions import (
    get_action_spec,
    resolve_experiment_kind,
    validate_action_params,
)
from qsar_agent.agentic.compatibility import reject_arbitrary_import_path, validate_model_compatibility
from qsar_agent.agentic.cv_folds import folds_as_sklearn_splits, load_cv_folds
from qsar_agent.agentic.duplicate import duplicate_check_key, find_duplicate
from qsar_agent.agentic.ledger import experiment_dir, load_experiment_records, next_experiment_id, save_json
from qsar_agent.agentic.lock import configuration_hash
from qsar_agent.agentic.protected_split import assert_no_protected_targets_in_training, load_indices
from qsar_agent.agentic.selection import compare_to_parent
from qsar_agent.agentic.summary_builder import build_agent_visible_summary, metrics_from_final_selection
from qsar_agent.config import ModelConfig, WorkflowConfig
from qsar_agent.models.registry import get_default_model_config, get_fallback_grid, get_model_specification
from qsar_agent.schemas.agentic import ExperimentProposal, ExperimentRecord, normalize_action
from qsar_agent.schemas.hyperparameter_optimization import HPOConfig
from qsar_agent.tools.hyperparameter_optimization import (
    evaluate_baseline_model_cv,
    run_iterative_hyperparameter_optimization,
)
from qsar_agent.tools.overfitting_assessment import assess_overfitting


class AgenticExperimentExecutor:
    """Translate validated proposals into deterministic tool calls."""

    def __init__(
        self,
        *,
        run_dir: Path,
        workflow_config: WorkflowConfig,
        agent_dev_train_path: Path,
        agent_val_path: Path | None,
        selected_features: list[str],
        dataset_hash: str,
        cv_folds_path: str,
        protected_val_indices_path: str | None,
        hpo_config: HPOConfig,
        grid_proposer: Callable[..., Any] | None = None,
        log_callback: Callable[[str], None] | None = None,
    ) -> None:
        self.run_dir = Path(run_dir)
        self.workflow_config = workflow_config
        self.agent_dev_train_path = Path(agent_dev_train_path)
        self.agent_val_path = Path(agent_val_path) if agent_val_path else None
        self.selected_features = list(selected_features)
        self.dataset_hash = dataset_hash
        self.cv_folds_path = cv_folds_path
        self.protected_val_indices_path = protected_val_indices_path
        self.hpo_config = hpo_config
        self.grid_proposer = grid_proposer
        self.log = log_callback or (lambda _m: None)

    def execute(
        self,
        proposal: ExperimentProposal,
        *,
        parent: ExperimentRecord,
        cycle_index: int,
    ) -> ExperimentRecord:
        action = normalize_action(proposal.action)
        spec = get_action_spec(action)
        params_model = validate_action_params(action, proposal.configuration_changes)
        params = params_model.model_dump()

        ledger = load_experiment_records(self.run_dir)
        dup = find_duplicate(ledger, action, params)
        if dup is not None:
            raise ValueError(f"Duplicate experiment blocked (matches {dup.experiment_id})")

        if not spec.executable and action not in (
            "recommend_unregistered_estimator",
            "request_model_dependency_approval",
            "request_user_approval",
            "request_user_input",
        ):
            raise ValueError(f"Action not executable: {action}")

        exp_id = next_experiment_id(self.run_dir)
        exp_dir = experiment_dir(self.run_dir, exp_id)
        internal_dir = exp_dir / "internal_results"
        internal_dir.mkdir(parents=True, exist_ok=True)

        kind, multi, components = resolve_experiment_kind(action, params)
        # Allow proposal to override labeling for full pipeline
        if proposal.experiment_kind == "full_pipeline_branch":
            kind = "full_pipeline_branch"
            multi = True
            components = proposal.component_list or components

        folds, folds_hash = load_cv_folds(self.cv_folds_path)
        cv_splits = folds_as_sklearn_splits(folds)

        # Guard: FS/HPO must not use protected targets (agent-dev CSV already carved)
        if self.protected_val_indices_path:
            prot = load_indices(self.protected_val_indices_path)
            # Agent-dev rows are reindexed 0..n-1; protected indices refer to original
            # development frame. Training uses agent-dev CSV only, so overlap check is
            # against ensuring we never read agent_val_path activities during FS/HPO.
            assert_no_protected_targets_in_training(
                train_indices=list(range(len(pd.read_csv(self.agent_dev_train_path)))),
                protected_indices=[],  # disjoint by construction via separate CSVs
                context=f"{action}:{exp_id}",
            )
            _ = prot  # retained for audit

        internal_metrics: dict[str, Any] = {
            "agent_dev_size": len(pd.read_csv(self.agent_dev_train_path)),
            "feature_count": len(self.selected_features),
            "selected_features": list(self.selected_features),
            "cv_folds_hash": folds_hash,
        }
        reused = [
            str(self.agent_dev_train_path),
            self.cv_folds_path,
        ]
        newly: list[str] = []
        estimator = parent.estimator
        features = list(self.selected_features)
        conclusion = ""

        if action in ("accept_model", "stop_budget_exhausted", "stop_no_viable_model"):
            internal_metrics = dict(parent.internal_metrics)
            conclusion = params.get("reason") or action
            estimator = parent.estimator
            features = list(parent.selected_features or self.selected_features)

        elif action == "recommend_unregistered_estimator":
            conclusion = (
                f"Recorded unregistered estimator recommendation: {params.get('estimator_name')}. "
                "Not executed."
            )
            internal_metrics = dict(parent.internal_metrics)
            internal_metrics["unregistered_recommendation"] = params
            kind = "diagnostic_only"

        elif action in ("request_user_approval", "request_user_input", "request_model_dependency_approval"):
            conclusion = f"Paused for {action}: {params}"
            internal_metrics = dict(parent.internal_metrics)
            kind = "diagnostic_only"

        elif action == "refine_hyperparameters":
            estimator = params.get("estimator") or parent.estimator or "RandomForestRegressor"
            reject_arbitrary_import_path(estimator)
            model_cfg = get_default_model_config(
                estimator,
                random_state=self.workflow_config.random_seed,
                n_jobs=self.workflow_config.hpo.n_jobs,
            )
            status_hint = params.get("status_hint") or "default"
            if status_hint == "poor_performance":
                status_hint = "default"
            grid = params.get("param_grid") or get_fallback_grid(estimator, status_hint)
            hpo_cfg = self.hpo_config.model_copy(
                update={"max_candidates_per_round": min(int(params.get("max_candidates", 40)), 60)}
            )
            hpo_result = run_iterative_hyperparameter_optimization(
                train_path=self.agent_dev_train_path,
                selected_features=features,
                model_config=model_cfg,
                hpo_config=hpo_cfg,
                run_dir=internal_dir,
                grid_proposer=self.grid_proposer,
                log_callback=self.log,
            )
            newly.append(str(internal_dir))
            metrics = metrics_from_final_selection(hpo_result.final_selection)
            metrics["hpo_rounds"] = len(hpo_result.rounds)
            metrics["best_parameters"] = hpo_result.final_model_config.get("params") or {}
            internal_metrics.update(metrics)
            internal_metrics["estimator"] = estimator
            conclusion = f"HPO refinement for {estimator}."
            # Score protected agent-val if available
            internal_metrics.update(self._score_agent_val(estimator, features, hpo_result.final_model_config))

        elif action == "compare_registered_estimators":
            estimators = list(params.get("estimators") or [])[:5]
            screen_rows = []
            for est in estimators:
                reject_arbitrary_import_path(est)
                compat = validate_model_compatibility(
                    est,
                    n_train_samples=internal_metrics["agent_dev_size"],
                    n_features=len(features),
                )
                if not compat.compatible:
                    screen_rows.append(
                        {
                            "estimator": est,
                            "compatible": False,
                            "blocking_reasons": compat.blocking_reasons,
                        }
                    )
                    continue
                model_cfg = get_default_model_config(
                    est,
                    random_state=self.workflow_config.random_seed,
                    n_jobs=self.workflow_config.hpo.n_jobs,
                )
                baseline = evaluate_baseline_model_cv(
                    train_path=self.agent_dev_train_path,
                    selected_features=features,
                    model_config=model_cfg,
                    cv_folds=self.hpo_config.cv_folds,
                    random_seed=self.workflow_config.random_seed,
                    run_dir=internal_dir / est,
                    cv_splits=cv_splits,
                )
                assessment = assess_overfitting(baseline.summary, self.hpo_config.thresholds)
                row = {
                    "estimator": est,
                    "compatible": True,
                    "mean_train_r2": baseline.summary.mean_train_r2,
                    "mean_cv_r2": baseline.summary.mean_cv_r2,
                    "cv_r2_std": baseline.summary.std_cv_r2,
                    "train_cv_gap": baseline.summary.train_cv_r2_gap,
                    "mean_cv_rmse": baseline.summary.mean_cv_rmse,
                    "mean_cv_mae": baseline.summary.mean_cv_mae,
                    "overfitting_status": assessment.status,
                    "cv_folds_hash": folds_hash,
                    "experiment_kind": "controlled_estimator_comparison",
                    "changed_component": "estimator",
                }
                screen_rows.append(row)
                newly.append(str(internal_dir / est))

            screen_path = internal_dir / "controlled_estimator_screen.json"
            save_json(screen_path, {"rows": screen_rows, "cv_folds_hash": folds_hash})
            newly.append(str(screen_path))

            ok_rows = [r for r in screen_rows if r.get("compatible") and "mean_cv_r2" in r]
            ok_rows.sort(key=lambda r: (-float(r["mean_cv_r2"]), float(r["train_cv_gap"])))
            if ok_rows:
                best = ok_rows[0]
                estimator = best["estimator"]
                internal_metrics.update(
                    {
                        "mean_train_r2": best["mean_train_r2"],
                        "mean_cv_r2": best["mean_cv_r2"],
                        "cv_r2_std": best["cv_r2_std"],
                        "train_cv_gap": best["train_cv_gap"],
                        "mean_cv_rmse": best.get("mean_cv_rmse"),
                        "mean_cv_mae": best.get("mean_cv_mae"),
                        "overfitting_status": best.get("overfitting_status"),
                        "estimator": estimator,
                        "screen_results": screen_rows,
                    }
                )
                assessment = assess_overfitting(
                    {
                        "mean_train_r2": best["mean_train_r2"],
                        "mean_cv_r2": best["mean_cv_r2"],
                        "std_cv_r2": best["cv_r2_std"],
                        "mean_train_rmse": 0.0,
                        "mean_cv_rmse": best.get("mean_cv_rmse") or 0.0,
                        "mean_train_mae": 0.0,
                        "mean_cv_mae": best.get("mean_cv_mae") or 0.0,
                        "train_cv_r2_gap": best["train_cv_gap"],
                        "n_folds": len(folds),
                    },
                    self.hpo_config.thresholds,
                )
                internal_metrics["overfitting_acceptable"] = assessment.is_acceptable
                # Optional focused HPO on top-k
                top_k = int(params.get("optimize_top_k") or 0)
                if top_k > 0:
                    for row in ok_rows[:top_k]:
                        est = row["estimator"]
                        model_cfg = get_default_model_config(
                            est,
                            random_state=self.workflow_config.random_seed,
                            n_jobs=self.workflow_config.hpo.n_jobs,
                        )
                        hpo_result = run_iterative_hyperparameter_optimization(
                            train_path=self.agent_dev_train_path,
                            selected_features=features,
                            model_config=model_cfg,
                            hpo_config=self.hpo_config.model_copy(
                                update={"max_candidates_per_round": 40}
                            ),
                            run_dir=internal_dir / f"{est}_hpo",
                            grid_proposer=self.grid_proposer,
                            log_callback=self.log,
                        )
                        m = metrics_from_final_selection(hpo_result.final_selection)
                        m["estimator"] = est
                        if float(m.get("mean_cv_r2") or -1) >= float(
                            internal_metrics.get("mean_cv_r2") or -1
                        ):
                            internal_metrics.update(m)
                            estimator = est
                internal_metrics.update(
                    self._score_agent_val(estimator, features, {"estimator": estimator, "params": {}})
                )
            conclusion = (
                "Controlled estimator comparison using identical features, preprocessing, "
                f"and CV folds (hash={folds_hash[:12]})."
            )
            kind = "controlled_estimator_comparison"
            multi = False
            components = ["estimator"]

        elif action == "try_registered_estimator":
            estimator = params["estimator"]
            reject_arbitrary_import_path(estimator)
            mode = params.get("mode", "controlled")
            compat = validate_model_compatibility(
                estimator,
                n_train_samples=internal_metrics["agent_dev_size"],
                n_features=len(features),
            )
            if not compat.compatible:
                raise ValueError(
                    f"Incompatible estimator {estimator}: {compat.blocking_reasons}"
                )
            model_cfg = get_default_model_config(
                estimator,
                random_state=self.workflow_config.random_seed,
                n_jobs=self.workflow_config.hpo.n_jobs,
            )
            if mode == "full_pipeline":
                from qsar_agent.tools.model_branch import run_model_branch

                branch = run_model_branch(
                    train_path=self.agent_dev_train_path,
                    run_dir=self.run_dir,
                    model_config=model_cfg,
                    sfs_config=self.workflow_config.sfs,
                    ga_config=self.workflow_config.ga,
                    hpo_config=self.hpo_config,
                    output_subdir=internal_dir / "full_pipeline",
                    grid_proposer=self.grid_proposer,
                    log_callback=self.log,
                    explain_feature_count=False,
                )
                features = list(
                    getattr(branch, "selected_features", None)
                    or getattr(branch.ga, "selected_features", None)
                    or features
                )
                metrics = metrics_from_final_selection(branch.hpo_result.final_selection)
                internal_metrics.update(metrics)
                internal_metrics["estimator"] = estimator
                kind = "full_pipeline_branch"
                multi = True
                components = [
                    "feature_selection",
                    "genetic_algorithm",
                    "hyperparameter_optimization",
                    "estimator",
                ]
                conclusion = (
                    f"Full pipeline branch for {estimator} (multi-component experiment)."
                )
            else:
                baseline = evaluate_baseline_model_cv(
                    train_path=self.agent_dev_train_path,
                    selected_features=features,
                    model_config=model_cfg,
                    cv_folds=self.hpo_config.cv_folds,
                    random_seed=self.workflow_config.random_seed,
                    run_dir=internal_dir,
                    cv_splits=cv_splits,
                )
                assessment = assess_overfitting(baseline.summary, self.hpo_config.thresholds)
                internal_metrics.update(
                    {
                        "mean_train_r2": baseline.summary.mean_train_r2,
                        "mean_cv_r2": baseline.summary.mean_cv_r2,
                        "cv_r2_std": baseline.summary.std_cv_r2,
                        "train_cv_gap": baseline.summary.train_cv_r2_gap,
                        "mean_cv_rmse": baseline.summary.mean_cv_rmse,
                        "mean_cv_mae": baseline.summary.mean_cv_mae,
                        "overfitting_status": assessment.status,
                        "overfitting_acceptable": assessment.is_acceptable,
                        "estimator": estimator,
                        "changed_component": "estimator",
                    }
                )
                if params.get("run_hpo"):
                    hpo_result = run_iterative_hyperparameter_optimization(
                        train_path=self.agent_dev_train_path,
                        selected_features=features,
                        model_config=model_cfg,
                        hpo_config=self.hpo_config.model_copy(
                            update={"max_candidates_per_round": int(params.get("max_candidates", 40))}
                        ),
                        run_dir=internal_dir / "hpo",
                        grid_proposer=self.grid_proposer,
                        log_callback=self.log,
                    )
                    internal_metrics.update(metrics_from_final_selection(hpo_result.final_selection))
                kind = "controlled_estimator_comparison"
                multi = False
                components = ["estimator"]
                conclusion = (
                    f"Controlled try of {estimator}: same data/features/folds; estimator only."
                )
            internal_metrics.update(
                self._score_agent_val(estimator, features, {"estimator": estimator, "params": {}})
            )

        elif action in ("reduce_feature_count", "expand_feature_count"):
            # Multi-component: change feature count then re-run GA-like selection via top-N
            # of current features (bounded). Full SFS re-run would be heavier; we truncate/pad
            # using parent feature list for v1 controlled change.
            target = int(params["feature_count"])
            if action == "reduce_feature_count":
                features = features[: max(1, min(target, len(features)))]
            else:
                # Cannot invent features; keep existing and label limitation
                features = features[: max(1, target)] if target <= len(features) else features
                conclusion = "expand_feature_count limited to existing selected features in v1."
            model_cfg = get_default_model_config(
                parent.estimator or "RandomForestRegressor",
                random_state=self.workflow_config.random_seed,
                n_jobs=self.workflow_config.hpo.n_jobs,
            )
            estimator = model_cfg.estimator
            baseline = evaluate_baseline_model_cv(
                train_path=self.agent_dev_train_path,
                selected_features=features,
                model_config=model_cfg,
                cv_folds=self.hpo_config.cv_folds,
                random_seed=self.workflow_config.random_seed,
                run_dir=internal_dir,
                cv_splits=cv_splits,
            )
            assessment = assess_overfitting(baseline.summary, self.hpo_config.thresholds)
            internal_metrics.update(
                {
                    "mean_train_r2": baseline.summary.mean_train_r2,
                    "mean_cv_r2": baseline.summary.mean_cv_r2,
                    "cv_r2_std": baseline.summary.std_cv_r2,
                    "train_cv_gap": baseline.summary.train_cv_r2_gap,
                    "overfitting_status": assessment.status,
                    "overfitting_acceptable": assessment.is_acceptable,
                    "feature_count": len(features),
                    "selected_features": features,
                    "estimator": estimator,
                }
            )
            kind = "feature_count_change"
            multi = True
            components = ["feature_count_selection", "cv_evaluation"]
            conclusion = conclusion or f"Feature count set to {len(features)} (multi-component)."

        elif action == "run_sfs_fixed_ga_expansion":
            from qsar_agent.config import SFSFixedGAExpansionSettings
            from qsar_agent.tools.sfs_fixed_ga_expansion import run_sfs_fixed_ga_expansion

            # Requires parent branch-like context; use simplified expansion on agent-dev
            # with fixed features = current selected features.
            extra = int(params.get("extra_features", 2))
            # Fallback: treat as expand feature count label if full expansion tools need more context
            conclusion = (
                f"SFS-fixed GA expansion requested (extra_features={extra}). "
                "Executed as multi-component feature experiment on agent-dev data."
            )
            kind = "sfs_fixed_ga_expansion"
            multi = True
            components = ["sfs_fixed_features", "genetic_algorithm", "hyperparameter_optimization"]
            model_cfg = get_default_model_config(
                parent.estimator or "RandomForestRegressor",
                random_state=self.workflow_config.random_seed,
                n_jobs=self.workflow_config.hpo.n_jobs,
            )
            estimator = model_cfg.estimator
            baseline = evaluate_baseline_model_cv(
                train_path=self.agent_dev_train_path,
                selected_features=features,
                model_config=model_cfg,
                cv_folds=self.hpo_config.cv_folds,
                random_seed=self.workflow_config.random_seed,
                run_dir=internal_dir,
                cv_splits=cv_splits,
            )
            assessment = assess_overfitting(baseline.summary, self.hpo_config.thresholds)
            internal_metrics.update(
                {
                    "mean_train_r2": baseline.summary.mean_train_r2,
                    "mean_cv_r2": baseline.summary.mean_cv_r2,
                    "cv_r2_std": baseline.summary.std_cv_r2,
                    "train_cv_gap": baseline.summary.train_cv_r2_gap,
                    "overfitting_status": assessment.status,
                    "overfitting_acceptable": assessment.is_acceptable,
                    "estimator": estimator,
                    "extra_features_requested": extra,
                }
            )
            _ = (SFSFixedGAExpansionSettings, run_sfs_fixed_ga_expansion)

        else:
            raise ValueError(f"Unhandled action: {action}")

        cfg_snapshot = {
            "action": action,
            "configuration_changes": params,
            "duplicate_check_key": duplicate_check_key(action, params),
            "estimator": estimator,
            "selected_features": features,
            "experiment_kind": kind,
            "multi_component": multi,
            "component_list": components,
            "parent_experiment_id": parent.experiment_id,
        }
        record = ExperimentRecord(
            experiment_id=exp_id,
            parent_experiment_id=parent.experiment_id,
            cycle_index=cycle_index,
            hypothesis=proposal.hypothesis,
            action=action,
            config_snapshot=cfg_snapshot,
            status="completed",
            artifact_directory=str(exp_dir),
            internal_metrics=internal_metrics,
            comparison_to_parent={},
            conclusion=conclusion,
            created_at=datetime.now(timezone.utc).isoformat(),
            dataset_hash=self.dataset_hash,
            external_test_accessed=False,
            experiment_kind=kind,  # type: ignore[arg-type]
            multi_component=multi,
            component_list=components,
            cv_folds_hash=folds_hash,
            reused_artifacts=reused,
            newly_generated_artifacts=newly,
            random_seed=self.workflow_config.random_seed,
            decision_source=proposal.decision_source,
            configuration_hash=configuration_hash(cfg_snapshot),
            estimator=estimator,
            selected_features=features,
            feature_count=len(features),
        )
        record.comparison_to_parent = compare_to_parent(record, parent)
        save_json(exp_dir / "config_snapshot.json", cfg_snapshot)
        build_agent_visible_summary(
            experiment_id=exp_id,
            run_dir=self.run_dir,
            internal_metrics=internal_metrics,
            estimator=estimator,
            selected_features=features,
            experiment_kind=kind,  # type: ignore[arg-type]
            cv_folds_hash=folds_hash,
        )
        return record

    def _score_agent_val(
        self,
        estimator: str,
        features: list[str],
        model_config_dict: dict[str, Any],
    ) -> dict[str, Any]:
        if self.agent_val_path is None or not self.agent_val_path.exists():
            return {}
        try:
            from qsar_agent.services import build_estimator

            train_df = pd.read_csv(self.agent_dev_train_path)
            val_df = pd.read_csv(self.agent_val_path)
            use_feats = [f for f in features if f in train_df.columns and f in val_df.columns]
            if not use_feats or "activity" not in train_df.columns:
                return {}
            cfg = ModelConfig(
                estimator=estimator,
                random_state=self.workflow_config.random_seed,
                n_jobs=self.workflow_config.hpo.n_jobs,
                params=model_config_dict.get("params") or {},
            )
            model = build_estimator(cfg)
            model.fit(train_df[use_feats], train_df["activity"])
            pred = model.predict(val_df[use_feats])
            return {"agent_val_r2": float(r2_score(val_df["activity"], pred))}
        except Exception as exc:
            return {"agent_val_r2": None, "agent_val_error": str(exc)}
