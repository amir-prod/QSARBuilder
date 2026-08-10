"""Agentic improvement loop orchestration."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from qsar_agent.agentic.acceptance import evaluate_acceptance
from qsar_agent.agentic.agents.data_quality import run_data_quality_agent
from qsar_agent.agentic.agents.descriptor_feature import run_descriptor_feature_agent
from qsar_agent.agentic.agents.modeling import run_modeling_agent
from qsar_agent.agentic.agents.supervisor import plan_specialists_for_cycle, run_supervisor
from qsar_agent.agentic.agents.validation import run_validation_agent
from qsar_agent.agentic.approvals import create_pending_approval
from qsar_agent.agentic.artifact_view import AgentArtifactView
from qsar_agent.agentic.cv_folds import create_cv_folds, persist_cv_folds
from qsar_agent.agentic.events import append_event
from qsar_agent.agentic.executor import AgenticExperimentExecutor
from qsar_agent.agentic.hard_failures import flags_from_runtime
from qsar_agent.agentic.ledger import (
    agent_workspace,
    append_experiment_record,
    append_supervisor_decision,
    experiment_dir,
    load_experiment_records,
    save_json,
    save_project_state,
)
from qsar_agent.agentic.lock import (
    AgenticOptimizationForbiddenError,
    assert_agentic_optimization_allowed,
    configuration_hash,
    lock_model,
)
from qsar_agent.agentic.protected_split import (
    carve_agent_validation_split,
    persist_protected_split,
    subset_dataframe,
)
from qsar_agent.agentic.provider import AgentProvider, ProviderError
from qsar_agent.agentic.report import write_final_agent_report
from qsar_agent.agentic.routing import SPECIALIST_VALIDATION
from qsar_agent.agentic.selection import select_best_experiment
from qsar_agent.agentic.summary_builder import build_agent_visible_summary, metrics_from_final_selection
from qsar_agent.config import WorkflowConfig
from qsar_agent.schemas.agentic import (
    AgenticImprovementConfig,
    AgenticProjectState,
    ExperimentRecord,
)
from qsar_agent.schemas.hyperparameter_optimization import HPOConfig


class AgenticImprovementLoop:
    def __init__(
        self,
        *,
        run_dir: Path,
        workflow_config: WorkflowConfig,
        hpo_config: HPOConfig,
        development_train_path: Path,
        selected_features: list[str],
        dataset_hash: str,
        initial_estimator: str,
        initial_final_selection: Any,
        provider: AgentProvider | None = None,
        grid_proposer: Callable[..., Any] | None = None,
        log_callback: Callable[[str], None] | None = None,
        stop_check: Callable[[], bool] | None = None,
    ) -> None:
        self.run_dir = Path(run_dir)
        self.config = workflow_config
        self.agentic: AgenticImprovementConfig = workflow_config.agentic
        self.hpo_config = hpo_config
        self.development_train_path = Path(development_train_path)
        self.selected_features = list(selected_features)
        self.dataset_hash = dataset_hash
        self.initial_estimator = initial_estimator
        self.initial_final_selection = initial_final_selection
        self.provider = provider
        self.grid_proposer = grid_proposer
        self.log = log_callback or (lambda m: None)
        self.stop_check = stop_check or (lambda: False)

    def run(self) -> AgenticProjectState:
        agent_workspace(self.run_dir)
        max_cycles = min(self.agentic.max_cycles, self.agentic.hard_max_cycles)
        max_experiments = min(
            self.agentic.max_total_experiments, self.agentic.hard_max_experiments
        )

        # Protected agent-validation carve-out from development data
        dev_df = pd.read_csv(self.development_train_path)
        agent_dev_idx, agent_val_idx, split_meta = carve_agent_validation_split(
            dev_df,
            agent_validation_fraction=self.agentic.agent_validation_fraction,
            random_seed=self.config.random_seed,
        )
        paths = persist_protected_split(self.run_dir, agent_dev_idx, agent_val_idx, split_meta)
        agent_dev_df = subset_dataframe(dev_df, agent_dev_idx)
        agent_val_df = subset_dataframe(dev_df, agent_val_idx)
        agent_dev_path = agent_workspace(self.run_dir) / "protected_split" / "agent_dev_train.csv"
        agent_val_path = agent_workspace(self.run_dir) / "protected_split" / "agent_val.csv"
        agent_dev_df.to_csv(agent_dev_path, index=False)
        if len(agent_val_df):
            agent_val_df.to_csv(agent_val_path, index=False)
        else:
            agent_val_path = None  # type: ignore[assignment]

        folds = create_cv_folds(
            len(agent_dev_df),
            n_splits=self.hpo_config.cv_folds,
            random_seed=self.config.random_seed,
        )
        cv_folds_path, cv_folds_hash = persist_cv_folds(
            self.run_dir, folds, metadata={"source": "agent_dev"}
        )

        # Initial experiment from deterministic pipeline winner (metrics already computed)
        init_metrics = metrics_from_final_selection(self.initial_final_selection)
        init_metrics.update(
            {
                "estimator": self.initial_estimator,
                "feature_count": len(self.selected_features),
                "selected_features": list(self.selected_features),
                "agent_dev_size": len(agent_dev_df),
                "agent_val_size": len(agent_val_df),
                "development_split_size": len(dev_df),
                "cv_folds_hash": cv_folds_hash,
            }
        )
        # Score protected val with current estimator defaults if possible
        executor = AgenticExperimentExecutor(
            run_dir=self.run_dir,
            workflow_config=self.config,
            agent_dev_train_path=agent_dev_path,
            agent_val_path=agent_val_path,
            selected_features=self.selected_features,
            dataset_hash=self.dataset_hash,
            cv_folds_path=cv_folds_path,
            protected_val_indices_path=paths.get("agent_val_indices_path"),
            hpo_config=self.hpo_config,
            grid_proposer=self.grid_proposer,
            log_callback=self.log,
        )
        init_metrics.update(
            executor._score_agent_val(
                self.initial_estimator,
                self.selected_features,
                {"estimator": self.initial_estimator, "params": {}},
            )
        )

        exp0 = ExperimentRecord(
            experiment_id="exp_001",
            parent_experiment_id=None,
            cycle_index=0,
            hypothesis="Initial deterministic QSAR pipeline result",
            action="accept_model",
            config_snapshot={
                "estimator": self.initial_estimator,
                "selected_features": self.selected_features,
                "source": "deterministic_pipeline",
            },
            status="completed",
            artifact_directory=str(experiment_dir(self.run_dir, "exp_001")),
            internal_metrics=init_metrics,
            comparison_to_parent={},
            conclusion="Baseline deterministic development result.",
            created_at=datetime.now(timezone.utc).isoformat(),
            dataset_hash=self.dataset_hash,
            experiment_kind="initial_deterministic",
            multi_component=True,
            component_list=["full_deterministic_pipeline"],
            cv_folds_hash=cv_folds_hash,
            random_seed=self.config.random_seed,
            decision_source="deterministic_code",
            configuration_hash=configuration_hash(
                {"estimator": self.initial_estimator, "features": self.selected_features}
            ),
            estimator=self.initial_estimator,
            selected_features=list(self.selected_features),
            feature_count=len(self.selected_features),
        )
        append_experiment_record(self.run_dir, exp0)
        summary = build_agent_visible_summary(
            experiment_id=exp0.experiment_id,
            run_dir=self.run_dir,
            internal_metrics=init_metrics,
            estimator=self.initial_estimator,
            selected_features=self.selected_features,
            experiment_kind="initial_deterministic",
            cv_folds_hash=cv_folds_hash,
            warnings=[split_meta.get("limitation", "")],
        )

        state = AgenticProjectState(
            project_id=self.run_dir.name,
            initial_run_id=self.run_dir.name,
            current_experiment_id=exp0.experiment_id,
            best_experiment_id=exp0.experiment_id,
            cycle_index=0,
            experiment_count=1,
            status="developing",
            acceptance_criteria=self.agentic.acceptance,
            external_test_locked=True,
            cv_folds_hash=cv_folds_hash,
            agent_dev_indices_path=paths.get("agent_dev_indices_path"),
            agent_val_indices_path=paths.get("agent_val_indices_path"),
        )
        save_project_state(self.run_dir, state)

        acceptance = evaluate_acceptance(
            init_metrics,
            self.agentic.acceptance,
            overfit_thresholds=self.hpo_config.thresholds,
        )
        # Validation first pass
        val_review, _ = run_validation_agent(
            provider=self.provider,
            payload={"summary": summary.model_dump()},
            acceptance_passed=acceptance.accepted,
            deterministic_flags=flags_from_runtime(
                acceptance_criteria_failed=not acceptance.accepted
            ),
        )
        acceptance = evaluate_acceptance(
            init_metrics,
            self.agentic.acceptance,
            validation_approved=val_review.approved if self.agentic.acceptance.require_validation_agent_approval else True,
            overfit_thresholds=self.hpo_config.thresholds,
        )
        state = state.model_copy(
            update={"last_acceptance": acceptance, "last_validation_review": val_review}
        )
        save_project_state(self.run_dir, state)

        if acceptance.accepted and val_review.approved:
            best, rationale = select_best_experiment(
                [exp0],
                practical_equivalence_tolerance=self.agentic.practical_equivalence_tolerance,
                acceptance_lookup={exp0.experiment_id: True},
            )
            assert best is not None
            state = lock_model(state, best, selection_rationale=rationale)
            save_project_state(self.run_dir, state)
            write_final_agent_report(
                self.run_dir,
                initial_assessment=acceptance.explanation,
                acceptance=acceptance,
                validation=val_review,
                selection_rationale=rationale,
                stopping_reason="accepted_initial",
            )
            return state

        # Improvement cycles
        current = exp0
        stopping_reason = "max_cycles"
        for cycle in range(1, max_cycles + 1):
            assert_agentic_optimization_allowed(state)
            if self.stop_check():
                stopping_reason = "user_stop"
                break
            if state.experiment_count >= max_experiments:
                stopping_reason = "budget_exhausted"
                break

            state = state.model_copy(update={"cycle_index": cycle})
            save_project_state(self.run_dir, state)

            view = AgentArtifactView(
                experiment_id=current.experiment_id,
                summary=summary,
                approved_artifact_paths=summary.source_artifact_paths,
                ledger_digest=[
                    {
                        "experiment_id": r.experiment_id,
                        "action": r.action,
                        "mean_cv_r2": r.internal_metrics.get("mean_cv_r2"),
                        "estimator": r.estimator,
                        "experiment_kind": r.experiment_kind,
                    }
                    for r in load_experiment_records(self.run_dir)
                ],
                acceptance_criteria=self.agentic.acceptance.model_dump(),
                budget={
                    "cycle": cycle,
                    "max_cycles": max_cycles,
                    "experiment_count": state.experiment_count,
                    "max_experiments": max_experiments,
                },
            )
            payload = view.to_agent_payload()

            specialists = plan_specialists_for_cycle(
                current.internal_metrics,
                summary.model_dump(),
                max_specialists=min(self.agentic.max_specialist_calls_per_cycle, 2),
                force_validation=False,
            )
            # Always allow validation as soft reviewer separately if room
            proposals = []
            reports_dir = agent_workspace(self.run_dir) / "specialist_reports"
            reports_dir.mkdir(parents=True, exist_ok=True)

            for specialist in specialists:
                if specialist == "data_quality":
                    diag, props, src = run_data_quality_agent(
                        provider=self.provider,
                        experiment_id=current.experiment_id,
                        payload=payload,
                    )
                elif specialist == "descriptor_feature":
                    diag, props, src = run_descriptor_feature_agent(
                        provider=self.provider,
                        experiment_id=current.experiment_id,
                        payload=payload,
                    )
                elif specialist == "modeling":
                    diag, props, src = run_modeling_agent(
                        provider=self.provider,
                        experiment_id=current.experiment_id,
                        payload=payload,
                    )
                elif specialist == SPECIALIST_VALIDATION:
                    continue
                else:
                    continue
                save_json(reports_dir / f"cycle_{cycle}_{specialist}.json", diag.model_dump())
                append_event(
                    self.run_dir,
                    event_type="specialist_diagnosis",
                    cycle_index=cycle,
                    experiment_id=current.experiment_id,
                    agent=specialist,
                    validated_response=diag.model_dump(),
                    decision_source=src,  # type: ignore[arg-type]
                )
                proposals.extend(props)

            # Validation soft review each cycle
            val_review, vsrc = run_validation_agent(
                provider=self.provider,
                payload=payload,
                acceptance_passed=False,
                deterministic_flags=flags_from_runtime(acceptance_criteria_failed=True),
            )
            append_event(
                self.run_dir,
                event_type="validation_review",
                cycle_index=cycle,
                agent="validation",
                validated_response=val_review.model_dump(),
                decision_source=vsrc,  # type: ignore[arg-type]
            )

            budget_exhausted = state.experiment_count >= max_experiments
            decision, selected, dsrc = run_supervisor(
                provider=self.provider,
                cycle_index=cycle,
                payload=payload,
                proposals=proposals,
                ledger=load_experiment_records(self.run_dir),
                acceptance_passed=False,
                budget_exhausted=budget_exhausted,
                specialists_consulted=specialists,
            )
            append_supervisor_decision(self.run_dir, decision)
            append_event(
                self.run_dir,
                event_type="supervisor_decision",
                cycle_index=cycle,
                agent="supervisor",
                selected_action=decision.action,
                validated_response=decision.model_dump(),
                decision_source=dsrc,  # type: ignore[arg-type]
            )

            if decision.action == "accept_model":
                stopping_reason = "supervisor_accept"
                break
            if decision.action in ("stop_budget_exhausted", "stop_no_viable_model"):
                stopping_reason = decision.action
                break
            if decision.action in (
                "request_user_approval",
                "request_user_input",
                "request_model_dependency_approval",
            ):
                state = create_pending_approval(
                    self.run_dir,
                    state,
                    topic=decision.action,
                    proposed_change=(selected.configuration_changes if selected else {}),
                    scientific_rationale=decision.rationale,
                    executable_topic=False,
                )
                stopping_reason = "awaiting_approval"
                save_project_state(self.run_dir, state)
                write_final_agent_report(
                    self.run_dir,
                    initial_assessment=acceptance.explanation,
                    acceptance=acceptance,
                    validation=val_review,
                    selection_rationale="Paused for user input/approval.",
                    stopping_reason=stopping_reason,
                )
                return state

            if selected is None:
                stopping_reason = "no_viable_action"
                break

            try:
                assert_agentic_optimization_allowed(state)
                new_rec = executor.execute(selected, parent=current, cycle_index=cycle)
            except AgenticOptimizationForbiddenError:
                raise
            except Exception as exc:
                append_event(
                    self.run_dir,
                    event_type="experiment_error",
                    cycle_index=cycle,
                    error=str(exc),
                )
                self.log(f"Experiment failed: {exc}")
                stopping_reason = "experiment_error"
                break

            append_experiment_record(self.run_dir, new_rec)
            append_event(
                self.run_dir,
                event_type="experiment_completed",
                cycle_index=cycle,
                experiment_id=new_rec.experiment_id,
                selected_action=str(new_rec.action),
                tool_execution={
                    "experiment_kind": new_rec.experiment_kind,
                    "multi_component": new_rec.multi_component,
                },
            )
            current = new_rec
            summary = build_agent_visible_summary(
                experiment_id=new_rec.experiment_id,
                run_dir=self.run_dir,
                internal_metrics=new_rec.internal_metrics,
                estimator=new_rec.estimator,
                selected_features=new_rec.selected_features,
                experiment_kind=new_rec.experiment_kind,
                cv_folds_hash=cv_folds_hash,
            )
            state = state.model_copy(
                update={
                    "current_experiment_id": new_rec.experiment_id,
                    "experiment_count": state.experiment_count + 1,
                }
            )

            acceptance = evaluate_acceptance(
                new_rec.internal_metrics,
                self.agentic.acceptance,
                overfit_thresholds=self.hpo_config.thresholds,
            )
            val_review, _ = run_validation_agent(
                provider=self.provider,
                payload={"summary": summary.model_dump()},
                acceptance_passed=acceptance.accepted,
                deterministic_flags=flags_from_runtime(
                    acceptance_criteria_failed=not acceptance.accepted
                ),
            )
            acceptance = evaluate_acceptance(
                new_rec.internal_metrics,
                self.agentic.acceptance,
                validation_approved=(
                    val_review.approved
                    if self.agentic.acceptance.require_validation_agent_approval
                    else True
                ),
                overfit_thresholds=self.hpo_config.thresholds,
            )
            state = state.model_copy(
                update={
                    "last_acceptance": acceptance,
                    "last_validation_review": val_review,
                }
            )
            save_project_state(self.run_dir, state)
            if acceptance.accepted and val_review.approved:
                stopping_reason = "accepted"
                break

        records = load_experiment_records(self.run_dir)
        acceptance_lookup = {
            r.experiment_id: bool(
                evaluate_acceptance(
                    r.internal_metrics,
                    self.agentic.acceptance,
                    validation_approved=True,
                    overfit_thresholds=self.hpo_config.thresholds,
                ).accepted
            )
            for r in records
        }
        best, rationale = select_best_experiment(
            records,
            practical_equivalence_tolerance=self.agentic.practical_equivalence_tolerance,
            acceptance_lookup=acceptance_lookup,
        )
        if best is None:
            best = current
            rationale = "Fallback to last experiment."
        state = lock_model(state, best, selection_rationale=rationale)
        state = state.model_copy(update={"stopping_reason": stopping_reason})
        save_project_state(self.run_dir, state)
        write_final_agent_report(
            self.run_dir,
            initial_assessment="See experiment ledger for initial assessment.",
            acceptance=state.last_acceptance,
            validation=state.last_validation_review,
            selection_rationale=rationale,
            stopping_reason=stopping_reason,
        )
        self.log(f"Agentic loop stopped: {stopping_reason}; locked {best.experiment_id}")
        return state


def maybe_create_provider(agentic: AgenticImprovementConfig) -> AgentProvider | None:
    """Return OpenAI provider or None (deterministic fallback path)."""
    if not agentic.enabled:
        return None
    try:
        from qsar_agent.agentic.provider import OpenAIAgentProvider
        from qsar_agent.config import get_openai_api_key

        if not get_openai_api_key():
            return None
        return OpenAIAgentProvider(model=agentic.model or None)
    except ProviderError:
        return None
    except Exception:
        return None
