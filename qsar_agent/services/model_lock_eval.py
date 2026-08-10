"""Shared model-lock and post-lock external evaluation helpers."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, Callable

from qsar_agent.agentic.ledger import (
    load_experiment_records,
    load_project_state,
    save_project_state,
)
from qsar_agent.agentic.lock import (
    assert_external_eval_allowed,
    configuration_hash,
    lock_model,
    mark_external_evaluated,
)
from qsar_agent.agentic.selection import select_best_experiment
from qsar_agent.config import ModelConfig, WorkflowConfig
from qsar_agent.schemas.agentic import AgenticProjectState, ExperimentRecord
from qsar_agent.services.artifact_manager import save_json
from qsar_agent.tools.applicability_domain import calculate_applicability_domain
from qsar_agent.tools.final_model import train_and_evaluate_final_model


def ensure_model_locked(
    run_dir: Path,
    *,
    workflow_config: WorkflowConfig,
    dataset_hash: str,
    estimator: str,
    selected_features: list[str],
    final_model_config: ModelConfig | dict[str, Any],
    selection_rationale: str,
    selection_record: dict[str, Any] | None = None,
    agentic_state: AgenticProjectState | None = None,
) -> AgenticProjectState:
    """Ensure project_state is model_locked before external evaluation."""
    state = agentic_state or load_project_state(run_dir)
    if state is not None and state.status == "model_locked" and state.lock_record is not None:
        return state

    if state is not None and state.status != "model_locked":
        records = load_experiment_records(run_dir)
        if records:
            best_rec, rationale = select_best_experiment(
                records,
                practical_equivalence_tolerance=workflow_config.agentic.practical_equivalence_tolerance,
            )
            if best_rec is not None:
                state = lock_model(
                    state.model_copy(update={"status": "developing"}),
                    best_rec,
                    selection_rationale=rationale,
                )
                save_project_state(run_dir, state)
                return state

    cfg = (
        final_model_config.model_dump()
        if isinstance(final_model_config, ModelConfig)
        else dict(final_model_config)
    )
    lock_cfg = {
        "estimator": estimator,
        "selected_features": selected_features,
        "final_model_config": cfg,
    }
    lock_rec = ExperimentRecord(
        experiment_id="deterministic_winner",
        hypothesis="CV-selected winner locked for external evaluation",
        action="accept_model",
        config_snapshot=lock_cfg,
        status="completed",
        internal_metrics={},
        dataset_hash=dataset_hash,
        configuration_hash=configuration_hash(lock_cfg),
        estimator=estimator,
        selected_features=list(selected_features),
        feature_count=len(selected_features),
    )
    state = AgenticProjectState(
        project_id=run_dir.name,
        initial_run_id=run_dir.name,
        current_experiment_id=lock_rec.experiment_id,
        best_experiment_id=lock_rec.experiment_id,
        status="developing",
        acceptance_criteria=workflow_config.agentic.acceptance,
    )
    state = lock_model(
        state,
        lock_rec,
        selection_rationale=selection_rationale,
        selection_record=selection_record
        or {
            "winning_estimator": estimator,
            "selected_features": selected_features,
        },
    )
    save_project_state(run_dir, state)
    return state


def evaluate_locked_winner_external(
    run_dir: Path,
    *,
    agentic_state: AgenticProjectState,
    train_path: Path,
    test_path: Path,
    selected_features: list[str],
    model_config: ModelConfig,
    dataset_hash: str = "",
    config_snapshot: dict[str, Any] | None = None,
    hpo_metadata: dict[str, Any] | None = None,
    activity_label: str = "activity",
    external_previously_evaluated: bool = False,
    source_run_id: str | None = None,
    log_callback: Callable[[str], None] | None = None,
) -> tuple[AgenticProjectState, Any, Any]:
    """
    Evaluate the locked winner on the external test once.

    Requires ``agentic_state.status == "model_locked"``. Writes artifacts to
    ``run_dir`` and ``locked_external/``, then marks the lineage completed.
    """
    log = log_callback or (lambda _m: None)
    assert_external_eval_allowed(agentic_state)

    modeling = train_and_evaluate_final_model(
        train_path=train_path,
        test_path=test_path,
        run_dir=run_dir,
        selected_features=selected_features,
        model_config=model_config,
        activity_label=activity_label,
        dataset_hash=dataset_hash,
        config_snapshot=config_snapshot,
        hpo_metadata=hpo_metadata,
    )
    ad = calculate_applicability_domain(
        train_path=train_path,
        test_path=test_path,
        predictions_path=modeling.predictions_path,
        run_dir=run_dir,
        selected_features=selected_features,
    )

    locked_ext = Path(run_dir) / "locked_external"
    locked_ext.mkdir(parents=True, exist_ok=True)
    for name in (
        "predictions.csv",
        "model_metrics.json",
        "prediction_scatter.png",
        "williams_plot.png",
        "applicability_domain.csv",
        "final_model.joblib",
    ):
        src = Path(run_dir) / name
        if src.exists():
            shutil.copy2(src, locked_ext / name)

    disclaimer = None
    if external_previously_evaluated:
        disclaimer = (
            "EXTERNAL-TEST DISCLAIMER: The external holdout used for this evaluation "
            f"was previously scored in source run `{source_run_id or 'unknown'}`. "
            "It is NOT an untouched independent external test for this forked lineage. "
            "If external-test results influence further model development, that test set "
            "is no longer independent and must not be reported as an untouched external test."
        )
        save_json(
            locked_ext / "external_independence_disclaimer.json",
            {
                "external_previously_evaluated": True,
                "source_run_id": source_run_id,
                "disclaimer": disclaimer,
            },
        )
        report_path = Path(run_dir) / "agent_workspace" / "final_agent_report.md"
        if report_path.exists():
            report_path.write_text(
                report_path.read_text(encoding="utf-8")
                + "\n\n## External-test independence (forked lineage)\n\n"
                + disclaimer
                + "\n",
                encoding="utf-8",
            )
        else:
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_path.write_text(
                "# Agentic Resume Report\n\n## External-test independence (forked lineage)\n\n"
                + disclaimer
                + "\n",
                encoding="utf-8",
            )
        log(disclaimer)

    save_json(
        locked_ext / "lock_record.json",
        agentic_state.lock_record.model_dump() if agentic_state.lock_record else {},
    )
    state = mark_external_evaluated(agentic_state)
    state = state.model_copy(update={"status": "completed"})
    save_project_state(run_dir, state)
    return state, modeling, ad
