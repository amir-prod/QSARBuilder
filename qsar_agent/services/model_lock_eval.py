"""Shared model-lock and post-lock external evaluation helpers."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, Callable

from qsar_agent.agentic.ledger import (
    get_experiment,
    load_experiment_records,
    load_project_state,
    save_project_state,
)
from qsar_agent.agentic.lock import (
    ExternalEvalLockError,
    assert_external_eval_allowed,
    configuration_hash,
    lock_model,
    mark_external_evaluated,
)
from qsar_agent.agentic.selection import select_best_experiment
from qsar_agent.config import ModelConfig, WorkflowConfig
from qsar_agent.models.registry import SUPPORTED_ESTIMATORS, normalize_estimator_name
from qsar_agent.schemas.agentic import AgenticProjectState, ExperimentRecord, ModelLockRecord
from qsar_agent.services.artifact_manager import save_json
from qsar_agent.tools.applicability_domain import calculate_applicability_domain
from qsar_agent.tools.final_model import train_and_evaluate_final_model


class ConfigurationHashMismatchError(RuntimeError):
    """Raised when eval config hash does not match the lock record."""


def build_lock_config_snapshot(
    *,
    estimator: str,
    selected_features: list[str],
    final_model_config: ModelConfig | dict[str, Any],
) -> dict[str, Any]:
    cfg = (
        final_model_config.model_dump()
        if isinstance(final_model_config, ModelConfig)
        else dict(final_model_config)
    )
    return {
        "estimator": estimator,
        "selected_features": list(selected_features),
        "final_model_config": cfg,
    }


def _pick_locked_estimator(
    exp: Any,
    selection: dict[str, Any],
    snap: dict[str, Any],
    changes: dict[str, Any],
    final_cfg: dict[str, Any],
) -> str:
    """Prefer a registry estimator name over expansion display labels."""
    candidates = [
        final_cfg.get("estimator"),
        selection.get("final_model_config", {}).get("estimator")
        if isinstance(selection.get("final_model_config"), dict)
        else None,
        exp.estimator if exp is not None else None,
        selection.get("winning_estimator"),
        changes.get("estimator"),
        snap.get("estimator"),
    ]
    first_nonempty = ""
    for raw in candidates:
        if not raw:
            continue
        name = normalize_estimator_name(str(raw))
        if not first_nonempty:
            first_nonempty = name
        if name in SUPPORTED_ESTIMATORS:
            return name
    return first_nonempty


def resolve_locked_eval_config(
    run_dir: Path,
    agentic_state: AgenticProjectState,
) -> tuple[str, list[str], ModelConfig, dict[str, Any], str]:
    """
    Resolve estimator/features/model config strictly from the lock record / locked experiment.

    Returns (estimator, features, model_config, eval_config_snapshot, eval_configuration_hash).
    """
    if agentic_state.lock_record is None:
        raise ExternalEvalLockError("Cannot resolve locked eval config without lock_record.")
    lock: ModelLockRecord = agentic_state.lock_record
    exp = None
    if agentic_state.locked_experiment_id:
        exp = get_experiment(run_dir, agentic_state.locked_experiment_id)

    selection = dict(lock.selection_record or {})
    snap = dict(exp.config_snapshot) if exp is not None else {}
    changes = snap.get("configuration_changes") or {}
    if not isinstance(changes, dict):
        changes = {}

    final_cfg = (
        snap.get("final_model_config")
        or selection.get("final_model_config")
        or changes.get("final_model_config")
        or {}
    )
    if not isinstance(final_cfg, dict):
        final_cfg = {}

    estimator = _pick_locked_estimator(exp, selection, snap, changes, final_cfg)
    if not estimator:
        raise ExternalEvalLockError("Locked experiment has no estimator.")

    features = list(
        (exp.selected_features if exp and exp.selected_features else None)
        or selection.get("selected_features")
        or snap.get("selected_features")
        or changes.get("selected_features")
        or []
    )
    if not features:
        raise ExternalEvalLockError("Locked experiment has no selected_features.")

    model_cfg_data = {**ModelConfig().model_dump(), **final_cfg}
    model_cfg_data["estimator"] = estimator
    params = changes.get("params") or final_cfg.get("params") or selection.get("params")
    if isinstance(params, dict) and params:
        model_cfg_data.setdefault("params", {})
        if isinstance(model_cfg_data["params"], dict):
            model_cfg_data["params"] = {**model_cfg_data["params"], **params}
        else:
            model_cfg_data["params"] = dict(params)
    # Common top-level HPO fields may live in configuration_changes
    for key in (
        "n_estimators",
        "max_depth",
        "min_samples_split",
        "min_samples_leaf",
        "max_features",
        "bootstrap",
        "max_samples",
        "criterion",
        "random_state",
        "n_jobs",
        "C",
        "epsilon",
        "kernel",
        "gamma",
        "alpha",
        "l1_ratio",
    ):
        if key in changes and key not in final_cfg:
            model_cfg_data[key] = changes[key]

    model_config = ModelConfig(**{
        k: v for k, v in model_cfg_data.items() if k in ModelConfig.model_fields
    })
    # Preserve unknown params in params dict
    extra = {k: v for k, v in model_cfg_data.items() if k not in ModelConfig.model_fields}
    if extra:
        model_config = model_config.model_copy(
            update={"params": {**(model_config.params or {}), **extra}}
        )

    eval_cfg = build_lock_config_snapshot(
        estimator=estimator,
        selected_features=features,
        final_model_config=model_config,
    )
    return estimator, features, model_config, eval_cfg, configuration_hash(eval_cfg)


def verify_lock_configuration_hash(
    run_dir: Path,
    agentic_state: AgenticProjectState,
    *,
    expected_configuration_hash: str | None = None,
) -> str:
    """
    Verify the lock record hash against the locked experiment (or selection snapshot).

    Returns the lock_record.configuration_hash when verification succeeds.
    """
    if agentic_state.lock_record is None:
        raise ExternalEvalLockError("missing lock_record")
    lock = agentic_state.lock_record
    expected = expected_configuration_hash or lock.configuration_hash
    if not expected:
        raise ConfigurationHashMismatchError("lock_record.configuration_hash is empty.")
    if expected != lock.configuration_hash:
        raise ConfigurationHashMismatchError(
            f"expected_configuration_hash ({expected[:16]}…) does not match "
            f"lock_record.configuration_hash ({lock.configuration_hash[:16]}…)."
        )

    exp = (
        get_experiment(run_dir, agentic_state.locked_experiment_id)
        if agentic_state.locked_experiment_id
        else None
    )
    if exp is not None and exp.config_snapshot:
        recomputed = configuration_hash(exp.config_snapshot)
        stored = exp.configuration_hash or recomputed
        if expected not in (recomputed, stored):
            raise ConfigurationHashMismatchError(
                f"Configuration hash mismatch before external evaluation: "
                f"lock_record={expected[:16]}… experiment={recomputed[:16]}…. "
                "Refusing to evaluate a config that does not match the lock record."
            )
        return expected

    # Synthetic / selection-only lock (e.g. deterministic_winner with no ledger file)
    selection = dict(lock.selection_record or {})
    if selection.get("final_model_config") and selection.get("selected_features"):
        features = list(selection.get("selected_features") or [])
        fc = selection["final_model_config"]
        raw_est = str(selection.get("winning_estimator") or selection.get("estimator") or "")
        acceptable = set()
        for est in (raw_est, normalize_estimator_name(raw_est)):
            if not est:
                continue
            acceptable.add(
                configuration_hash(
                    build_lock_config_snapshot(
                        estimator=est,
                        selected_features=features,
                        final_model_config=fc,
                    )
                )
            )
        if expected not in acceptable:
            raise ConfigurationHashMismatchError(
                f"Configuration hash mismatch before external evaluation: "
                f"lock_record={expected[:16]}… selection={next(iter(acceptable), '')[:16]}…."
            )
        return expected

    # Last resort: trust lock hash only if experiment id is present but file missing
    # (should not happen for normal agentic locks)
    if not agentic_state.locked_experiment_id:
        raise ConfigurationHashMismatchError(
            "Cannot verify configuration hash: no locked experiment and incomplete selection_record."
        )
    return expected


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
    estimator = normalize_estimator_name(estimator) or estimator
    if isinstance(final_model_config, ModelConfig):
        cfg_est = normalize_estimator_name(final_model_config.estimator)
        if cfg_est and cfg_est != final_model_config.estimator:
            final_model_config = final_model_config.model_copy(update={"estimator": cfg_est})
        estimator = cfg_est or estimator
    elif isinstance(final_model_config, dict):
        final_model_config = dict(final_model_config)
        cfg_est = normalize_estimator_name(
            str(final_model_config.get("estimator") or estimator)
        )
        if cfg_est:
            final_model_config["estimator"] = cfg_est
            estimator = cfg_est
    state = agentic_state or load_project_state(run_dir)
    if state is not None and state.status == "model_locked" and state.lock_record is not None:
        # Enrich selection_record with eval config if missing (helps hash verify + resolve)
        lock = state.lock_record
        sel = dict(lock.selection_record or {})
        if "final_model_config" not in sel:
            sel["final_model_config"] = (
                final_model_config.model_dump()
                if isinstance(final_model_config, ModelConfig)
                else dict(final_model_config)
            )
            sel.setdefault("winning_estimator", estimator)
            sel.setdefault("selected_features", list(selected_features))
            state = state.model_copy(
                update={
                    "lock_record": lock.model_copy(update={"selection_record": sel})
                }
            )
            save_project_state(run_dir, state)
        return state

    if state is not None and state.status != "model_locked":
        records = load_experiment_records(run_dir)
        if records:
            best_rec, rationale = select_best_experiment(
                records,
                practical_equivalence_tolerance=workflow_config.agentic.practical_equivalence_tolerance,
            )
            if best_rec is not None:
                sel = dict(selection_record or {})
                sel.setdefault("winning_estimator", best_rec.estimator or estimator)
                sel.setdefault(
                    "selected_features",
                    list(best_rec.selected_features or selected_features),
                )
                sel.setdefault(
                    "final_model_config",
                    final_model_config.model_dump()
                    if isinstance(final_model_config, ModelConfig)
                    else dict(final_model_config),
                )
                state = lock_model(
                    state.model_copy(update={"status": "developing"}),
                    best_rec,
                    selection_rationale=rationale,
                    selection_record=sel,
                )
                save_project_state(run_dir, state)
                return state

    lock_cfg = build_lock_config_snapshot(
        estimator=estimator,
        selected_features=selected_features,
        final_model_config=final_model_config,
    )
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
    sel_rec = dict(selection_record or {})
    sel_rec["winning_estimator"] = estimator
    sel_rec.setdefault("selected_features", selected_features)
    sel_rec.setdefault(
        "final_model_config",
        final_model_config.model_dump()
        if isinstance(final_model_config, ModelConfig)
        else dict(final_model_config),
    )
    state = lock_model(
        state,
        lock_rec,
        selection_rationale=selection_rationale,
        selection_record=sel_rec,
    )
    save_project_state(run_dir, state)
    return state


def save_post_test_audit_criteria_snapshot(
    run_dir: Path,
    criteria: Any,
) -> Path:
    """Persist frozen criteria before external unlock."""
    from qsar_agent.agentic.ledger import agent_workspace

    path = agent_workspace(run_dir) / "post_test_audit_criteria.json"
    payload = criteria.model_dump() if hasattr(criteria, "model_dump") else dict(criteria)
    save_json(path, payload)
    return path


def evaluate_locked_winner_external(
    run_dir: Path,
    *,
    agentic_state: AgenticProjectState,
    train_path: Path,
    test_path: Path,
    selected_features: list[str] | None = None,
    model_config: ModelConfig | None = None,
    dataset_hash: str = "",
    config_snapshot: dict[str, Any] | None = None,
    hpo_metadata: dict[str, Any] | None = None,
    activity_label: str = "activity",
    external_previously_evaluated: bool = False,
    source_run_id: str | None = None,
    expected_configuration_hash: str | None = None,
    log_callback: Callable[[str], None] | None = None,
    use_lock_record_config: bool = True,
) -> tuple[AgenticProjectState, Any, Any]:
    """
    Evaluate the locked winner on the external test once.

    When ``use_lock_record_config`` is True (default), estimator/features/params are
    taken from the lock record / locked experiment and verified against the lock hash.
    """
    log = log_callback or (lambda _m: None)
    assert_external_eval_allowed(agentic_state)
    if agentic_state.lock_record is None:
        raise ExternalEvalLockError("External evaluation requires a lock_record.")

    lock_hash = verify_lock_configuration_hash(
        run_dir,
        agentic_state,
        expected_configuration_hash=expected_configuration_hash,
    )

    if use_lock_record_config:
        estimator, features, model_config, eval_cfg, eval_hash = resolve_locked_eval_config(
            run_dir, agentic_state
        )
        selected_features = features
    else:
        if model_config is None or selected_features is None:
            raise ValueError("model_config and selected_features required when not using lock record")
        estimator = model_config.estimator
        eval_cfg = build_lock_config_snapshot(
            estimator=estimator,
            selected_features=selected_features,
            final_model_config=model_config,
        )
        eval_hash = configuration_hash(eval_cfg)
        if expected_configuration_hash and eval_hash != expected_configuration_hash:
            raise ConfigurationHashMismatchError(
                "Configuration hash mismatch before external evaluation."
            )

    assert selected_features is not None and model_config is not None

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

    save_json(
        locked_ext / "evaluated_config.json",
        {
            "configuration_hash": eval_hash,
            "lock_record_configuration_hash": lock_hash,
            "estimator": estimator,
            "selected_features": selected_features,
            "final_model_config": model_config.model_dump(),
            "eval_config_snapshot": eval_cfg,
        },
    )

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
    criteria_src = Path(run_dir) / "agent_workspace" / "post_test_audit_criteria.json"
    if criteria_src.exists():
        shutil.copy2(criteria_src, locked_ext / "post_test_audit_criteria.json")

    state = mark_external_evaluated(agentic_state)
    state = state.model_copy(update={"status": "completed"})
    save_project_state(run_dir, state)
    return state, modeling, ad
