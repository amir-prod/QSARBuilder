"""Load a prior deterministic run and resume with agentic improvement only."""

from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from pydantic import BaseModel, Field

from qsar_agent.agentic.loop import AgenticImprovementLoop, maybe_create_provider
from qsar_agent.agents.qsar_agent import propose_hyperparameter_grid
from qsar_agent.config import ModelConfig, WorkflowConfig
from qsar_agent.schemas.agentic import AgenticImprovementConfig, AgenticProjectState
from qsar_agent.schemas.hyperparameter_optimization import FinalModelSelection, HPOConfig
from qsar_agent.services.artifact_manager import file_hash, generate_run_id, get_run_dir, save_json
from qsar_agent.services.model_lock_eval import (
    ensure_model_locked,
    evaluate_locked_winner_external,
    save_post_test_audit_criteria_snapshot,
)
from qsar_agent.services.post_test_audit import run_post_test_audit
from qsar_agent.services.workflow_runner import _hpo_config_from_workflow


# Files that indicate the source lineage already saw the external test.
_EXTERNAL_ARTIFACT_NAMES = (
    "predictions.csv",
    "model_metrics.json",
    "prediction_scatter.png",
    "williams_plot.png",
    "applicability_domain.csv",
)


class ExternalAccessInfo(BaseModel):
    external_previously_evaluated: bool
    reasons: list[str] = Field(default_factory=list)
    artifact_paths: list[str] = Field(default_factory=list)


class WinnerSnapshot(BaseModel):
    winning_estimator_label: str
    estimator: str
    selected_features: list[str]
    final_model_config: dict[str, Any]
    final_selection: FinalModelSelection
    selection_rationale: str = ""
    mean_cv_r2: float | None = None
    source: str = ""


class RunSummary(BaseModel):
    run_id: str
    run_dir: str
    resumable: bool
    external_previously_evaluated: bool
    estimator: str | None = None
    feature_count: int | None = None
    mean_cv_r2: float | None = None
    reasons: list[str] = Field(default_factory=list)
    missing: list[str] = Field(default_factory=list)


class AgenticResumeResult(BaseModel):
    source_run_id: str
    forked_run_id: str
    forked_run_dir: str
    external_previously_evaluated: bool
    agentic_state: AgenticProjectState
    evaluated_external: bool = False
    disclaimer: str | None = None
    artifact_paths: dict[str, str] = Field(default_factory=dict)


class InPlaceResumeForbiddenError(RuntimeError):
    """Raised when attempting agentic resume inside a tainted source run_dir."""


def detect_external_access(run_dir: Path) -> ExternalAccessInfo:
    run_dir = Path(run_dir)
    reasons: list[str] = []
    paths: list[str] = []

    for name in _EXTERNAL_ARTIFACT_NAMES:
        p = run_dir / name
        if p.exists():
            reasons.append(f"found {name}")
            paths.append(str(p))

    locked = run_dir / "locked_external"
    if locked.is_dir() and any(locked.iterdir()):
        reasons.append("found locked_external/")
        paths.append(str(locked))

    branch_ext = run_dir / "branch_external_artifacts.json"
    if branch_ext.exists():
        try:
            data = json.loads(branch_ext.read_text(encoding="utf-8"))
            if isinstance(data, list) and data:
                if any(
                    isinstance(row, dict)
                    and (
                        row.get("test_r2") is not None
                        or row.get("scatter_png_path")
                        or row.get("metrics_path")
                    )
                    for row in data
                ):
                    reasons.append("found branch_external_artifacts.json with test metrics")
                    paths.append(str(branch_ext))
        except (OSError, json.JSONDecodeError):
            pass

    state_path = run_dir / "agent_workspace" / "project_state.json"
    if state_path.exists():
        try:
            state = json.loads(state_path.read_text(encoding="utf-8"))
            if state.get("external_test_accessed"):
                reasons.append("project_state.external_test_accessed=true")
                paths.append(str(state_path))
            if state.get("status") in ("external_evaluated", "completed") and state.get(
                "lock_record"
            ):
                # completed after lock often implies external was run in new pipeline
                if (run_dir / "predictions.csv").exists():
                    reasons.append("project_state completed with predictions")
        except (OSError, json.JSONDecodeError):
            pass

    # de-dupe reasons
    uniq_reasons = list(dict.fromkeys(reasons))
    return ExternalAccessInfo(
        external_previously_evaluated=bool(uniq_reasons),
        reasons=uniq_reasons,
        artifact_paths=paths,
    )


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _normalize_estimator(label: str, final_model_config: dict[str, Any]) -> str:
    cfg_est = final_model_config.get("estimator")
    if isinstance(cfg_est, str) and cfg_est.strip():
        return cfg_est.strip()
    # Strip expansion suffixes like "SVR (sfs_fixed_ga_plus2)"
    base = label.split("(")[0].strip()
    return base or label


def load_winner_from_run(run_dir: Path) -> WinnerSnapshot:
    run_dir = Path(run_dir)
    comparison = run_dir / "model_comparison.json"
    if comparison.exists():
        data = _load_json(comparison)
        final_cfg = data.get("final_model_config") or {}
        selection = FinalModelSelection.model_validate(data["final_selection"])
        features = list(data.get("selected_features") or [])
        label = str(data.get("winning_estimator") or final_cfg.get("estimator") or "unknown")
        estimator = _normalize_estimator(label, final_cfg)
        mean_cv = None
        if selection.cv_summary is not None:
            mean_cv = float(selection.cv_summary.mean_cv_r2)
        return WinnerSnapshot(
            winning_estimator_label=label,
            estimator=estimator,
            selected_features=features,
            final_model_config=final_cfg,
            final_selection=selection,
            selection_rationale=str(data.get("selection_rationale") or ""),
            mean_cv_r2=mean_cv,
            source="model_comparison.json",
        )

    hpo_path = run_dir / "hpo_final_selection.json"
    if not hpo_path.exists():
        raise FileNotFoundError(
            f"No model_comparison.json or hpo_final_selection.json in {run_dir}"
        )
    selection_data = _load_json(hpo_path)
    # Some runs nest under final_selection
    if "final_selection" in selection_data and "cv_summary" not in selection_data:
        selection = FinalModelSelection.model_validate(selection_data["final_selection"])
        params = selection_data.get("params") or selection.params
    else:
        selection = FinalModelSelection.model_validate(selection_data)
        params = selection.params

    features = _load_selected_features(run_dir)
    manifest = {}
    manifest_path = run_dir / "run_manifest.json"
    if manifest_path.exists():
        manifest = _load_json(manifest_path)
    estimator = str(
        (manifest.get("hyperparameter_optimization") or {}).get("winning_estimator")
        or (manifest.get("workflow_config") or {}).get("model", {}).get("estimator")
        or "RandomForestRegressor"
    )
    final_cfg = manifest.get("final_model_config") or {
        "estimator": estimator,
        "params": params,
        "random_state": (manifest.get("workflow_config") or {}).get("random_seed", 42),
    }
    if "estimator" not in final_cfg:
        final_cfg["estimator"] = estimator
    estimator = _normalize_estimator(estimator, final_cfg)
    mean_cv = float(selection.cv_summary.mean_cv_r2) if selection.cv_summary else None
    return WinnerSnapshot(
        winning_estimator_label=estimator,
        estimator=estimator,
        selected_features=features,
        final_model_config=final_cfg,
        final_selection=selection,
        selection_rationale=selection.selection_rationale or "",
        mean_cv_r2=mean_cv,
        source="hpo_final_selection.json",
    )


def _load_selected_features(run_dir: Path) -> list[str]:
    ga_path = run_dir / "ga_selected_features.json"
    if ga_path.exists():
        data = _load_json(ga_path)
        features = data.get("selected_features") or data.get("features")
        if features:
            return list(features)
    manifest_path = run_dir / "run_manifest.json"
    if manifest_path.exists():
        manifest = _load_json(manifest_path)
        features = manifest.get("selected_features")
        if features:
            return list(features)
    raise FileNotFoundError(f"No selected features found in {run_dir}")


def is_resumable_run(run_dir: Path) -> tuple[bool, list[str]]:
    run_dir = Path(run_dir)
    missing: list[str] = []
    if not (run_dir / "preprocessed_train_descriptors.csv").exists():
        missing.append("preprocessed_train_descriptors.csv")
    if not (run_dir / "model_comparison.json").exists() and not (
        run_dir / "hpo_final_selection.json"
    ).exists():
        missing.append("model_comparison.json|hpo_final_selection.json")
    return (len(missing) == 0, missing)


def list_resumable_runs(output_dir: str | Path = "outputs") -> list[RunSummary]:
    root = Path(output_dir)
    if not root.is_dir():
        return []
    summaries: list[RunSummary] = []
    for child in sorted(root.iterdir(), key=lambda p: p.name, reverse=True):
        if not child.is_dir():
            continue
        # Skip forks' nested noise? include all
        ok, missing = is_resumable_run(child)
        ext = detect_external_access(child)
        estimator = None
        feature_count = None
        mean_cv = None
        if ok:
            try:
                winner = load_winner_from_run(child)
                estimator = winner.estimator
                feature_count = len(winner.selected_features)
                mean_cv = winner.mean_cv_r2
            except Exception as exc:
                ok = False
                missing.append(f"winner_load_error: {exc}")
        summaries.append(
            RunSummary(
                run_id=child.name,
                run_dir=str(child),
                resumable=ok,
                external_previously_evaluated=ext.external_previously_evaluated,
                estimator=estimator,
                feature_count=feature_count,
                mean_cv_r2=mean_cv,
                reasons=ext.reasons,
                missing=missing,
            )
        )
    return [s for s in summaries if s.resumable]


def assert_not_inplace_on_tainted(source_dir: Path, target_dir: Path) -> None:
    """Forbid running agentic inside a source run that already saw the external test."""
    source_dir = Path(source_dir).resolve()
    target_dir = Path(target_dir).resolve()
    ext = detect_external_access(source_dir)
    if ext.external_previously_evaluated and source_dir == target_dir:
        raise InPlaceResumeForbiddenError(
            f"Cannot resume agentic in-place on run '{source_dir.name}' because the "
            f"external test was already evaluated ({'; '.join(ext.reasons)}). "
            "Fork a new lineage instead."
        )


def fork_run_for_agentic(
    source_dir: Path,
    output_root: str | Path = "outputs",
) -> tuple[Path, dict[str, Any]]:
    """
    Create a forked run directory with development-safe artifacts only.

    Never copies external predictions/metrics/scatter/Williams into the fork root.
    """
    source_dir = Path(source_dir)
    ok, missing = is_resumable_run(source_dir)
    if not ok:
        raise FileNotFoundError(f"Source run is not resumable; missing: {missing}")

    ext = detect_external_access(source_dir)
    winner = load_winner_from_run(source_dir)
    suffix = generate_run_id()[:6]
    fork_id = f"{source_dir.name}_agentic_{suffix}"
    # sanitize: run ids allow alnum _ -
    fork_dir = get_run_dir(output_root, fork_id)

    allowlisted = [
        "preprocessed_train_descriptors.csv",
        "preprocessed_test_descriptors.csv",
        "run_manifest.json",
        "input_dataset.csv",
        "cleaned_dataset.csv",
        "descriptor_preprocessor.joblib",
        "retained_descriptors.json",
    ]
    for name in allowlisted:
        src = source_dir / name
        if src.exists():
            shutil.copy2(src, fork_dir / name)

    save_json(fork_dir / "agentic_resume_winner.json", winner.model_dump())
    meta = {
        "source_run_id": source_dir.name,
        "source_run_dir": str(source_dir.resolve()),
        "forked_run_id": fork_id,
        "forked_at": datetime.now(timezone.utc).isoformat(),
        "external_previously_evaluated": ext.external_previously_evaluated,
        "external_access_reasons": ext.reasons,
        "winner_source": winner.source,
        "estimator": winner.estimator,
        "selected_features": winner.selected_features,
        "policy": (
            "In-place agentic resume on externally evaluated runs is forbidden. "
            "This fork reuses training artifacts only; prior external metrics are not copied."
        ),
    }
    save_json(fork_dir / "agentic_resume_meta.json", meta)

    # Quarantine reference paths only (no agent-visible metrics copied)
    if ext.external_previously_evaluated:
        ref = fork_dir / "source_external_reference"
        ref.mkdir(exist_ok=True)
        (ref / "README.txt").write_text(
            "This directory records that the source run already evaluated the external "
            "test. External metric files are intentionally NOT copied here so agents "
            "cannot access them.\n"
            f"Source run: {source_dir.name}\n"
            f"Reasons: {', '.join(ext.reasons)}\n",
            encoding="utf-8",
        )
        save_json(ref / "source_external_access.json", ext.model_dump())

    return fork_dir, meta


def _workflow_config_from_source(
    source_dir: Path,
    agentic_override: AgenticImprovementConfig,
) -> WorkflowConfig:
    manifest_path = Path(source_dir) / "run_manifest.json"
    base = WorkflowConfig()
    if manifest_path.exists():
        manifest = _load_json(manifest_path)
        wc = manifest.get("workflow_config")
        if isinstance(wc, dict):
            try:
                base = WorkflowConfig.model_validate(wc)
            except Exception:
                base = WorkflowConfig(**{k: v for k, v in wc.items() if k in WorkflowConfig.model_fields})
    return base.model_copy(update={"agentic": agentic_override})


def run_agentic_only(
    source_run_dir: str | Path,
    *,
    workflow_config: WorkflowConfig | None = None,
    agentic_config: AgenticImprovementConfig | None = None,
    output_root: str | Path = "outputs",
    evaluate_external_after_lock: bool = False,
    provider: Any = None,
    log_callback: Callable[[str], None] | None = None,
    stop_check: Callable[[], bool] | None = None,
) -> AgenticResumeResult:
    """
    Fork a prior run and execute only the agentic improvement loop.

    Always forks. Raises ``InPlaceResumeForbiddenError`` if called in a way that
    would mutate a tainted source directory in place.
    """
    source_dir = Path(source_run_dir)
    log = log_callback or (lambda m: None)
    ext = detect_external_access(source_dir)

    agentic_cfg = agentic_config or (workflow_config.agentic if workflow_config else AgenticImprovementConfig(enabled=True))
    if not agentic_cfg.enabled:
        agentic_cfg = agentic_cfg.model_copy(update={"enabled": True})

    cfg = workflow_config or _workflow_config_from_source(source_dir, agentic_cfg)
    cfg = cfg.model_copy(update={"agentic": agentic_cfg, "output_dir": str(output_root)})

    fork_dir, meta = fork_run_for_agentic(source_dir, output_root=output_root)
    assert_not_inplace_on_tainted(source_dir, fork_dir)
    # Extra safety: never write agentic into source
    if fork_dir.resolve() == source_dir.resolve():
        raise InPlaceResumeForbiddenError("Fork resolved to source directory; aborting.")

    winner = load_winner_from_run(source_dir)
    # Prefer snapshot written into fork
    snap_path = fork_dir / "agentic_resume_winner.json"
    if snap_path.exists():
        snap = _load_json(snap_path)
        winner = WinnerSnapshot.model_validate(snap)

    train_path = fork_dir / "preprocessed_train_descriptors.csv"
    test_path = fork_dir / "preprocessed_test_descriptors.csv"
    dataset_hash = ""
    input_csv = fork_dir / "input_dataset.csv"
    if input_csv.exists():
        dataset_hash = file_hash(input_csv)
    elif (source_dir / "input_dataset.csv").exists():
        dataset_hash = file_hash(source_dir / "input_dataset.csv")

    hpo_cfg = _hpo_config_from_workflow(cfg)
    if provider is None:
        provider = maybe_create_provider(cfg.agentic)
        if provider is None:
            log("Agentic resume: no OpenAI API key; using labeled deterministic_fallback agents.")

    def grid_proposer(**kwargs):
        return propose_hyperparameter_grid(
            openai_model=hpo_cfg.openai_model or None,
            **kwargs,
        )

    log(
        f"Forked agentic lineage {fork_dir.name} from {source_dir.name} "
        f"(external_previously_evaluated={ext.external_previously_evaluated})."
    )
    loop = AgenticImprovementLoop(
        run_dir=fork_dir,
        workflow_config=cfg,
        hpo_config=hpo_cfg,
        development_train_path=train_path,
        selected_features=list(winner.selected_features),
        dataset_hash=dataset_hash,
        initial_estimator=winner.estimator,
        initial_final_selection=winner.final_selection,
        provider=provider,
        grid_proposer=grid_proposer if hpo_cfg.enabled else None,
        log_callback=log,
        stop_check=stop_check,
    )
    agentic_state = loop.run()

    model_cfg = ModelConfig(**{**ModelConfig().model_dump(), **winner.final_model_config})
    # Prefer locked experiment estimator/features if available
    from qsar_agent.agentic.ledger import get_experiment

    locked_exp = None
    if agentic_state.locked_experiment_id:
        locked_exp = get_experiment(fork_dir, agentic_state.locked_experiment_id)
    estimator = (locked_exp.estimator if locked_exp and locked_exp.estimator else winner.estimator)
    features = list(
        locked_exp.selected_features
        if locked_exp and locked_exp.selected_features
        else winner.selected_features
    )
    if locked_exp and locked_exp.config_snapshot.get("configuration_changes", {}).get("estimator"):
        estimator = locked_exp.config_snapshot["configuration_changes"]["estimator"]
    if locked_exp and locked_exp.config_snapshot.get("final_model_config"):
        model_cfg = ModelConfig(
            **{**ModelConfig().model_dump(), **locked_exp.config_snapshot["final_model_config"]}
        )
    else:
        model_cfg = model_cfg.model_copy(update={"estimator": estimator})

    agentic_state = ensure_model_locked(
        fork_dir,
        workflow_config=cfg,
        dataset_hash=dataset_hash,
        estimator=estimator,
        selected_features=features,
        final_model_config=model_cfg,
        selection_rationale=winner.selection_rationale
        or "Locked after agentic resume using training-only evidence.",
        selection_record={
            "source_run_id": source_dir.name,
            "winning_estimator": estimator,
            "selected_features": features,
        },
        agentic_state=agentic_state,
    )

    evaluated = False
    disclaimer = None
    artifact_paths: dict[str, str] = {
        "agent_workspace": str(fork_dir / "agent_workspace"),
        "agentic_resume_meta": str(fork_dir / "agentic_resume_meta.json"),
    }

    if evaluate_external_after_lock:
        if not test_path.exists():
            log("External evaluation requested but preprocessed_test_descriptors.csv is missing.")
        else:
            if ext.external_previously_evaluated:
                disclaimer = (
                    f"EXTERNAL-TEST DISCLAIMER: holdout from source run `{source_dir.name}` "
                    "was previously scored and is not an untouched independent external test "
                    "for this forked lineage."
                )
            # Freeze audit criteria before external unlock.
            save_post_test_audit_criteria_snapshot(fork_dir, cfg.agentic.post_test_audit)
            agentic_state, modeling, ad = evaluate_locked_winner_external(
                fork_dir,
                agentic_state=agentic_state,
                train_path=train_path,
                test_path=test_path,
                selected_features=features,
                model_config=model_cfg,
                dataset_hash=dataset_hash,
                config_snapshot=cfg.to_dict(),
                hpo_metadata={"winning_estimator": estimator, "resumed_from": source_dir.name},
                external_previously_evaluated=ext.external_previously_evaluated,
                source_run_id=source_dir.name,
                log_callback=log,
                use_lock_record_config=True,
            )
            evaluated = True
            audit = run_post_test_audit(fork_dir)
            log(f"Post-test audit: {audit.primary_outcome}")
            artifact_paths.update(
                {
                    "predictions": modeling.predictions_path,
                    "final_model": modeling.model_path,
                    "prediction_scatter": modeling.scatter_png_path,
                    "williams_plot": ad.williams_png_path,
                    "locked_external": str(fork_dir / "locked_external"),
                    "post_test_audit": str(fork_dir / "locked_external" / "post_test_audit.json"),
                }
            )

    return AgenticResumeResult(
        source_run_id=source_dir.name,
        forked_run_id=fork_dir.name,
        forked_run_dir=str(fork_dir),
        external_previously_evaluated=ext.external_previously_evaluated,
        agentic_state=agentic_state,
        evaluated_external=evaluated,
        disclaimer=disclaimer,
        artifact_paths=artifact_paths,
    )
