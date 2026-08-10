"""Code-enforced model lock and external-evaluation gate."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from typing import Any

from qsar_agent.schemas.agentic import AgenticProjectState, ExperimentRecord, ModelLockRecord


class ExternalEvalLockError(RuntimeError):
    """Raised when external evaluation is attempted without a valid lock."""


class AgenticOptimizationForbiddenError(RuntimeError):
    """Raised when agentic optimization is attempted after external-test access."""


def configuration_hash(config_snapshot: dict[str, Any]) -> str:
    payload = json.dumps(config_snapshot, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def lock_model(
    state: AgenticProjectState,
    experiment: ExperimentRecord,
    *,
    selection_rationale: str,
    selection_record: dict[str, Any] | None = None,
) -> AgenticProjectState:
    """Lock the winning experiment before any external evaluation."""
    if state.external_test_accessed:
        raise AgenticOptimizationForbiddenError(
            "Cannot lock a new model after external-test access in this lineage."
        )
    cfg_hash = experiment.configuration_hash or configuration_hash(experiment.config_snapshot)
    record = ModelLockRecord(
        locked_experiment_id=experiment.experiment_id,
        configuration_hash=cfg_hash,
        locked_at=datetime.now(timezone.utc).isoformat(),
        selection_rationale=selection_rationale,
        selection_record=selection_record or {},
        dataset_hash=experiment.dataset_hash,
        cv_folds_hash=experiment.cv_folds_hash or state.cv_folds_hash or "",
    )
    return state.model_copy(
        update={
            "status": "model_locked",
            "locked_experiment_id": experiment.experiment_id,
            "best_experiment_id": experiment.experiment_id,
            "current_experiment_id": experiment.experiment_id,
            "lock_record": record,
            "external_test_locked": True,
        }
    )


def assert_external_eval_allowed(state: AgenticProjectState) -> ModelLockRecord:
    """Require model_locked status and complete lock prerequisites."""
    if state.status != "model_locked":
        raise ExternalEvalLockError(
            f"External evaluation requires project_state.status == 'model_locked' "
            f"(got '{state.status}')."
        )
    if state.lock_record is None:
        raise ExternalEvalLockError("External evaluation requires a lock_record.")
    lock = state.lock_record
    if not lock.locked_experiment_id:
        raise ExternalEvalLockError("missing locked_experiment_id")
    if not lock.configuration_hash:
        raise ExternalEvalLockError("missing configuration_hash")
    if not lock.locked_at:
        raise ExternalEvalLockError("missing lock timestamp")
    if not lock.selection_rationale and not lock.selection_record:
        raise ExternalEvalLockError("missing selection record/rationale")
    if state.locked_experiment_id != lock.locked_experiment_id:
        raise ExternalEvalLockError("locked_experiment_id mismatch between state and lock_record")
    return lock


def mark_external_evaluated(state: AgenticProjectState) -> AgenticProjectState:
    """After external eval: mark lineage completed and forbid further agentic cycles."""
    return state.model_copy(
        update={
            "status": "external_evaluated",
            "external_test_accessed": True,
            "external_test_locked": True,
        }
    )


def assert_agentic_optimization_allowed(state: AgenticProjectState) -> None:
    """Prohibit further agentic optimization after external-test access or completion."""
    if state.external_test_accessed or state.status in (
        "external_evaluated",
        "completed",
    ):
        raise AgenticOptimizationForbiddenError(
            "Agentic optimization is prohibited after external-test access in this lineage."
        )
    if state.status == "model_locked":
        raise AgenticOptimizationForbiddenError(
            "Agentic optimization is prohibited after model lock; unlock is not supported. "
            "External evaluation should proceed next."
        )
