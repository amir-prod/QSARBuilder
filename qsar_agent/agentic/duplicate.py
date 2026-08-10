"""Duplicate and near-duplicate experiment prevention."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from qsar_agent.schemas.agentic import ExperimentProposal, ExperimentRecord, normalize_action


INCONSEQUENTIAL_KEYS = frozenset(
    {
        "log_callback",
        "openai_model",
        "n_jobs",
        "explanation",
        "notes",
    }
)


def canonicalize_config(action: str, configuration_changes: dict[str, Any]) -> dict[str, Any]:
    action_n = normalize_action(action)
    cleaned = {
        k: configuration_changes[k]
        for k in sorted(configuration_changes.keys())
        if k not in INCONSEQUENTIAL_KEYS
    }
    return {"action": action_n, "configuration_changes": cleaned}


def duplicate_check_key(action: str, configuration_changes: dict[str, Any]) -> str:
    canonical = canonicalize_config(action, configuration_changes)
    payload = json.dumps(canonical, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def find_duplicate(
    ledger: list[ExperimentRecord],
    action: str,
    configuration_changes: dict[str, Any],
) -> ExperimentRecord | None:
    key = duplicate_check_key(action, configuration_changes)
    for rec in ledger:
        rec_key = rec.config_snapshot.get("duplicate_check_key")
        if rec_key == key:
            return rec
        # Also compare canonical action+changes stored on record
        stored_changes = rec.config_snapshot.get("configuration_changes", {})
        if (
            normalize_action(str(rec.action)) == normalize_action(action)
            and duplicate_check_key(str(rec.action), stored_changes) == key
        ):
            return rec
    return None


def near_duplicate_warnings(
    ledger: list[ExperimentRecord],
    action: str,
    configuration_changes: dict[str, Any],
) -> list[str]:
    """Warn when only inconsequential params differ from a prior experiment."""
    warnings: list[str] = []
    action_n = normalize_action(action)
    core = {
        k: v
        for k, v in configuration_changes.items()
        if k not in INCONSEQUENTIAL_KEYS
    }
    for rec in ledger:
        if normalize_action(str(rec.action)) != action_n:
            continue
        prior = {
            k: v
            for k, v in rec.config_snapshot.get("configuration_changes", {}).items()
            if k not in INCONSEQUENTIAL_KEYS
        }
        if prior == core:
            continue
        # Same estimator / feature_count with tiny diffs
        same_estimator = prior.get("estimator") == core.get("estimator")
        same_features = prior.get("feature_count") == core.get("feature_count")
        if same_estimator and same_features and prior and core:
            warnings.append(
                f"Near-duplicate of {rec.experiment_id}: same action/estimator/feature_count "
                f"with minor configuration differences."
            )
    return warnings


def ensure_proposal_key(proposal: ExperimentProposal) -> ExperimentProposal:
    if proposal.duplicate_check_key:
        return proposal
    key = duplicate_check_key(proposal.action, proposal.configuration_changes)
    return proposal.model_copy(update={"duplicate_check_key": key})
