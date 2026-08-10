"""Pending approval persistence for Streamlit pause/resume."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from qsar_agent.agentic.actions import EXECUTABLE_APPROVAL_TOPICS
from qsar_agent.agentic.ledger import agent_workspace, load_project_state, save_project_state
from qsar_agent.schemas.agentic import AgenticProjectState


def create_pending_approval(
    run_dir: Path,
    state: AgenticProjectState,
    *,
    topic: str,
    proposed_change: dict[str, Any],
    scientific_rationale: str,
    evidence: list[dict[str, Any]] | None = None,
    expected_benefit: str = "",
    risks: str = "",
    lineage_comparable: bool = True,
    executable_topic: bool = False,
) -> AgenticProjectState:
    """Persist a pending approval. Refuse non-executable data-mutation topics in v1."""
    if not executable_topic and topic not in EXECUTABLE_APPROVAL_TOPICS:
        # Still record informational pause for user input, but mark non-executable
        executable_topic = False

    pending = {
        "topic": topic,
        "proposed_change": proposed_change,
        "scientific_rationale": scientific_rationale,
        "evidence": evidence or [],
        "expected_benefit": expected_benefit,
        "risks": risks,
        "lineage_comparable": lineage_comparable,
        "executable_topic": executable_topic and topic in EXECUTABLE_APPROVAL_TOPICS,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "pending",
    }
    path = agent_workspace(run_dir) / "pending_approval.json"
    path.write_text(json.dumps(pending, indent=2, default=str), encoding="utf-8")
    updated = state.model_copy(
        update={"pending_approval": pending, "status": "awaiting_approval"}
    )
    save_project_state(run_dir, updated)
    return updated


def resolve_pending_approval(
    run_dir: Path,
    *,
    approve: bool,
    user_note: str = "",
) -> AgenticProjectState:
    state = load_project_state(run_dir)
    if state is None:
        raise RuntimeError("No agentic project state found")
    pending = state.pending_approval or {}
    pending = {
        **pending,
        "status": "approved" if approve else "rejected",
        "resolved_at": datetime.now(timezone.utc).isoformat(),
        "user_note": user_note,
    }
    path = agent_workspace(run_dir) / "pending_approval.json"
    path.write_text(json.dumps(pending, indent=2, default=str), encoding="utf-8")
    # Resume developing unless rejected stop
    new_status = "developing" if approve else "developing"
    updated = state.model_copy(update={"pending_approval": pending, "status": new_status})
    save_project_state(run_dir, updated)
    return updated


def load_pending_approval(run_dir: Path) -> dict[str, Any] | None:
    path = agent_workspace(run_dir) / "pending_approval.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


# Streamlit session-state helpers
SESSION_AGENTIC_KEYS = (
    "agentic_project_state",
    "agentic_pending_approval",
    "agentic_stop_requested",
    "agentic_enabled",
)


def init_agentic_session_state(st_module: Any) -> None:
    defaults = {
        "agentic_project_state": None,
        "agentic_pending_approval": None,
        "agentic_stop_requested": False,
    }
    for key, value in defaults.items():
        if key not in st_module.session_state:
            st_module.session_state[key] = value


def sync_agentic_session_from_disk(st_module: Any, run_dir: Path) -> None:
    state = load_project_state(run_dir)
    if state is not None:
        st_module.session_state["agentic_project_state"] = state.model_dump()
        st_module.session_state["agentic_pending_approval"] = state.pending_approval
