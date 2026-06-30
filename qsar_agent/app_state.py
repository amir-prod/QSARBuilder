"""Streamlit session-state helpers."""

from __future__ import annotations

from typing import Any

import streamlit as st

from qsar_agent.schemas.workflow import StageStatus, WorkflowState


def init_session_state() -> None:
    defaults: dict[str, Any] = {
        "run_id": None,
        "uploaded_filename": None,
        "dataset_preview": None,
        "column_mapping": {},
        "workflow_config": None,
        "workflow_state": None,
        "stage_statuses": {},
        "logs": [],
        "warnings": [],
        "final_report": None,
        "artifact_paths": {},
        "validation_result": None,
        "workflow_running": False,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def get_workflow_state() -> WorkflowState | None:
    return st.session_state.get("workflow_state")


def set_workflow_state(state: WorkflowState) -> None:
    st.session_state["workflow_state"] = state
    st.session_state["stage_statuses"] = {
        s.stage: s.status for s in state.stages
    }
    st.session_state["logs"] = state.logs
    st.session_state["warnings"] = state.warnings
    st.session_state["artifact_paths"] = state.artifact_paths
    st.session_state["final_report"] = state.final_report
    st.session_state["run_id"] = state.run_id


def reset_session() -> None:
    keys_to_clear = [
        "run_id",
        "uploaded_filename",
        "dataset_preview",
        "column_mapping",
        "workflow_config",
        "workflow_state",
        "stage_statuses",
        "logs",
        "warnings",
        "final_report",
        "artifact_paths",
        "validation_result",
        "workflow_running",
    ]
    for key in keys_to_clear:
        if key in st.session_state:
            del st.session_state[key]
    init_session_state()


def update_stage_status(stage: str, status: StageStatus) -> None:
    st.session_state.setdefault("stage_statuses", {})[stage] = status.value
