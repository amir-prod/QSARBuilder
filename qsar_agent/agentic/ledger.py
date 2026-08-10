"""Persistent experiment ledger and project state."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from qsar_agent.schemas.agentic import (
    AgenticProjectState,
    ExperimentRecord,
    SupervisorDecision,
)


def agent_workspace(run_dir: Path) -> Path:
    path = Path(run_dir) / "agent_workspace"
    path.mkdir(parents=True, exist_ok=True)
    return path


def experiment_dir(run_dir: Path, experiment_id: str) -> Path:
    path = agent_workspace(run_dir) / "experiments" / experiment_id
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_project_state(run_dir: Path, state: AgenticProjectState) -> Path:
    path = agent_workspace(run_dir) / "project_state.json"
    path.write_text(state.model_dump_json(indent=2), encoding="utf-8")
    return path


def load_project_state(run_dir: Path) -> AgenticProjectState | None:
    path = agent_workspace(run_dir) / "project_state.json"
    if not path.exists():
        return None
    return AgenticProjectState.model_validate_json(path.read_text(encoding="utf-8"))


def append_experiment_record(run_dir: Path, record: ExperimentRecord) -> Path:
    path = agent_workspace(run_dir) / "experiment_ledger.jsonl"
    with path.open("a", encoding="utf-8") as fh:
        fh.write(record.model_dump_json() + "\n")
    return path


def load_experiment_records(run_dir: Path) -> list[ExperimentRecord]:
    path = agent_workspace(run_dir) / "experiment_ledger.jsonl"
    if not path.exists():
        return []
    records: list[ExperimentRecord] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        records.append(ExperimentRecord.model_validate_json(line))
    return records


def get_experiment(run_dir: Path, experiment_id: str) -> ExperimentRecord | None:
    for rec in load_experiment_records(run_dir):
        if rec.experiment_id == experiment_id:
            return rec
    return None


def append_supervisor_decision(run_dir: Path, decision: SupervisorDecision) -> Path:
    path = agent_workspace(run_dir) / "supervisor_decisions.jsonl"
    with path.open("a", encoding="utf-8") as fh:
        fh.write(decision.model_dump_json() + "\n")
    return path


def next_experiment_id(run_dir: Path) -> str:
    existing = load_experiment_records(run_dir)
    return f"exp_{len(existing) + 1:03d}"


def save_json(path: Path, payload: dict[str, Any] | list[Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return path
