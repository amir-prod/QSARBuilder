"""Append-only agent experiment ledger."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from qsar_agent.services.artifact_manager import atomic_write_text

LEDGER_COLUMNS = [
    "experiment_id",
    "parent_experiment",
    "timestamp",
    "diagnosis",
    "hypothesis",
    "tool_name",
    "arguments",
    "dataset_hash",
    "development_split_hash",
    "model",
    "representation",
    "feature_selection_config",
    "selected_features",
    "cv_r2",
    "cv_r2_std",
    "train_cv_gap",
    "refit_train_cv_gap",
    "cv_rmse",
    "cv_mae",
    "val_r2",
    "feature_count",
    "acceptance_status",
    "agent_interpretation",
    "runtime_seconds",
    "compute_usage",
    "artifact_paths",
    "decision",
]


def ledger_path(agent_dir: Path) -> Path:
    return Path(agent_dir) / "agent_experiment_ledger.csv"


def existing_experiment_ids(agent_dir: Path) -> set[str]:
    path = ledger_path(agent_dir)
    if not path.is_file():
        return set()
    with path.open(encoding="utf-8", newline="") as handle:
        return {row["experiment_id"] for row in csv.DictReader(handle) if row.get("experiment_id")}


def append_ledger_row(agent_dir: Path, row: dict[str, Any]) -> None:
    """Append one row. Never rewrite previous results. Skip if the ID already exists."""
    agent_dir = Path(agent_dir)
    agent_dir.mkdir(parents=True, exist_ok=True)
    path = ledger_path(agent_dir)
    experiment_id = str(row.get("experiment_id") or "")
    if experiment_id and experiment_id in existing_experiment_ids(agent_dir):
        return
    payload = {col: row.get(col, "") for col in LEDGER_COLUMNS}
    if not payload["timestamp"]:
        payload["timestamp"] = datetime.now(timezone.utc).isoformat()
    write_header = not path.is_file()
    # Append is not fully atomic across processes, but we never rewrite old rows.
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=LEDGER_COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerow({k: _cell(v) for k, v in payload.items()})


def _cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, dict)):
        import json

        return json.dumps(value, default=str, sort_keys=True)
    return str(value)


def write_decisions_jsonl(agent_dir: Path, decision: dict[str, Any]) -> None:
    import json

    path = Path(agent_dir) / "agent_decisions.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(decision, default=str) + "\n"
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line)


def dump_agent_state(agent_dir: Path, state: dict[str, Any]) -> None:
    import json

    path = Path(agent_dir) / "agent_state.json"
    atomic_write_text(path, json.dumps(state, indent=2, default=str) + "\n")
