"""Append-only agent event log (no secrets)."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from qsar_agent.agentic.ledger import agent_workspace
from qsar_agent.schemas.agentic import AgentEvent, DecisionSource


def append_event(
    run_dir: Path,
    *,
    event_type: str,
    cycle_index: int | None = None,
    experiment_id: str | None = None,
    agent: str | None = None,
    input_artifact_refs: list[str] | None = None,
    validated_response: dict[str, Any] | None = None,
    selected_action: str | None = None,
    tool_execution: dict[str, Any] | None = None,
    error: str | None = None,
    retry_or_fallback: str | None = None,
    approval_state: str | None = None,
    token_usage: dict[str, Any] | None = None,
    decision_source: DecisionSource | None = None,
) -> AgentEvent:
    event = AgentEvent(
        timestamp=datetime.now(timezone.utc).isoformat(),
        cycle_index=cycle_index,
        experiment_id=experiment_id,
        agent=agent,
        event_type=event_type,
        input_artifact_refs=input_artifact_refs or [],
        validated_response=validated_response,
        selected_action=selected_action,
        tool_execution=tool_execution,
        error=error,
        retry_or_fallback=retry_or_fallback,
        approval_state=approval_state,
        token_usage=token_usage,
        decision_source=decision_source,
    )
    path = agent_workspace(run_dir) / "agent_events.jsonl"
    # Redact obvious secret keys if somehow present
    payload = event.model_dump()
    _redact_secrets(payload)
    with path.open("a", encoding="utf-8") as fh:
        import json

        fh.write(json.dumps(payload, default=str) + "\n")
    return event


def load_events(run_dir: Path) -> list[AgentEvent]:
    path = agent_workspace(run_dir) / "agent_events.jsonl"
    if not path.exists():
        return []
    events: list[AgentEvent] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            events.append(AgentEvent.model_validate_json(line))
    return events


def _redact_secrets(obj: Any) -> None:
    if isinstance(obj, dict):
        for k, v in list(obj.items()):
            if any(s in str(k).lower() for s in ("api_key", "secret", "password", "token")):
                if k != "token_usage":
                    obj[k] = "***REDACTED***"
            else:
                _redact_secrets(v)
    elif isinstance(obj, list):
        for item in obj:
            _redact_secrets(item)
