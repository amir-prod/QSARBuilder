"""Final agent report and snapshot helpers."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from qsar_agent.agentic.ledger import dump_agent_state
from qsar_agent.agentic.ranking import rank_candidates
from qsar_agent.agentic.requirements import requirements_from_config
from qsar_agent.config import WorkflowConfig
from qsar_agent.schemas.agentic import ModelingAgentState
from qsar_agent.services.artifact_manager import atomic_write_text


def write_final_report_md(state: ModelingAgentState, config: WorkflowConfig, agent_dir: Path) -> str:
    rankings = rank_candidates(state.completed_experiments, requirements_from_config(config))
    ranking_lines = []
    for row in rankings:
        ranking_lines.append(
            f"{row.rank}. `{row.experiment_id}` eligible={row.eligible} "
            f"score={row.selection_score:.4f} — {row.selection_reason}"
        )
    best = rankings[0] if rankings else None
    if state.stopping_reason == "requirements_satisfied" and best is not None:
        headline = (
            f"Selected pipeline `{best.experiment_id}` because it satisfies every hard "
            f"requirement and ranks first under the fixed policy: {best.selection_reason}."
        )
    else:
        failed = state.failed_requirements or []
        names = ", ".join(
            item.get("name", "?") if isinstance(item, dict) else str(item) for item in failed
        )
        headline = (
            f"Search stopped with reason `{state.stopping_reason or 'unknown'}`. "
            f"Unsatisfied requirements: {names or 'none recorded'}. "
            f"Attempted {state.adaptive_experiments_used} adaptive experiments "
            f"over {state.agent_iteration} iterations."
        )
        if state.pending_capability_request:
            headline += (
                " A capability request was written; additional data or a new "
                "deterministic tool may be required."
            )
    sealed = ""
    if state.sealed_test_result:
        sealed = (
            "\n## Sealed external-test evaluation\n\n"
            "Evaluated once after freeze. These confirmatory metrics were not used "
            "for model improvement.\n\n"
            f"```json\n{state.sealed_test_result}\n```\n"
        )
    body = f"""# Modeling-improvement agent report

Generated: {datetime.now(timezone.utc).isoformat()}
Project: `{state.project_id}`
Phase: `{state.phase}`
Stopping reason: `{state.stopping_reason or "completed"}`

## Outcome

{headline}

## Ranking (deterministic policy)

Hard-requirement passers rank above failers, then outer-CV R², lower error,
lower variability, smaller train–CV gap, fewer features, simpler estimator.

{chr(10).join(ranking_lines) or "_No adaptive experiments were recorded._"}

## Diagnosis trail

- Last diagnosis: {state.current_diagnosis or "n/a"}
- Last hypothesis: {state.current_hypothesis or "n/a"}
- Stagnation count: {state.stagnation_count}

## Integrity

- dataset_hash: `{state.dataset_hash}`
- development_split_hash: `{state.development_split_hash}`
- sealed_test_hash: `{state.sealed_test_hash}`
- handoff validation passed: {state.validation_passed}

{sealed}
"""
    path = Path(agent_dir) / "agent_final_report.md"
    atomic_write_text(path, body)
    return str(path)


def snapshot_state(state: ModelingAgentState, agent_dir: Path) -> None:
    dump_agent_state(agent_dir, state.model_dump(mode="json"))
