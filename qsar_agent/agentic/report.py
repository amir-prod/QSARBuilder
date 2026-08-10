"""Generate final_agent_report.md."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from qsar_agent.agentic.ledger import agent_workspace, load_experiment_records, load_project_state
from qsar_agent.schemas.agentic import AcceptanceResult, ValidationReview


def write_final_agent_report(
    run_dir: Path,
    *,
    initial_assessment: str,
    acceptance: AcceptanceResult | None,
    validation: ValidationReview | None,
    selection_rationale: str,
    stopping_reason: str,
    external_metrics: dict[str, Any] | None = None,
    ad_summary: dict[str, Any] | None = None,
) -> Path:
    state = load_project_state(run_dir)
    records = load_experiment_records(run_dir)
    lines: list[str] = []
    lines.append("# Agentic Model Improvement Report")
    lines.append("")
    lines.append("## 1. Initial model assessment")
    lines.append(initial_assessment)
    lines.append("")
    lines.append("## 2. Acceptance criteria")
    if state is not None:
        lines.append("```json")
        lines.append(state.acceptance_criteria.model_dump_json(indent=2))
        lines.append("```")
    if acceptance is not None:
        lines.append(acceptance.explanation)
        lines.append(f"Accepted: {acceptance.accepted}")
    lines.append("")
    lines.append("## 3. Failure diagnoses")
    lines.append("See specialist reports under `agent_workspace/specialist_reports/`.")
    lines.append("")
    lines.append("## 4. Experiments attempted")
    for rec in records:
        lines.append(
            f"- `{rec.experiment_id}` parent={rec.parent_experiment_id} action={rec.action} "
            f"kind={rec.experiment_kind} multi_component={rec.multi_component} "
            f"mean_cv_r2={rec.internal_metrics.get('mean_cv_r2')}"
        )
    lines.append("")
    lines.append("## 5. Hypotheses and configuration changes")
    for rec in records:
        lines.append(f"### {rec.experiment_id}")
        lines.append(f"- Hypothesis: {rec.hypothesis}")
        lines.append(f"- Conclusion: {rec.conclusion}")
        lines.append(f"- Config: `{rec.config_snapshot}`")
    lines.append("")
    lines.append("## 6. Internal CV comparisons")
    for rec in records:
        lines.append(f"- {rec.experiment_id}: {rec.comparison_to_parent}")
    lines.append("")
    lines.append("## 7. Rejected or duplicated experiments")
    lines.append("See `supervisor_decisions.jsonl` for rejected proposals.")
    lines.append("")
    lines.append("## 8. Winning experiment and deterministic selection rationale")
    if state is not None:
        lines.append(f"Best experiment: `{state.best_experiment_id}`")
        if state.lock_record:
            lines.append(state.lock_record.selection_rationale)
    lines.append(selection_rationale)
    lines.append("")
    lines.append("## 9. Validation Agent review")
    if validation is not None:
        lines.append(validation.summary)
        lines.append(f"Hard veto: {validation.hard_veto}")
        lines.append(f"Soft rejection recommended: {validation.soft_rejection_recommended}")
        for w in validation.warnings:
            lines.append(f"- Warning: {w}")
    lines.append("")
    lines.append("## 10. Stopping reason")
    lines.append(stopping_reason)
    lines.append("")
    lines.append("## 11. External-test isolation statement")
    lines.append(
        "If external-test results influence further model development, that test set is "
        "no longer independent and must not be reported as an untouched external test. "
        "In this run, agentic development used training / agent-development / protected "
        "agent-validation evidence only. External evaluation occurred only after model lock."
    )
    lines.append("")
    lines.append("## 12. Final external-test metrics (post-lock only)")
    if external_metrics:
        lines.append("```json")
        import json

        lines.append(json.dumps(external_metrics, indent=2, default=str))
        lines.append("```")
    else:
        lines.append("Not yet evaluated or unavailable.")
    lines.append("")
    lines.append("## 13. Applicability-domain summary")
    if ad_summary:
        lines.append(str(ad_summary))
    else:
        lines.append("See applicability domain artifacts after external evaluation.")
    lines.append("")
    lines.append("## 14. Limitations")
    lines.append(
        "- Protected agent-validation is not full nested CV; adaptive search can still "
        "overfit the agent-validation set across cycles."
    )
    lines.append("- Adaptive experiment selection introduces multiple-comparison bias.")
    lines.append("- Data Quality Agent is diagnostic-only in v1.")
    lines.append("- Optional boosting libraries require installed dependencies.")
    lines.append("")
    lines.append("## 15. Artifact references")
    lines.append(f"- Run directory: `{run_dir}`")
    lines.append(f"- Agent workspace: `{agent_workspace(run_dir)}`")
    lines.append(f"- Locked external: `{Path(run_dir) / 'locked_external'}`")

    out = agent_workspace(run_dir) / "final_agent_report.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out
