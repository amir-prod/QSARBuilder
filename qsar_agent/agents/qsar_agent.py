"""QSAR Workflow Agent orchestration."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from qsar_agent.config import WorkflowConfig, get_openai_api_key, get_openai_model
from qsar_agent.schemas.feature_selection import FeatureCountSelection, SFSResult
from qsar_agent.schemas.hyperparameter_optimization import AgentGridProposal
from qsar_agent.schemas.workflow import AgentFinalReport
from qsar_agent.tools.feature_count_selection import (
    save_feature_count_selection,
    select_feature_count_one_se_rule,
)
from qsar_agent.models.registry import (
    count_grid_combinations,
    get_fallback_grid,
    get_hpo_prompt_spec,
)


def _parse_agent_json(content: str) -> dict[str, Any]:
    text = content.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        lines = [ln for ln in lines if not ln.strip().startswith("```")]
        text = "\n".join(lines)
    return json.loads(text)


def _summarize_hpo_round_for_prompt(rr: Any, top_k: int = 5) -> dict[str, Any]:
    """Build a rich, JSON-serializable summary of one completed HPO round."""
    summary = rr.best_cv_summary
    assessment = rr.assessment
    top_candidates = []
    for cand in sorted(rr.candidates, key=lambda c: c.rank)[:top_k]:
        top_candidates.append(
            {
                "rank": cand.rank,
                "params": cand.params,
                "mean_train_r2": cand.mean_train_r2,
                "mean_cv_r2": cand.mean_cv_r2,
                "std_cv_r2": cand.std_cv_r2,
                "train_cv_r2_gap": cand.train_cv_r2_gap,
                "is_best": cand.is_best,
            }
        )

    proposal = getattr(rr, "agent_proposal", None)
    sanitization = getattr(rr, "sanitization", None)
    return {
        "round": rr.round_index,
        "best_params": rr.best_params,
        "candidates_searched": getattr(rr, "candidates_searched", len(rr.candidates)),
        "cv_summary": {
            "mean_train_r2": summary.mean_train_r2,
            "mean_cv_r2": summary.mean_cv_r2,
            "std_cv_r2": summary.std_cv_r2,
            "train_cv_r2_gap": summary.train_cv_r2_gap,
        },
        "assessment": {
            "status": assessment.status,
            "is_acceptable": assessment.is_acceptable,
            "is_overfit": assessment.is_overfit,
            "is_underfit": assessment.is_underfit,
            "is_unstable": assessment.is_unstable,
            "is_severe_overfit": assessment.is_severe_overfit,
            "mean_train_r2": assessment.mean_train_r2,
            "mean_cv_r2": assessment.mean_cv_r2,
            "train_cv_r2_gap": assessment.train_cv_r2_gap,
            "cv_r2_std": assessment.cv_r2_std,
            "warnings": list(assessment.warnings),
            "explanation": assessment.explanation,
        },
        "search_strategy": getattr(proposal, "search_strategy", None) if proposal else None,
        "proposed_grid": getattr(proposal, "proposed_grid", None) if proposal else None,
        "sanitized_grid": (
            getattr(sanitization, "sanitized_grid", None) if sanitization else None
        ),
        "top_candidates": top_candidates,
    }


def propose_hyperparameter_grid(
    round_index: int,
    model_type: str,
    baseline_assessment: Any,
    previous_hpo_results: list[Any],
    constraints: dict[str, Any],
    openai_model: str | None = None,
    use_openai: bool = True,
) -> AgentGridProposal:
    """
    Ask OpenAI to propose a structured hyperparameter grid.

    Later rounds receive rich feedback from prior HPO rounds and are instructed
    to refine around the previous best rather than restart from scratch.
    Falls back to deterministic grids when the API is unavailable or output is invalid.
    """
    max_candidates = constraints.get("max_candidates", 120)
    n_features = constraints.get("n_features")
    n_train_samples = constraints.get("n_train_samples")

    def _fallback(reason: str) -> AgentGridProposal:
        # Prefer latest round status when available so fallback templates track progress.
        status = baseline_assessment.status
        if previous_hpo_results:
            status = previous_hpo_results[-1].assessment.status
        grid = get_fallback_grid(model_type, status)
        return AgentGridProposal(
            round_index=round_index,
            reasoning_summary=f"Deterministic fallback: {reason}",
            search_strategy="fallback",
            proposed_grid=grid,
            expected_effect_on_overfitting="Template grid for detected issue.",
            expected_effect_on_underfitting="Template grid for detected issue.",
            computational_cost_estimate=str(count_grid_combinations(grid)),
            warnings=[reason],
        )

    api_key = get_openai_api_key() if use_openai else None
    if not api_key:
        return _fallback("OpenAI API key not configured.")

    prev_summary = [_summarize_hpo_round_for_prompt(rr) for rr in previous_hpo_results]
    latest_round = prev_summary[-1] if prev_summary else None
    has_prior_rounds = bool(previous_hpo_results)

    hpo_spec = get_hpo_prompt_spec(model_type) or get_hpo_prompt_spec("RandomForestRegressor")
    refine_instructions = (
        "Round 1: propose an initial grid that addresses the baseline assessment "
        "while respecting the dataset size (small n_train => prefer stronger regularization "
        "and compact grids)."
        if not has_prior_rounds
        else (
            "Follow-up round: do NOT start from scratch. Use the latest round feedback as the "
            "primary signal. Center the new grid around the previous best_params, then make "
            "small local adjustments that specifically target the latest assessment status "
            "(e.g. more regularization for overfit/unstable, more capacity for underfit/"
            "poor_performance). Keep promising nearby values from top_candidates; drop "
            "clearly worse regions. Respect dataset size when choosing ranges."
        )
    )
    system_prompt = (
        "You are a QSAR modeling assistant. Propose ONLY a JSON hyperparameter search grid "
        f"for sklearn {model_type}. Do NOT train models or invent metrics. "
        f"{hpo_spec} "
        f"Keep total combinations near or below {max_candidates}. "
        f"{refine_instructions} "
        "In reasoning_summary, explicitly mention how you used prior-round feedback "
        "(or baseline only for round 1) and the dataset size. "
        "Return JSON matching the AgentGridProposal schema fields."
    )

    dataset_block = (
        "Dataset size (critical for regularization choices):\n"
        f"- n_train_samples (training compounds): {n_train_samples}\n"
        f"- n_features (selected descriptors): {n_features}\n"
        f"- samples_per_feature: "
        f"{(float(n_train_samples) / float(n_features)) if n_train_samples and n_features else 'unknown'}\n"
    )

    if latest_round is not None:
        latest_block = (
            "LATEST ROUND FEEDBACK (primary signal for this proposal):\n"
            f"{json.dumps(latest_round, indent=2)}\n\n"
            "All previous HPO rounds (oldest to newest):\n"
            f"{json.dumps(prev_summary, indent=2)}\n"
        )
        strategy_hint = (
            f"Primary problem to fix now: status={latest_round['assessment']['status']} "
            f"(acceptable={latest_round['assessment']['is_acceptable']}). "
            f"Refine around best_params={json.dumps(latest_round['best_params'])}."
        )
    else:
        latest_block = "No previous HPO rounds yet; this is the first search round.\n"
        strategy_hint = (
            f"Primary problem to fix now: baseline status={baseline_assessment.status} "
            f"(acceptable={baseline_assessment.is_acceptable})."
        )

    user_prompt = (
        f"Round: {round_index}\n"
        f"Model type: {model_type}\n"
        f"{dataset_block}\n"
        f"{strategy_hint}\n\n"
        f"Baseline assessment (original, before HPO):\n"
        f"{baseline_assessment.model_dump_json()}\n\n"
        f"{latest_block}\n"
        f"Max candidates: {max_candidates}\n"
        "Respond with JSON only:\n"
        "{"
        '"round_index": int, "reasoning_summary": str, "search_strategy": str, '
        '"proposed_grid": {param: [values]}, '
        '"expected_effect_on_overfitting": str, '
        '"expected_effect_on_underfitting": str, '
        '"computational_cost_estimate": str, "warnings": [str]'
        "}"
    )

    model = openai_model or get_openai_model()

    def _call_and_validate(repair: bool = False) -> AgentGridProposal:
        from openai import OpenAI

        client = OpenAI(api_key=api_key)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        if repair:
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "Your previous response was invalid. Return valid JSON only, "
                        "with allowed hyperparameter values."
                    ),
                }
            )
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0,
        )
        raw = response.choices[0].message.content or ""
        data = _parse_agent_json(raw)
        data["round_index"] = round_index
        return AgentGridProposal.model_validate(data)

    try:
        return _call_and_validate()
    except (json.JSONDecodeError, ValidationError, Exception):
        try:
            return _call_and_validate(repair=True)
        except Exception as exc:
            return _fallback(f"Agent grid validation failed: {exc}")


def run_agent_feature_count_selection(
    sfs_result: SFSResult,
    run_dir: Path,
    use_openai: bool = True,
) -> FeatureCountSelection:
    """
    Select optimal descriptor count using one-standard-error rule.

    When OpenAI is available, the agent validates and explains the deterministic
    selection; it does not override the numerical rule.
    """
    selection = select_feature_count_one_se_rule(sfs_result)
    selection = save_feature_count_selection(selection, run_dir)

    api_key = get_openai_api_key() if use_openai else None
    if api_key:
        try:
            from openai import OpenAI

            client = OpenAI(api_key=api_key)
            prompt = (
                "You are the QSAR Workflow Agent. Based on these sequential feature "
                "selection results, explain the one-standard-error rule selection. "
                "Do NOT invent metrics or change the selected count.\n\n"
                f"Deterministic selection: {selection.model_dump_json()}\n\n"
                f"SFS results: {json.dumps([r.model_dump() for r in sfs_result.results])}"
            )
            response = client.chat.completions.create(
                model=get_openai_model(),
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You explain QSAR feature selection decisions based only on "
                            "provided numerical results. Never fabricate metrics."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0,
            )
            agent_text = response.choices[0].message.content or ""
            if agent_text:
                selection = selection.model_copy(
                    update={"explanation": selection.explanation + "\n\n" + agent_text}
                )
                save_feature_count_selection(selection, run_dir)
        except Exception:
            pass

    return selection


def build_final_report(
    run_id: str,
    validation_result: Any,
    descriptor_result: Any,
    preprocessing_result: Any,
    split_result: Any,
    feature_selection: FeatureCountSelection,
    ga_result: Any,
    modeling_result: Any,
    ad_result: Any,
    artifact_paths: dict[str, str],
    warnings: list[str],
    estimator: str = "RandomForestRegressor",
    model_comparison_summary: str = "",
) -> AgentFinalReport:
    return AgentFinalReport(
        run_id=run_id,
        dataset_size=validation_result.original_row_count,
        valid_compounds=validation_result.valid_compound_count,
        train_size=split_result.train_count,
        val_size=split_result.val_count,
        test_size=split_result.test_count,
        initial_descriptor_count=descriptor_result.descriptor_count,
        final_preprocessed_descriptors=preprocessing_result.final_descriptor_count,
        selected_descriptor_count=feature_selection.selected_feature_count,
        ga_selected_descriptors=ga_result.selected_features,
        train_metrics=modeling_result.train_metrics,
        val_metrics=getattr(modeling_result, "val_metrics", None),
        test_metrics=modeling_result.test_metrics,
        applicability_domain_summary=ad_result.summary,
        warnings=warnings,
        artifact_paths=artifact_paths,
        agent_explanation=feature_selection.explanation,
        estimator=estimator,
        model_comparison_summary=model_comparison_summary,
    )
