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

    Falls back to deterministic grids when the API is unavailable or output is invalid.
    """
    max_candidates = constraints.get("max_candidates", 120)
    n_features = constraints.get("n_features")
    n_train_samples = constraints.get("n_train_samples")

    def _fallback(reason: str) -> AgentGridProposal:
        grid = get_fallback_grid(model_type, baseline_assessment.status)
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

    prev_summary = []
    for rr in previous_hpo_results:
        prev_summary.append(
            {
                "round": rr.round_index,
                "best_params": rr.best_params,
                "mean_cv_r2": rr.best_cv_summary.mean_cv_r2,
                "gap": rr.assessment.train_cv_r2_gap,
                "status": rr.assessment.status,
            }
        )

    hpo_spec = get_hpo_prompt_spec(model_type) or get_hpo_prompt_spec("RandomForestRegressor")
    system_prompt = (
        "You are a QSAR modeling assistant. Propose ONLY a JSON hyperparameter search grid "
        f"for sklearn {model_type}. Do NOT train models or invent metrics. "
        f"{hpo_spec} "
        f"Keep total combinations near or below {max_candidates}. "
        "Return JSON matching the AgentGridProposal schema fields."
    )

    user_prompt = (
        f"Round: {round_index}\n"
        f"Model type: {model_type}\n"
        f"Baseline assessment: {baseline_assessment.model_dump_json()}\n"
        f"Previous HPO rounds: {json.dumps(prev_summary)}\n"
        f"Dataset: n_train={n_train_samples}, n_features={n_features}\n"
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
    mordred_result: Any,
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
        test_size=split_result.test_count,
        initial_mordred_descriptors=mordred_result.descriptor_count,
        final_preprocessed_descriptors=preprocessing_result.final_descriptor_count,
        selected_descriptor_count=feature_selection.selected_feature_count,
        ga_selected_descriptors=ga_result.selected_features,
        train_metrics=modeling_result.train_metrics,
        test_metrics=modeling_result.test_metrics,
        applicability_domain_summary=ad_result.summary,
        warnings=warnings,
        artifact_paths=artifact_paths,
        agent_explanation=feature_selection.explanation,
        estimator=estimator,
        model_comparison_summary=model_comparison_summary,
    )
