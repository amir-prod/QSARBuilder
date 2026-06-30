"""QSAR Workflow Agent orchestration."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from qsar_agent.config import WorkflowConfig, get_openai_api_key, get_openai_model
from qsar_agent.schemas.feature_selection import FeatureCountSelection, SFSResult
from qsar_agent.schemas.workflow import AgentFinalReport
from qsar_agent.tools.feature_count_selection import (
    save_feature_count_selection,
    select_feature_count_one_se_rule,
)


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
    )
