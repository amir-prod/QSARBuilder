"""Persistent-outlier aggregation and exclusion sensitivity (no row deletion)."""

from __future__ import annotations

import pandas as pd
import pytest

from qsar_agent.agentic.tools import execute_tool
from qsar_agent.schemas.agentic import PipelinePhase
from qsar_agent.tools.outlier_persistence import (
    mask_compounds,
    persistent_outliers_from_oof,
    proposal_from_report,
)
from tests.agentic_harness import write_agent_run, write_development_tables


def _oof_table(ids, residuals, family="RF"):
    return pd.DataFrame(
        {
            "compound_id": ids,
            "activity": [0.0] * len(ids),
            "predicted_activity": [-r for r in residuals],
            "residual": residuals,
        }
    )


def test_persistent_outliers_flag_repeated_residual_hits():
    ids = [f"C{i:03d}" for i in range(20)]
    normal = [0.05 * ((-1) ** i) for i in range(19)] + [12.0]
    tables = [_oof_table(ids, normal) for _ in range(3)]
    reports = persistent_outliers_from_oof(
        tables,
        structural_flags={"C019": 0.8},
        model_family="RandomForestRegressor",
        residual_z_threshold=2.5,
    )
    by_id = {r.compound_id: r for r in reports}
    assert "C019" in by_id
    assert by_id["C019"].oof_response_outlier_frequency == pytest.approx(1.0)
    assert by_id["C019"].recommended_action == "propose_exclusion"
    proposal = proposal_from_report(by_id["C019"])
    assert proposal.required_approval is True
    assert proposal.compound_id == "C019"


def test_mask_compounds_does_not_write_and_keeps_other_rows(tmp_path):
    path = tmp_path / "train.csv"
    df = pd.DataFrame({"compound_id": ["a", "b", "c"], "activity": [1.0, 2.0, 3.0]})
    df.to_csv(path, index=False)
    reduced = mask_compounds(df, ["b"])
    assert list(reduced["compound_id"]) == ["a", "c"]
    assert pd.read_csv(path)["compound_id"].tolist() == ["a", "b", "c"]


def test_exclusion_sensitivity_requires_approval(tmp_path):
    run_dir = write_agent_run(tmp_path, passing=False)
    write_development_tables(run_dir)
    with pytest.raises(PermissionError, match="approval"):
        execute_tool(
            "run_exclusion_sensitivity_analysis",
            {"compound_id": "C000", "selected_features": ["feat_0", "feat_1", "feat_2"]},
            run_dir=run_dir,
            state_phase=PipelinePhase.DEVELOPMENT,
            dataset_hash="ds",
            development_split_hash="dev",
            selected_features=["feat_0", "feat_1", "feat_2"],
            exclusion_approved=False,
        )


def test_approved_exclusion_leaves_original_train_csv(tmp_path):
    run_dir = write_agent_run(tmp_path, passing=False)
    write_development_tables(run_dir)
    train_path = run_dir / "preprocessed_train_descriptors.csv"
    before = train_path.read_text(encoding="utf-8")
    result = execute_tool(
        "run_exclusion_sensitivity_analysis",
        {
            "compound_id": "C000",
            "approved": True,
            "selected_features": ["feat_0", "feat_1", "feat_2"],
        },
        run_dir=run_dir,
        state_phase=PipelinePhase.DEVELOPMENT,
        dataset_hash="ds",
        development_split_hash="dev",
        selected_features=["feat_0", "feat_1", "feat_2"],
        exclusion_approved=True,
    )
    assert result.tool_name == "run_exclusion_sensitivity_analysis"
    assert "delta" in result.extra
    assert train_path.read_text(encoding="utf-8") == before
    assert "C000" in before
