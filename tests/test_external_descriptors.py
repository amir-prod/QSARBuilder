"""Tests for external descriptor merge."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from qsar_agent.config import DescriptorConfig
from qsar_agent.tools.dataset_validation import validate_dataset
from qsar_agent.tools.descriptor_calculation import (
    META_COLUMNS,
    calculate_descriptors,
    merge_external_descriptors,
)
from tests.descriptor_test_utils import fake_descjocky_pipeline


def test_merge_external_on_compound_id():
    generated = pd.DataFrame(
        {
            "compound_id": ["C1", "C2", "C3"],
            "canonical_smiles": ["CCO", "CCN", "CCC"],
            "activity": [1.0, 2.0, 3.0],
            "original_row_index": [0, 1, 2],
            "feat_a": [0.1, 0.2, 0.3],
        }
    )
    # write external via merge helper using a temp file in cwd of fixture-less test
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "ext.csv"
        pd.DataFrame(
            {
                "compound_id": ["C1", "C2", "C9"],
                "ext_feat": [10.0, 20.0, 99.0],
                "feat_a": [1.0, 2.0, 3.0],  # collision
            }
        ).to_csv(path, index=False)

        merged, warnings, n_ext = merge_external_descriptors(generated, path)

    assert n_ext == 2
    assert "ext_feat" in merged.columns
    assert "ext__feat_a" in merged.columns
    assert "feat_a" in merged.columns
    assert merged.loc[merged["compound_id"] == "C1", "ext_feat"].iloc[0] == 10.0
    assert pd.isna(merged.loc[merged["compound_id"] == "C3", "ext_feat"].iloc[0])
    assert any("ext__" in w for w in warnings)
    assert any("not in generated" in w for w in warnings)
    assert any("no external" in w for w in warnings)


def test_merge_requires_compound_id():
    generated = pd.DataFrame(
        {
            "compound_id": ["C1"],
            "canonical_smiles": ["CCO"],
            "activity": [1.0],
            "original_row_index": [0],
            "feat_a": [0.1],
        }
    )
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "bad.csv"
        pd.DataFrame({"smiles": ["CCO"], "x": [1]}).to_csv(path, index=False)
        with pytest.raises(ValueError, match="compound_id"):
            merge_external_descriptors(generated, path)


def test_calculate_descriptors_merges_external(synthetic_dataset, tmp_run_dir):
    cleaned = validate_dataset(
        synthetic_dataset,
        "smiles",
        "pIC50",
        "compound_id",
        tmp_run_dir,
        min_valid_compounds=15,
    )
    cleaned_df = pd.read_csv(cleaned.cleaned_dataset_path)
    ext_path = tmp_run_dir / "user_external.csv"
    ext = cleaned_df[["compound_id"]].copy()
    ext["user_desc"] = range(len(ext))
    ext.to_csv(ext_path, index=False)

    result = calculate_descriptors(
        cleaned.cleaned_dataset_path,
        tmp_run_dir,
        DescriptorConfig(
            backends=["RDKit"],
            run_geometry_optimization=False,
            external_descriptors_path=str(ext_path),
        ),
        pipeline_runner=fake_descjocky_pipeline,
    )
    assert result.external_descriptor_count == 1
    df = pd.read_csv(result.raw_descriptors_path)
    assert "user_desc" in df.columns
    assert all(c in df.columns for c in META_COLUMNS)
    assert Path(result.external_descriptors_path).exists()
