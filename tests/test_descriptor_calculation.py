"""Tests for DescJocky descriptor calculation adapter."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from qsar_agent.config import DescriptorConfig
from qsar_agent.tools.dataset_validation import validate_dataset
from qsar_agent.tools.descriptor_calculation import META_COLUMNS, calculate_descriptors
from tests.descriptor_test_utils import fake_descjocky_pipeline


def test_calculate_descriptors_aligns_meta_and_writes_artifacts(
    synthetic_dataset, tmp_run_dir
):
    cleaned = validate_dataset(
        synthetic_dataset,
        "smiles",
        "pIC50",
        "compound_id",
        tmp_run_dir,
        min_valid_compounds=15,
    )
    result = calculate_descriptors(
        cleaned.cleaned_dataset_path,
        tmp_run_dir,
        DescriptorConfig(backends=["RDKit"], run_geometry_optimization=False),
        pipeline_runner=fake_descjocky_pipeline,
    )

    assert Path(result.raw_descriptors_path).exists()
    assert Path(result.generated_descriptors_path).exists()
    assert Path(result.generated_descriptors_path).name == "generated_descriptors.csv"
    assert Path(result.calculation_report_path).exists()
    assert Path(result.calculation_report_md_path).exists()
    assert result.generated_descriptor_count == 20
    assert result.external_descriptor_count == 0
    assert result.descriptor_count == 20
    assert "Mordred" in result.backends or "RDKit" in result.backends
    assert result.backends_detail
    assert isinstance(result.three_d_descriptors_included, bool)
    assert result.generated_descriptor_columns

    report = json.loads(Path(result.calculation_report_path).read_text(encoding="utf-8"))
    assert "backends_detail" in report
    assert "three_d_descriptors_included" in report
    assert "generated_descriptor_columns" in report
    md = Path(result.calculation_report_md_path).read_text(encoding="utf-8")
    assert "Backends calculated" in md
    assert "3D status" in md

    df = pd.read_csv(result.raw_descriptors_path)
    for col in META_COLUMNS:
        assert col in df.columns
    assert "feat_0" in df.columns
    assert len(df) == result.compound_count
    assert (tmp_run_dir / "descjocky" / "smiles.txt").exists()
    assert (tmp_run_dir / "descjocky" / "mol_id_map.json").exists()
    assert (tmp_run_dir / "generated_descriptors.csv").exists()


def test_geometry_off_writes_light_sdfs(synthetic_dataset, tmp_run_dir):
    cleaned = validate_dataset(
        synthetic_dataset,
        "smiles",
        "pIC50",
        "compound_id",
        tmp_run_dir,
        min_valid_compounds=15,
    )
    result = calculate_descriptors(
        cleaned.cleaned_dataset_path,
        tmp_run_dir,
        DescriptorConfig(backends=["RDKit", "Mordred"], run_geometry_optimization=False),
        pipeline_runner=fake_descjocky_pipeline,
    )
    opt_dir = tmp_run_dir / "descjocky" / "mols" / "optimized"
    assert opt_dir.exists()
    assert any(opt_dir.glob("*.sdf"))
    assert result.three_d_descriptors_included is False
    assert result.three_d_geometries_used is False
    assert result.geometry_source == "rdkit_light_sdf_no_xtb"


def test_geometry_on_marks_3d_when_backend_supports_it(synthetic_dataset, tmp_run_dir):
    cleaned = validate_dataset(
        synthetic_dataset,
        "smiles",
        "pIC50",
        "compound_id",
        tmp_run_dir,
        min_valid_compounds=15,
    )
    result = calculate_descriptors(
        cleaned.cleaned_dataset_path,
        tmp_run_dir,
        DescriptorConfig(backends=["Mordred"], run_geometry_optimization=True),
        pipeline_runner=fake_descjocky_pipeline,
    )
    assert result.three_d_geometries_used is True
    assert result.three_d_descriptors_included is True
    assert result.geometry_source == "xtb"
