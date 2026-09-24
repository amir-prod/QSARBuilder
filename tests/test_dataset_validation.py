"""Tests for dataset validation."""

import pandas as pd
import pytest

from qsar_agent.tools.dataset_validation import validate_dataset


def test_missing_required_column(tmp_run_dir):
    df = pd.DataFrame({"smiles": ["CCO"], "activity": [1.0]})
    path = tmp_run_dir / "data.csv"
    df.to_csv(path, index=False)
    with pytest.raises(ValueError, match="not found"):
        validate_dataset(path, "smiles", "missing_col", None, tmp_run_dir)


def test_invalid_smiles(tmp_run_dir):
    smiles_list = [f"C{'C' * i}O" for i in range(20)] + ["invalid_xyz"] * 5
    act = list(range(len(smiles_list)))
    df = pd.DataFrame({"smiles": smiles_list, "activity": act})
    path = tmp_run_dir / "data.csv"
    df.to_csv(path, index=False)
    result = validate_dataset(path, "smiles", "activity", None, tmp_run_dir, min_valid_compounds=15)
    assert result.invalid_smiles_count >= 1
    assert result.valid_compound_count == 20


def test_missing_activity(tmp_run_dir):
    smiles = [f"C{'C' * i}O" for i in range(25)]
    act = [float(i) for i in range(20)] + [None] * 5
    df = pd.DataFrame({"smiles": smiles, "activity": act})
    path = tmp_run_dir / "data.csv"
    df.to_csv(path, index=False)
    result = validate_dataset(path, "smiles", "activity", None, tmp_run_dir, min_valid_compounds=15)
    assert result.missing_or_invalid_activity_count == 5


def test_duplicate_compounds(tmp_run_dir):
    # The generated series starts at "CCO" (i=1), so ethanol appears 11 times in
    # total and 10 copies are removed, leaving 15 unique structures.
    smiles = ["CCO"] * 10 + [f"C{'C' * i}O" for i in range(1, 16)]
    df = pd.DataFrame({
        "smiles": smiles,
        "activity": list(range(len(smiles))),
    })
    path = tmp_run_dir / "data.csv"
    df.to_csv(path, index=False)
    result = validate_dataset(path, "smiles", "activity", None, tmp_run_dir, min_valid_compounds=10)
    assert result.duplicate_compound_count == 10
    assert result.valid_compound_count == 15


def test_too_few_compounds(tmp_run_dir):
    df = pd.DataFrame({"smiles": ["CCO", "CCC"], "activity": [1.0, 2.0]})
    path = tmp_run_dir / "data.csv"
    df.to_csv(path, index=False)
    with pytest.raises(ValueError, match="Too few"):
        validate_dataset(path, "smiles", "activity", None, tmp_run_dir, min_valid_compounds=20)
