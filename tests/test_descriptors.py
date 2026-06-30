"""Tests for Mordred descriptor calculation."""

import numpy as np
import pandas as pd

from qsar_agent.tools.mordred_descriptors import calculate_mordred_descriptors
from qsar_agent.tools.dataset_validation import validate_dataset


def test_mordred_calculation(synthetic_dataset, tmp_run_dir):
    cleaned = validate_dataset(
        synthetic_dataset, "smiles", "pIC50", "compound_id", tmp_run_dir, min_valid_compounds=15
    )
    result = calculate_mordred_descriptors(cleaned.cleaned_dataset_path, tmp_run_dir)
    assert result.compound_count >= 15
    assert result.descriptor_count > 0
    raw = pd.read_csv(result.raw_descriptors_path)
    assert "canonical_smiles" in raw.columns
    assert "activity" in raw.columns


def test_infinite_values_replaced(synthetic_dataset, tmp_run_dir):
    cleaned = validate_dataset(
        synthetic_dataset, "smiles", "pIC50", "compound_id", tmp_run_dir, min_valid_compounds=15
    )
    result = calculate_mordred_descriptors(cleaned.cleaned_dataset_path, tmp_run_dir)
    raw = pd.read_csv(result.raw_descriptors_path)
    meta = {"compound_id", "canonical_smiles", "activity", "original_row_index"}
    desc_cols = [c for c in raw.columns if c not in meta]
    values = raw[desc_cols].values.astype(float)
    assert not np.isinf(values).any()
