"""Tests for descriptor calculation (DescJocky adapter)."""

import numpy as np
import pandas as pd

from qsar_agent.tools.dataset_validation import validate_dataset
from qsar_agent.tools.descriptor_calculation import calculate_descriptors
from tests.descriptor_test_utils import fake_descjocky_pipeline


def test_descriptor_calculation(synthetic_dataset, tmp_run_dir):
    cleaned = validate_dataset(
        synthetic_dataset, "smiles", "pIC50", "compound_id", tmp_run_dir, min_valid_compounds=15
    )
    result = calculate_descriptors(
        cleaned.cleaned_dataset_path,
        tmp_run_dir,
        pipeline_runner=fake_descjocky_pipeline,
    )
    assert result.compound_count >= 15
    assert result.descriptor_count > 0
    raw = pd.read_csv(result.raw_descriptors_path)
    assert "canonical_smiles" in raw.columns
    assert "activity" in raw.columns


def test_infinite_values_replaced(synthetic_dataset, tmp_run_dir):
    cleaned = validate_dataset(
        synthetic_dataset, "smiles", "pIC50", "compound_id", tmp_run_dir, min_valid_compounds=15
    )
    result = calculate_descriptors(
        cleaned.cleaned_dataset_path,
        tmp_run_dir,
        pipeline_runner=fake_descjocky_pipeline,
    )
    raw = pd.read_csv(result.raw_descriptors_path)
    meta = {"compound_id", "canonical_smiles", "activity", "original_row_index"}
    desc_cols = [c for c in raw.columns if c not in meta]
    values = raw[desc_cols].values.astype(float)
    assert not np.isinf(values).any()
