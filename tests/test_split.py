"""Tests for UMAP cluster split."""

import pandas as pd

from qsar_agent.tools.dataset_validation import validate_dataset
from qsar_agent.tools.mordred_descriptors import calculate_mordred_descriptors
from qsar_agent.tools.umap_split import create_umap_cluster_split


def test_reproducible_split(synthetic_dataset, tmp_run_dir):
    cleaned = validate_dataset(
        synthetic_dataset, "smiles", "pIC50", "compound_id", tmp_run_dir, min_valid_compounds=15
    )
    mordred = calculate_mordred_descriptors(cleaned.cleaned_dataset_path, tmp_run_dir)
    split1 = create_umap_cluster_split(
        mordred.raw_descriptors_path, tmp_run_dir / "run1", random_seed=42
    )
    (tmp_run_dir / "run2").mkdir(exist_ok=True)
    split2 = create_umap_cluster_split(
        mordred.raw_descriptors_path, tmp_run_dir / "run2", random_seed=42
    )
    a1 = pd.read_csv(split1.split_assignments_path).sort_values("compound_id")
    a2 = pd.read_csv(split2.split_assignments_path).sort_values("compound_id")
    assert a1["split"].tolist() == a2["split"].tolist()


def test_no_train_test_overlap(synthetic_dataset, tmp_run_dir):
    cleaned = validate_dataset(
        synthetic_dataset, "smiles", "pIC50", "compound_id", tmp_run_dir, min_valid_compounds=15
    )
    mordred = calculate_mordred_descriptors(cleaned.cleaned_dataset_path, tmp_run_dir)
    split = create_umap_cluster_split(mordred.raw_descriptors_path, tmp_run_dir)
    assignments = pd.read_csv(split.split_assignments_path)
    train_ids = set(assignments[assignments["split"] == "train"]["compound_id"])
    test_ids = set(assignments[assignments["split"] == "test"]["compound_id"])
    assert len(train_ids & test_ids) == 0
