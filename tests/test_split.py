"""Tests for UMAP cluster split."""

import pandas as pd

from qsar_agent.tools.dataset_validation import validate_dataset
from qsar_agent.tools.descriptor_calculation import calculate_descriptors
from qsar_agent.tools.umap_split import create_umap_cluster_split
from tests.descriptor_test_utils import fake_descjocky_pipeline


def test_reproducible_split(synthetic_dataset, tmp_run_dir):
    cleaned = validate_dataset(
        synthetic_dataset, "smiles", "pIC50", "compound_id", tmp_run_dir, min_valid_compounds=15
    )
    descriptors = calculate_descriptors(
        cleaned.cleaned_dataset_path,
        tmp_run_dir,
        pipeline_runner=fake_descjocky_pipeline,
    )
    split1 = create_umap_cluster_split(
        descriptors.raw_descriptors_path, tmp_run_dir / "run1", random_seed=42
    )
    (tmp_run_dir / "run2").mkdir(exist_ok=True)
    split2 = create_umap_cluster_split(
        descriptors.raw_descriptors_path, tmp_run_dir / "run2", random_seed=42
    )
    a1 = pd.read_csv(split1.split_assignments_path).sort_values("compound_id")
    a2 = pd.read_csv(split2.split_assignments_path).sort_values("compound_id")
    assert a1["split"].tolist() == a2["split"].tolist()


def test_no_train_test_overlap(synthetic_dataset, tmp_run_dir):
    cleaned = validate_dataset(
        synthetic_dataset, "smiles", "pIC50", "compound_id", tmp_run_dir, min_valid_compounds=15
    )
    descriptors = calculate_descriptors(
        cleaned.cleaned_dataset_path,
        tmp_run_dir,
        pipeline_runner=fake_descjocky_pipeline,
    )
    split = create_umap_cluster_split(descriptors.raw_descriptors_path, tmp_run_dir)
    assignments = pd.read_csv(split.split_assignments_path)
    train_ids = set(assignments[assignments["split"] == "train"]["compound_id"])
    test_ids = set(assignments[assignments["split"] == "test"]["compound_id"])
    assert len(train_ids & test_ids) == 0
    assert split.test_count >= 1
    assert split.train_count >= 1


def test_split_not_empty_when_all_clusters_small(tmp_run_dir):
    """Regression: tiny clusters must not yield an empty external test set."""
    import numpy as np

    n = 20
    rows = {
        "compound_id": [f"C{i}" for i in range(n)],
        "canonical_smiles": ["CCO"] * n,
        "activity": np.linspace(1, 5, n),
        "original_row_index": list(range(n)),
    }
    for j in range(10):
        rows[f"feat_{j}"] = np.random.default_rng(0).normal(size=n)
    path = tmp_run_dir / "raw.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    split = create_umap_cluster_split(path, tmp_run_dir, test_fraction=0.2, random_seed=0)
    assert split.test_count >= 1
    assert split.train_count + split.test_count == n
    test_df = pd.read_csv(split.test_path)
    assert len(test_df) == split.test_count
