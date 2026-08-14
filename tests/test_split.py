"""Tests for UMAP cluster split and activity-sorted split."""

import numpy as np
import pandas as pd

from qsar_agent.tools.dataset_validation import validate_dataset
from qsar_agent.tools.descriptor_calculation import calculate_descriptors
from qsar_agent.tools.umap_split import (
    assign_sorted_split_indices,
    create_sorted_split,
    create_umap_cluster_split,
    sorted_split_stride,
)
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


def test_sorted_split_stride_for_20_percent():
    assert sorted_split_stride(0.20) == 5
    assert sorted_split_stride(0.10) == 10
    assert sorted_split_stride(0.25) == 4


def test_sorted_split_every_fifth_keeps_min_max_in_train():
    activity = np.linspace(1.0, 10.0, 20)
    train_idx, test_idx, stride = assign_sorted_split_indices(activity, test_fraction=0.20)
    assert stride == 5
    # 1-indexed ranks 5, 10, 15, 20; rank 20 is max so it stays in train.
    assert set(test_idx) == {4, 9, 14}
    assert 0 in train_idx
    assert 19 in train_idx
    assert activity[train_idx].min() == activity.min()
    assert activity[train_idx].max() == activity.max()
    assert set(train_idx) | set(test_idx) == set(range(20))
    assert set(train_idx) & set(test_idx) == set()


def test_sorted_split_shuffled_input_still_ranks_by_activity():
    rng = np.random.default_rng(7)
    activity = rng.permutation(np.linspace(0.0, 1.0, 25))
    train_idx, test_idx, stride = assign_sorted_split_indices(activity, test_fraction=0.20)
    assert stride == 5
    order = np.argsort(activity, kind="mergesort")
    min_idx = int(order[0])
    max_idx = int(order[-1])
    assert min_idx in train_idx
    assert max_idx in train_idx
    # Every 5th in sorted order, excluding protected min/max.
    expected_test = [
        int(order[rank])
        for rank in range(len(order))
        if (rank + 1) % 5 == 0 and int(order[rank]) not in {min_idx, max_idx}
    ]
    assert test_idx == expected_test


def test_sorted_split_tied_min_max_stay_in_train():
    activity = np.array([0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 9.0, 9.0, 9.0])
    train_idx, test_idx, _ = assign_sorted_split_indices(activity, test_fraction=0.20)
    for i, value in enumerate(activity):
        if value in {0.0, 9.0}:
            assert i in train_idx
            assert i not in test_idx


def _write_descriptor_table(path, activity) -> None:
    n = len(activity)
    rows = {
        "compound_id": [f"C{i}" for i in range(n)],
        "canonical_smiles": ["CCO"] * n,
        "activity": list(activity),
        "original_row_index": list(range(n)),
        "feat_0": np.arange(n, dtype=float),
    }
    pd.DataFrame(rows).to_csv(path, index=False)


def test_create_sorted_split_artifacts(tmp_run_dir):
    activity = np.linspace(1.0, 10.0, 20)
    path = tmp_run_dir / "raw.csv"
    _write_descriptor_table(path, activity)
    split = create_sorted_split(path, tmp_run_dir, test_fraction=0.20)
    assert split.split_method == "sorted"
    assert split.test_stride == 5
    assert split.test_count == 3
    assert split.train_count == 17
    train_df = pd.read_csv(split.train_path)
    test_df = pd.read_csv(split.test_path)
    assert train_df["activity"].min() == activity.min()
    assert train_df["activity"].max() == activity.max()
    assert test_df["activity"].min() > activity.min()
    assert test_df["activity"].max() < activity.max()
    assignments = pd.read_csv(split.split_assignments_path)
    assert set(assignments["split"]) == {"train", "test"}
    assert (tmp_run_dir / "sorted_split.png").exists()
