"""Tests for UMAP cluster split and activity-sorted split."""

import numpy as np
import pandas as pd

from qsar_agent.tools.dataset_validation import validate_dataset
from qsar_agent.tools.descriptor_calculation import calculate_descriptors
from qsar_agent.tools.umap_split import (
    assign_sorted_split_indices,
    create_sorted_split,
    create_umap_cluster_split,
    split_indices_three_way,
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


def test_no_train_val_test_overlap(synthetic_dataset, tmp_run_dir):
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
    val_ids = set(assignments[assignments["split"] == "val"]["compound_id"])
    test_ids = set(assignments[assignments["split"] == "test"]["compound_id"])
    assert len(train_ids & test_ids) == 0
    assert len(train_ids & val_ids) == 0
    assert len(val_ids & test_ids) == 0
    assert split.test_count >= 1
    assert split.val_count >= 1
    assert split.train_count >= 1
    assert split.train_count + split.val_count + split.test_count == len(assignments)


def test_split_not_empty_when_all_clusters_small(tmp_run_dir):
    """Regression: tiny clusters must not yield empty val or test sets."""
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
    split = create_umap_cluster_split(
        path, tmp_run_dir, test_fraction=0.1, val_fraction=0.1, random_seed=0
    )
    assert split.test_count >= 1
    assert split.val_count >= 1
    assert split.train_count + split.val_count + split.test_count == n
    test_df = pd.read_csv(split.test_path)
    val_df = pd.read_csv(split.val_path)
    assert len(test_df) == split.test_count
    assert len(val_df) == split.val_count


def test_split_indices_three_way_no_overlap():
    idx = list(range(30))
    train, val, test = split_indices_three_way(idx, 0.10, 0.10, random_seed=42)
    assert set(train) & set(val) == set()
    assert set(train) & set(test) == set()
    assert set(val) & set(test) == set()
    assert set(train) | set(val) | set(test) == set(idx)
    assert len(val) >= 1
    assert len(test) >= 1


def test_split_indices_three_way_small_cluster_all_train():
    train, val, test = split_indices_three_way([1, 2, 3, 4, 5], 0.10, 0.10, 0)
    assert train == [1, 2, 3, 4, 5]
    assert val == []
    assert test == []


def test_sorted_split_stride_for_20_percent():
    assert sorted_split_stride(0.20) == 5
    assert sorted_split_stride(0.10) == 10
    assert sorted_split_stride(0.25) == 4


def test_sorted_split_80_10_10_keeps_min_max_in_train():
    activity = np.linspace(1.0, 10.0, 20)
    train_idx, val_idx, test_idx, test_stride, val_stride = assign_sorted_split_indices(
        activity, val_fraction=0.10, test_fraction=0.10
    )
    assert test_stride == 10
    assert val_stride == 10
    # 1-indexed ranks 10, 20 → test; rank 20 is max so it stays in train.
    # ranks 5, 15 → val.
    assert set(test_idx) == {9}
    assert set(val_idx) == {4, 14}
    assert 0 in train_idx
    assert 19 in train_idx
    assert activity[train_idx].min() == activity.min()
    assert activity[train_idx].max() == activity.max()
    assert set(train_idx) | set(val_idx) | set(test_idx) == set(range(20))
    assert set(train_idx) & set(test_idx) == set()
    assert set(val_idx) & set(test_idx) == set()


def test_sorted_split_prefers_test_on_collision():
    activity = np.linspace(1.0, 10.0, 20)
    train_idx, val_idx, test_idx, test_stride, val_stride = assign_sorted_split_indices(
        activity, val_fraction=0.10, test_fraction=0.20
    )
    assert test_stride == 5
    assert val_stride == 10
    # Test every 5th (except protected max); val uses a non-colliding offset.
    assert set(test_idx) == {4, 9, 14}
    assert set(val_idx) & set(test_idx) == set()
    assert 0 in train_idx
    assert 19 in train_idx


def test_sorted_split_shuffled_input_still_ranks_by_activity():
    rng = np.random.default_rng(7)
    activity = rng.permutation(np.linspace(0.0, 1.0, 25))
    train_idx, val_idx, test_idx, test_stride, val_stride = assign_sorted_split_indices(
        activity, val_fraction=0.10, test_fraction=0.10
    )
    assert test_stride == 10
    assert val_stride == 10
    order = np.argsort(activity, kind="mergesort")
    min_idx = int(order[0])
    max_idx = int(order[-1])
    assert min_idx in train_idx
    assert max_idx in train_idx
    expected_test = [
        int(order[rank])
        for rank in range(len(order))
        if (rank + 1) % 10 == 0 and int(order[rank]) not in {min_idx, max_idx}
    ]
    expected_val = [
        int(order[rank])
        for rank in range(len(order))
        if (rank + 1) % 10 == 5 and int(order[rank]) not in {min_idx, max_idx}
    ]
    assert test_idx == expected_test
    assert val_idx == expected_val


def test_sorted_split_tied_min_max_stay_in_train():
    activity = np.array([0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 9.0, 9.0, 9.0])
    train_idx, val_idx, test_idx, _, _ = assign_sorted_split_indices(
        activity, val_fraction=0.10, test_fraction=0.20
    )
    for i, value in enumerate(activity):
        if value in {0.0, 9.0}:
            assert i in train_idx
            assert i not in test_idx
            assert i not in val_idx


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
    split = create_sorted_split(path, tmp_run_dir, test_fraction=0.10, val_fraction=0.10)
    assert split.split_method == "sorted"
    assert split.test_stride == 10
    assert split.val_stride == 10
    assert split.test_count >= 1
    assert split.val_count >= 1
    assert split.train_count + split.val_count + split.test_count == 20
    train_df = pd.read_csv(split.train_path)
    val_df = pd.read_csv(split.val_path)
    test_df = pd.read_csv(split.test_path)
    assert train_df["activity"].min() == activity.min()
    assert train_df["activity"].max() == activity.max()
    assert test_df["activity"].min() > activity.min()
    assert test_df["activity"].max() < activity.max()
    assert val_df["activity"].min() > activity.min()
    assignments = pd.read_csv(split.split_assignments_path)
    assert set(assignments["split"]) == {"train", "val", "test"}
    assert (tmp_run_dir / "sorted_split.png").exists()
    assert (tmp_run_dir / "val_set_raw_descriptors.csv").exists()
