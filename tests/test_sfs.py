"""Tests for sequential feature selection."""

from qsar_agent.tools.dataset_validation import validate_dataset
from qsar_agent.tools.descriptor_preprocessing import fit_descriptor_preprocessor
from qsar_agent.tools.descriptor_calculation import calculate_descriptors
from qsar_agent.tools.sequential_feature_selection import run_sequential_feature_selection
from qsar_agent.tools.umap_split import create_umap_cluster_split
from tests.descriptor_test_utils import fake_descjocky_pipeline


def _preprocessed_splits(synthetic_dataset, tmp_run_dir):
    cleaned = validate_dataset(
        synthetic_dataset, "smiles", "pIC50", "compound_id", tmp_run_dir, min_valid_compounds=15
    )
    descriptors = calculate_descriptors(
        cleaned.cleaned_dataset_path,
        tmp_run_dir,
        pipeline_runner=fake_descjocky_pipeline,
    )
    split = create_umap_cluster_split(descriptors.raw_descriptors_path, tmp_run_dir)
    prep = fit_descriptor_preprocessor(
        split.train_path, split.test_path, tmp_run_dir, val_path=split.val_path
    )
    return prep.preprocessed_train_path, prep.preprocessed_val_path, prep.preprocessed_test_path


def test_sfs_evaluates_full_range(synthetic_dataset, tmp_run_dir):
    train_path, val_path, _test_path = _preprocessed_splits(synthetic_dataset, tmp_run_dir)
    sfs = run_sequential_feature_selection(
        train_path, tmp_run_dir, max_features=5, cv_folds=3, val_path=val_path
    )
    assert len(sfs.results) == 5
    assert sfs.results[0].n_features == 1
    assert sfs.results[-1].n_features == 5
    assert all(row.val_r2 is not None for row in sfs.results)


def test_sfs_cv_not_test(synthetic_dataset, tmp_run_dir):
    train_path, val_path, test_path = _preprocessed_splits(synthetic_dataset, tmp_run_dir)
    sfs = run_sequential_feature_selection(
        train_path, tmp_run_dir, max_features=3, cv_folds=3, val_path=val_path
    )
    for row in sfs.results:
        assert row.mean_cv_r2 <= 1.0
        assert row.val_r2 is not None
    # SFS must not read the external test file.
    import pandas as pd

    test_ids = set(pd.read_csv(test_path)["compound_id"])
    train_ids = set(pd.read_csv(train_path)["compound_id"])
    assert test_ids.isdisjoint(train_ids)
