"""Tests for sequential feature selection."""

from qsar_agent.config import GAConfig, SFSConfig
from qsar_agent.tools.dataset_validation import validate_dataset
from qsar_agent.tools.descriptor_preprocessing import fit_descriptor_preprocessor
from qsar_agent.tools.mordred_descriptors import calculate_mordred_descriptors
from qsar_agent.tools.sequential_feature_selection import run_sequential_feature_selection
from qsar_agent.tools.umap_split import create_umap_cluster_split


def _preprocessed_train(synthetic_dataset, tmp_run_dir):
    cleaned = validate_dataset(
        synthetic_dataset, "smiles", "pIC50", "compound_id", tmp_run_dir, min_valid_compounds=15
    )
    mordred = calculate_mordred_descriptors(cleaned.cleaned_dataset_path, tmp_run_dir)
    split = create_umap_cluster_split(mordred.raw_descriptors_path, tmp_run_dir)
    prep = fit_descriptor_preprocessor(split.train_path, split.test_path, tmp_run_dir)
    return prep.preprocessed_train_path


def test_sfs_evaluates_full_range(synthetic_dataset, tmp_run_dir):
    train_path = _preprocessed_train(synthetic_dataset, tmp_run_dir)
    sfs = run_sequential_feature_selection(
        train_path, tmp_run_dir, max_features=5, cv_folds=3
    )
    assert len(sfs.results) == 5
    assert sfs.results[0].n_features == 1
    assert sfs.results[-1].n_features == 5


def test_sfs_cv_not_test(synthetic_dataset, tmp_run_dir):
    train_path = _preprocessed_train(synthetic_dataset, tmp_run_dir)
    sfs = run_sequential_feature_selection(train_path, tmp_run_dir, max_features=3, cv_folds=3)
    for row in sfs.results:
        assert row.mean_cv_r2 <= 1.0
