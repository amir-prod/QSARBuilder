"""Tests for descriptor preprocessing."""

import numpy as np
import pandas as pd

from qsar_agent.config import PreprocessingConfig
from qsar_agent.tools.descriptor_preprocessing import fit_descriptor_preprocessor
from qsar_agent.tools.descriptor_calculation import META_COLUMNS


def _make_train_val_test(n_features=10, n_train=30, n_val=8, n_test=10):
    cols = [f"desc_{i}" for i in range(n_features)]
    meta_train = pd.DataFrame({
        "compound_id": [f"t{i}" for i in range(n_train)],
        "canonical_smiles": ["CCO"] * n_train,
        "activity": np.random.randn(n_train),
        "original_row_index": range(n_train),
    })
    meta_val = pd.DataFrame({
        "compound_id": [f"v{i}" for i in range(n_val)],
        "canonical_smiles": ["CCN"] * n_val,
        "activity": np.random.randn(n_val),
        "original_row_index": range(n_val),
    })
    meta_test = pd.DataFrame({
        "compound_id": [f"e{i}" for i in range(n_test)],
        "canonical_smiles": ["CCC"] * n_test,
        "activity": np.random.randn(n_test),
        "original_row_index": range(n_test),
    })
    X_train = pd.DataFrame(np.random.randn(n_train, n_features), columns=cols)
    X_val = pd.DataFrame(np.random.randn(n_val, n_features), columns=cols)
    X_test = pd.DataFrame(np.random.randn(n_test, n_features), columns=cols)
    X_train.iloc[:, 0] = 5.0  # constant
    X_train.iloc[:, 1] = 5.0 + np.random.randn(n_train) * 0.001  # near-constant
    X_train.iloc[:, 2] = X_train.iloc[:, 3] + np.random.randn(n_train) * 0.01  # correlated
    X_val.iloc[:, 0] = 5.0
    X_test.iloc[:, 0] = 5.0
    train = pd.concat([meta_train, X_train], axis=1)
    val = pd.concat([meta_val, X_val], axis=1)
    test = pd.concat([meta_test, X_test], axis=1)
    return train, val, test


def _write_splits(tmp_run_dir, n_features=10):
    train, val, test = _make_train_val_test(n_features=n_features)
    train_path = tmp_run_dir / "train.csv"
    val_path = tmp_run_dir / "val.csv"
    test_path = tmp_run_dir / "test.csv"
    train.to_csv(train_path, index=False)
    val.to_csv(val_path, index=False)
    test.to_csv(test_path, index=False)
    return train_path, val_path, test_path


def test_constant_and_near_constant_removal(tmp_run_dir):
    train_path, val_path, test_path = _write_splits(tmp_run_dir)
    result = fit_descriptor_preprocessor(train_path, test_path, tmp_run_dir, val_path=val_path)
    assert result.removed_constant >= 1
    assert result.removed_near_constant >= 1


def test_scaler_fitted_on_train_only(tmp_run_dir):
    train_path, val_path, test_path = _write_splits(tmp_run_dir, n_features=5)
    result = fit_descriptor_preprocessor(train_path, test_path, tmp_run_dir, val_path=val_path)
    train_pp = pd.read_csv(result.preprocessed_train_path)
    val_pp = pd.read_csv(result.preprocessed_val_path)
    test_pp = pd.read_csv(result.preprocessed_test_path)
    desc = [c for c in train_pp.columns if c not in META_COLUMNS]
    assert abs(train_pp[desc].std().mean() - 1.0) < 0.2
    assert list(val_pp.columns) == list(train_pp.columns)
    assert list(test_pp.columns) == list(train_pp.columns)


def test_correlation_removal(tmp_run_dir):
    train_path, val_path, test_path = _write_splits(tmp_run_dir)
    result = fit_descriptor_preprocessor(
        train_path, test_path, tmp_run_dir,
        PreprocessingConfig(correlation_threshold=0.95),
        val_path=val_path,
    )
    assert result.removed_correlated >= 1
