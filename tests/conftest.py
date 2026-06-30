"""Pytest configuration and shared fixtures."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd
import pytest


@pytest.fixture
def tmp_run_dir(tmp_path):
    return tmp_path


@pytest.fixture
def synthetic_dataset(tmp_path) -> Path:
    """Small synthetic QSAR dataset for testing."""
    smiles = [
        "CCO", "CC(C)O", "c1ccccc1", "CC(=O)O", "CCN", "CCC", "CCCC", "CC(C)C",
        "c1ccc(O)cc1", "c1ccc(N)cc1", "C1CCCCC1", "CCOC", "CC(C)OC", "CCCOC",
        "CC(=O)OC", "CNC", "CCNC", "c1ccncc1", "c1ccc(Cl)cc1", "c1ccc(Br)cc1",
        "CCCl", "CCBr", "CCI", "CCCCCC", "CCCCCCC", "CC(C)(C)O", "CC(C)CC",
        "c1ccc(C)cc1", "c1ccc(CO)cc1", "c1ccccc1O",
    ]
    activity = [i * 0.5 + (i % 3) * 0.1 for i in range(len(smiles))]
    df = pd.DataFrame({
        "compound_id": [f"C{i:03d}" for i in range(len(smiles))],
        "smiles": smiles,
        "pIC50": activity,
    })
    path = tmp_path / "synthetic.csv"
    df.to_csv(path, index=False)
    return path
