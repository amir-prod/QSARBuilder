"""Shared helpers for descriptor calculation tests."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def fake_descjocky_pipeline(
    *,
    smiles_path,
    mol_dir,
    csv_output,
    backends,
    skip_phase1,
    num_workers,
    xtb_timeout,
) -> None:
    """Write a minimal DescJocky-format CSV without running DescJocky/xtb."""
    smiles = [
        line.strip()
        for line in Path(smiles_path).read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    # Deterministic but weakly correlated features so preprocessing retains enough columns.
    rng = __import__("numpy").random.default_rng(42)
    n = len(smiles)
    matrix = rng.normal(size=(n, 20))
    # Mild signal for activity-like structure without collapsing correlation filters.
    matrix[:, 0] += __import__("numpy").linspace(0, 2, n)

    rows = []
    for i, smi in enumerate(smiles):
        mol_id = f"mol_{i + 1:04d}"
        row = {
            "mol_id": mol_id,
            "smiles": smi,
            "error": "",
        }
        for j in range(matrix.shape[1]):
            row[f"feat_{j}"] = float(matrix[i, j])
        rows.append(row)
    Path(csv_output).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(csv_output, index=False)
