"""Protected agent-validation carve-out within development data.

This is not full nested CV. Adaptive agentic selection can still overfit the
protected agent-validation set over many cycles; budgets and Validation Agent
warnings mitigate but do not eliminate that risk.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def carve_agent_validation_split(
    development_df: pd.DataFrame,
    *,
    agent_validation_fraction: float = 0.20,
    random_seed: int = 42,
    min_agent_val: int = 3,
    min_agent_dev: int = 8,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Split development-row indices into agent-dev and protected agent-val.

    Returns (agent_dev_indices, agent_val_indices, metadata).
    """
    n = len(development_df)
    if n < min_agent_dev + min_agent_val:
        # Too small: use all for agent-dev; empty protected set (documented limitation)
        idx = np.arange(n)
        meta = {
            "agent_validation_fraction": agent_validation_fraction,
            "agent_dev_size": n,
            "agent_val_size": 0,
            "protected_validation_available": False,
            "limitation": (
                "Dataset too small to carve a protected agent-validation set; "
                "adaptive selection uses development CV only."
            ),
        }
        return idx, np.array([], dtype=int), meta

    rng = np.random.RandomState(random_seed)
    indices = np.arange(n)
    rng.shuffle(indices)
    n_val = max(min_agent_val, int(round(n * agent_validation_fraction)))
    n_val = min(n_val, n - min_agent_dev)
    val_idx = np.sort(indices[:n_val])
    dev_idx = np.sort(indices[n_val:])
    meta = {
        "agent_validation_fraction": agent_validation_fraction,
        "agent_dev_size": int(len(dev_idx)),
        "agent_val_size": int(len(val_idx)),
        "protected_validation_available": True,
        "random_seed": random_seed,
        "limitation": (
            "Protected agent-validation is a single holdout within development data, "
            "not nested CV. Repeated adaptive experimentation can overfit this set."
        ),
    }
    return dev_idx, val_idx, meta


def persist_protected_split(
    run_dir: Path,
    agent_dev_indices: np.ndarray,
    agent_val_indices: np.ndarray,
    metadata: dict[str, Any],
) -> dict[str, str]:
    """Write indices under agent_workspace/protected_split/."""
    out = Path(run_dir) / "agent_workspace" / "protected_split"
    out.mkdir(parents=True, exist_ok=True)
    dev_path = out / "agent_dev_indices.json"
    val_path = out / "agent_val_indices.json"
    meta_path = out / "protected_split_meta.json"
    dev_path.write_text(json.dumps([int(i) for i in agent_dev_indices], indent=2), encoding="utf-8")
    val_path.write_text(json.dumps([int(i) for i in agent_val_indices], indent=2), encoding="utf-8")
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return {
        "agent_dev_indices_path": str(dev_path),
        "agent_val_indices_path": str(val_path),
        "protected_split_meta_path": str(meta_path),
    }


def load_indices(path: str | Path) -> np.ndarray:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return np.asarray(data, dtype=int)


def assert_no_protected_targets_in_training(
    train_indices: np.ndarray | list[int],
    protected_indices: np.ndarray | list[int],
    *,
    context: str,
) -> None:
    """Raise if protected agent-val indices appear in FS/HPO training indices."""
    train_set = set(int(i) for i in train_indices)
    prot_set = set(int(i) for i in protected_indices)
    overlap = train_set & prot_set
    if overlap:
        raise RuntimeError(
            f"{context}: protected agent-validation indices leaked into training "
            f"({len(overlap)} overlapping indices). Adaptive FS/HPO must not use "
            "protected agent-validation targets."
        )


def subset_dataframe(df: pd.DataFrame, indices: np.ndarray) -> pd.DataFrame:
    return df.iloc[list(indices)].reset_index(drop=True)
