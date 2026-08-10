"""Persist and reuse identical CV fold assignments across screening experiments."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.model_selection import KFold


def create_cv_folds(
    n_samples: int,
    *,
    n_splits: int = 5,
    random_seed: int = 42,
) -> list[tuple[list[int], list[int]]]:
    """Create deterministic KFold train/val index pairs for agent-dev rows."""
    n_splits = min(n_splits, max(2, n_samples))
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_seed)
    X_dummy = np.zeros((n_samples, 1))
    folds: list[tuple[list[int], list[int]]] = []
    for tr, va in kf.split(X_dummy):
        folds.append(([int(i) for i in tr], [int(i) for i in va]))
    return folds


def hash_cv_folds(folds: list[tuple[list[int], list[int]]]) -> str:
    payload = json.dumps(folds, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def persist_cv_folds(
    run_dir: Path,
    folds: list[tuple[list[int], list[int]]],
    *,
    metadata: dict[str, Any] | None = None,
) -> tuple[str, str]:
    """Write cv_folds.json and return (path, hash)."""
    out_dir = Path(run_dir) / "agent_workspace"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "cv_folds.json"
    fold_hash = hash_cv_folds(folds)
    payload = {
        "folds": [{"train": tr, "val": va} for tr, va in folds],
        "cv_folds_hash": fold_hash,
        "n_splits": len(folds),
        "metadata": metadata or {},
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return str(path), fold_hash


def load_cv_folds(path: str | Path) -> tuple[list[tuple[list[int], list[int]]], str]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    folds = [(list(item["train"]), list(item["val"])) for item in data["folds"]]
    fold_hash = data.get("cv_folds_hash") or hash_cv_folds(folds)
    return folds, fold_hash


def folds_as_sklearn_splits(
    folds: list[tuple[list[int], list[int]]],
) -> list[tuple[np.ndarray, np.ndarray]]:
    return [(np.asarray(tr, dtype=int), np.asarray(va, dtype=int)) for tr, va in folds]
