"""Train/test splitting for the agentic loop.

Three strategies, all producing index arrays into the cleaned dataset:

- ``random``     — plain shuffle split.
- ``stratified`` — preserves class proportions (classification only).
- ``scaffold``   — Bemis-Murcko scaffold split, which puts whole chemical series
  on one side and is the harder, more realistic test of generalisation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from qsar_agent.tools.rdkit_features import murcko_scaffold


def random_split(
    n: int, test_size: float, random_seed: int
) -> tuple[np.ndarray, np.ndarray]:
    indices = np.arange(n)
    train_idx, test_idx = train_test_split(
        indices, test_size=test_size, random_state=random_seed, shuffle=True
    )
    return np.sort(train_idx), np.sort(test_idx)


def stratified_split(
    labels: np.ndarray, test_size: float, random_seed: int
) -> tuple[np.ndarray, np.ndarray]:
    indices = np.arange(len(labels))
    train_idx, test_idx = train_test_split(
        indices,
        test_size=test_size,
        random_state=random_seed,
        shuffle=True,
        stratify=labels,
    )
    return np.sort(train_idx), np.sort(test_idx)


def scaffold_split(
    smiles: list[str], test_size: float, random_seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Assign whole scaffold groups to the test set, largest groups to train.

    Compounds whose scaffold cannot be derived (for example acyclic molecules
    with an empty Murcko scaffold) are grouped by their own SMILES so they are
    still kept together with identical structures rather than silently dropped.
    """
    groups: dict[str, list[int]] = {}
    for index, smi in enumerate(smiles):
        scaffold = murcko_scaffold(smi) or f"__nogroup__{smi}"
        groups.setdefault(scaffold, []).append(index)

    # Largest scaffold groups first keeps the common series in training; the
    # random tiebreak stops the order being an artefact of row order.
    rng = np.random.default_rng(random_seed)
    ordered = sorted(groups.values(), key=lambda members: (-len(members), rng.random()))

    n_total = len(smiles)
    n_test_target = max(1, int(round(test_size * n_total)))
    test_idx: list[int] = []
    train_idx: list[int] = []
    for members in ordered:
        if len(test_idx) < n_test_target and len(members) <= n_test_target - len(test_idx) + 1:
            test_idx.extend(members)
        else:
            train_idx.extend(members)

    if not test_idx:
        # Every group was too large for the target; fall back to the smallest one.
        smallest = min(ordered, key=len)
        test_idx = list(smallest)
        train_idx = [i for i in range(n_total) if i not in set(test_idx)]

    return np.sort(np.asarray(train_idx, dtype=int)), np.sort(np.asarray(test_idx, dtype=int))


def make_split(
    method: str,
    frame: pd.DataFrame,
    task: str,
    test_size: float = 0.2,
    random_seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, str]:
    """Split ``frame`` and return ``(train_idx, test_idx, method_used)``.

    A stratified split is silently promoted to random for regression endpoints,
    and demoted to random when a class has too few members to appear on both
    sides; the returned method name records what actually happened.
    """
    n = len(frame)
    if method == "scaffold":
        train_idx, test_idx = scaffold_split(
            frame["canonical_smiles"].tolist(), test_size, random_seed
        )
        return train_idx, test_idx, "scaffold"

    if method == "stratified":
        if task != "classification":
            train_idx, test_idx = random_split(n, test_size, random_seed)
            return train_idx, test_idx, "random"
        labels = frame["activity"].to_numpy()
        counts = pd.Series(labels).value_counts()
        if counts.min() < 2:
            train_idx, test_idx = random_split(n, test_size, random_seed)
            return train_idx, test_idx, "random"
        train_idx, test_idx = stratified_split(labels, test_size, random_seed)
        return train_idx, test_idx, "stratified"

    train_idx, test_idx = random_split(n, test_size, random_seed)
    return train_idx, test_idx, "random"
