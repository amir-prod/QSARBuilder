"""Tests for the random, stratified and scaffold split strategies."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from qsar_agent.tools.splitting import (
    make_split,
    random_split,
    scaffold_split,
    stratified_split,
)


def frame_from(smiles, activities):
    return pd.DataFrame({"canonical_smiles": smiles, "activity": activities})


SMILES = [
    "c1ccccc1C", "c1ccccc1CC", "c1ccccc1CCC", "c1ccccc1O", "c1ccccc1N",
    "C1CCCCC1C", "C1CCCCC1CC", "C1CCCCC1O",
    "c1ccncc1C", "c1ccncc1CC",
    "CCCC", "CCCCC", "CCCCCC",
]


# -- random --------------------------------------------------------------------


def test_random_split_partitions_every_index():
    train, test = random_split(100, test_size=0.2, random_seed=0)
    assert len(train) == 80
    assert len(test) == 20
    assert set(train).isdisjoint(test)
    assert sorted([*train, *test]) == list(range(100))


def test_random_split_is_reproducible():
    first = random_split(50, 0.2, 7)
    second = random_split(50, 0.2, 7)
    assert np.array_equal(first[0], second[0])


def test_a_different_seed_gives_a_different_split():
    first = random_split(100, 0.2, 1)
    second = random_split(100, 0.2, 2)
    assert not np.array_equal(first[1], second[1])


# -- stratified ----------------------------------------------------------------


def test_stratified_split_preserves_class_proportions():
    labels = np.array([0] * 80 + [1] * 20)
    train, test = stratified_split(labels, test_size=0.2, random_seed=0)
    assert labels[test].mean() == pytest.approx(0.20, abs=0.05)
    assert labels[train].mean() == pytest.approx(0.20, abs=0.05)


# -- scaffold ------------------------------------------------------------------


def test_scaffold_split_keeps_a_scaffold_on_one_side():
    train, test = scaffold_split(SMILES, test_size=0.3, random_seed=0)
    from qsar_agent.tools.rdkit_features import murcko_scaffold

    train_scaffolds = {murcko_scaffold(SMILES[i]) for i in train}
    test_scaffolds = {murcko_scaffold(SMILES[i]) for i in test}
    shared = {s for s in train_scaffolds & test_scaffolds if s}
    assert not shared


def test_scaffold_split_covers_every_compound():
    train, test = scaffold_split(SMILES, test_size=0.3, random_seed=0)
    assert sorted([*train, *test]) == list(range(len(SMILES)))


def test_scaffold_split_always_yields_a_test_set():
    # A single shared scaffold is the pathological case for group splitting.
    identical = ["c1ccccc1C"] * 10
    train, test = scaffold_split(identical, test_size=0.2, random_seed=0)
    assert len(test) >= 1
    assert len(train) + len(test) == 10


def test_scaffold_split_is_reproducible():
    first = scaffold_split(SMILES, 0.3, 5)
    second = scaffold_split(SMILES, 0.3, 5)
    assert np.array_equal(first[1], second[1])


# -- dispatch ------------------------------------------------------------------


def test_make_split_reports_the_method_used():
    frame = frame_from(SMILES, list(range(len(SMILES))))
    _train, _test, method = make_split("random", frame, "regression", 0.3, 0)
    assert method == "random"


def test_stratified_is_demoted_to_random_for_regression():
    frame = frame_from(SMILES, [float(i) for i in range(len(SMILES))])
    _train, _test, method = make_split("stratified", frame, "regression", 0.3, 0)
    assert method == "random"


def test_stratified_is_used_for_classification():
    labels = ["0"] * 8 + ["1"] * 5
    frame = frame_from(SMILES, labels)
    _train, test, method = make_split("stratified", frame, "classification", 0.3, 0)
    assert method == "stratified"
    assert len(test) >= 1


def test_stratified_falls_back_when_a_class_is_too_rare():
    # One singleton class cannot appear on both sides of a stratified split.
    labels = ["0"] * 12 + ["1"]
    frame = frame_from(SMILES, labels)
    _train, _test, method = make_split("stratified", frame, "classification", 0.3, 0)
    assert method == "random"


def test_scaffold_is_dispatched():
    frame = frame_from(SMILES, list(range(len(SMILES))))
    _train, _test, method = make_split("scaffold", frame, "regression", 0.3, 0)
    assert method == "scaffold"
