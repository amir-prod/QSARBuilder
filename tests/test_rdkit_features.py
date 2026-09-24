"""Tests for the RDKit descriptor and fingerprint blocks."""

from __future__ import annotations

import numpy as np
import pytest

from qsar_agent.schemas.agentic import FeatureRecipe
from qsar_agent.tools.rdkit_features import (
    AVAILABLE_BLOCKS,
    compute_feature_blocks,
    compute_features,
    mols_from_smiles,
    murcko_scaffold,
)

SMILES = ["CCO", "c1ccccc1", "CC(=O)O", "c1ccc(Cl)cc1", "CCN(CC)CC"]


def test_descriptor_block_has_one_row_per_molecule():
    frame = compute_feature_blocks(SMILES, ["rdkit_descriptors"])
    assert len(frame) == len(SMILES)
    assert frame.shape[1] > 100
    assert all(name.startswith("rdkit_") for name in frame.columns)


def test_descriptor_values_are_finite_or_missing_never_infinite():
    frame = compute_feature_blocks(SMILES, ["rdkit_descriptors"])
    assert not np.isinf(frame.to_numpy(dtype=float)).any()


def test_morgan_block_respects_requested_bit_count():
    frame = compute_feature_blocks(SMILES, ["morgan_fp"], fingerprint_bits=256)
    assert frame.shape == (len(SMILES), 256)
    assert set(np.unique(frame.to_numpy())) <= {0.0, 1.0}


def test_morgan_counts_can_exceed_one():
    frame = compute_feature_blocks(
        ["CCCCCCCCCC"], ["morgan_counts"], fingerprint_bits=256
    )
    assert frame.to_numpy().max() > 1.0


def test_maccs_block_has_167_keys():
    frame = compute_feature_blocks(SMILES, ["maccs_keys"])
    assert frame.shape == (len(SMILES), 167)


def test_blocks_concatenate_in_order():
    frame = compute_feature_blocks(
        SMILES, ["rdkit_descriptors", "morgan_fp"], fingerprint_bits=64
    )
    descriptor_only = compute_feature_blocks(SMILES, ["rdkit_descriptors"])
    assert frame.shape[1] == descriptor_only.shape[1] + 64
    assert list(frame.columns[: descriptor_only.shape[1]]) == list(descriptor_only.columns)


def test_duplicate_blocks_are_collapsed_by_the_recipe():
    recipe = FeatureRecipe(blocks=["morgan_fp", "morgan_fp"], fingerprint_bits=32)
    assert recipe.blocks == ["morgan_fp"]
    assert compute_features(SMILES, recipe).shape == (len(SMILES), 32)


@pytest.mark.parametrize("block", sorted(AVAILABLE_BLOCKS))
def test_every_advertised_block_computes(block):
    frame = compute_feature_blocks(SMILES, [block], fingerprint_bits=64)
    assert len(frame) == len(SMILES)
    assert frame.shape[1] > 0


def test_unknown_block_is_rejected():
    with pytest.raises(ValueError, match="Unknown feature block"):
        compute_feature_blocks(SMILES, ["quantum_magic"])


def test_invalid_smiles_names_its_position():
    with pytest.raises(ValueError, match="position 1"):
        mols_from_smiles(["CCO", "not_a_molecule"])


def test_murcko_scaffold_strips_substituents():
    assert murcko_scaffold("c1ccc(CCN)cc1") == "c1ccccc1"


def test_murcko_scaffold_of_acyclic_molecule_is_empty():
    assert murcko_scaffold("CCCC") == ""


def test_murcko_scaffold_of_invalid_smiles_is_empty():
    assert murcko_scaffold("@@@") == ""


def test_recipe_signature_distinguishes_configurations():
    first = FeatureRecipe(blocks=["morgan_fp"], fingerprint_bits=1024)
    second = FeatureRecipe(blocks=["morgan_fp"], fingerprint_bits=2048)
    assert first.signature() != second.signature()
