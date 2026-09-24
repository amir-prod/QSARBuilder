"""RDKit descriptor and fingerprint blocks for the agentic loop.

Deliberately fast: a full block computes in seconds for a few hundred
compounds, which is what makes a multi-iteration loop practical. The existing
DescJocky/Mordred path stays available for the heavyweight Streamlit workflow.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Descriptors, MACCSkeys, rdFingerprintGenerator

from qsar_agent.schemas.agentic import FeatureRecipe

RDLogger.DisableLog("rdApp.*")

#: Feature blocks the Strategist may select from.
AVAILABLE_BLOCKS: dict[str, str] = {
    "rdkit_descriptors": (
        "~210 interpretable physicochemical and topological RDKit descriptors "
        "(MolWt, LogP, TPSA, rings, counts, connectivity indices)."
    ),
    "morgan_fp": "Binary Morgan (ECFP-like) circular fingerprint bits.",
    "morgan_counts": "Count-based Morgan fingerprint (substructure occurrence counts).",
    "maccs_keys": "167 MACCS structural keys.",
    "rdkit_fp": "RDKit path-based topological fingerprint bits.",
    "atom_pair_fp": "Hashed atom-pair fingerprint bits.",
}

#: Descriptors that are frequently non-finite or constant and add no signal.
_EXCLUDED_DESCRIPTORS = {"Ipc", "MaxPartialCharge", "MinPartialCharge",
                         "MaxAbsPartialCharge", "MinAbsPartialCharge"}


def _descriptor_functions() -> list[tuple[str, object]]:
    return [
        (name, func)
        for name, func in Descriptors.descList
        if name not in _EXCLUDED_DESCRIPTORS
    ]


def mols_from_smiles(smiles: list[str]) -> list[Chem.Mol]:
    """Parse SMILES, raising on the first unparseable entry.

    Callers reach this only after dataset validation, so an invalid structure
    here is a programming error rather than a data problem.
    """
    mols = []
    for index, smi in enumerate(smiles):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            raise ValueError(f"Unparseable SMILES at position {index}: {smi!r}")
        mols.append(mol)
    return mols


def compute_rdkit_descriptors(mols: list[Chem.Mol]) -> pd.DataFrame:
    functions = _descriptor_functions()
    rows = []
    for mol in mols:
        values = []
        for _name, func in functions:
            try:
                value = func(mol)
            except Exception:
                value = np.nan
            values.append(float(value) if value is not None else np.nan)
        rows.append(values)
    frame = pd.DataFrame(rows, columns=[f"rdkit_{name}" for name, _ in functions])
    return frame.replace([np.inf, -np.inf], np.nan)


def compute_morgan(
    mols: list[Chem.Mol], n_bits: int, radius: int, counts: bool = False
) -> pd.DataFrame:
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
    prefix = "mgc" if counts else "mg"
    rows = []
    for mol in mols:
        fp = (
            generator.GetCountFingerprintAsNumPy(mol)
            if counts
            else generator.GetFingerprintAsNumPy(mol)
        )
        rows.append(np.asarray(fp, dtype=float))
    return pd.DataFrame(rows, columns=[f"{prefix}{radius}_{i}" for i in range(n_bits)])


def compute_maccs(mols: list[Chem.Mol]) -> pd.DataFrame:
    rows = [np.asarray(MACCSkeys.GenMACCSKeys(mol), dtype=float) for mol in mols]
    n_bits = len(rows[0]) if rows else 167
    return pd.DataFrame(rows, columns=[f"maccs_{i}" for i in range(n_bits)])


def compute_rdkit_fp(mols: list[Chem.Mol], n_bits: int) -> pd.DataFrame:
    generator = rdFingerprintGenerator.GetRDKitFPGenerator(fpSize=n_bits)
    rows = [np.asarray(generator.GetFingerprintAsNumPy(mol), dtype=float) for mol in mols]
    return pd.DataFrame(rows, columns=[f"rdkfp_{i}" for i in range(n_bits)])


def compute_atom_pair_fp(mols: list[Chem.Mol], n_bits: int) -> pd.DataFrame:
    generator = rdFingerprintGenerator.GetAtomPairGenerator(fpSize=n_bits)
    rows = [np.asarray(generator.GetFingerprintAsNumPy(mol), dtype=float) for mol in mols]
    return pd.DataFrame(rows, columns=[f"ap_{i}" for i in range(n_bits)])


def compute_feature_blocks(
    smiles: list[str],
    blocks: list[str],
    fingerprint_bits: int = 1024,
    fingerprint_radius: int = 2,
) -> pd.DataFrame:
    """Concatenate the requested feature blocks into one matrix."""
    unknown = [b for b in blocks if b not in AVAILABLE_BLOCKS]
    if unknown:
        raise ValueError(
            f"Unknown feature block(s): {unknown}. Available: {sorted(AVAILABLE_BLOCKS)}"
        )
    mols = mols_from_smiles(smiles)
    frames: list[pd.DataFrame] = []
    for block in blocks:
        if block == "rdkit_descriptors":
            frames.append(compute_rdkit_descriptors(mols))
        elif block == "morgan_fp":
            frames.append(compute_morgan(mols, fingerprint_bits, fingerprint_radius))
        elif block == "morgan_counts":
            frames.append(
                compute_morgan(mols, fingerprint_bits, fingerprint_radius, counts=True)
            )
        elif block == "maccs_keys":
            frames.append(compute_maccs(mols))
        elif block == "rdkit_fp":
            frames.append(compute_rdkit_fp(mols, fingerprint_bits))
        elif block == "atom_pair_fp":
            frames.append(compute_atom_pair_fp(mols, fingerprint_bits))
    combined = pd.concat(frames, axis=1)
    combined.index = range(len(smiles))
    return combined


def compute_features(smiles: list[str], recipe: FeatureRecipe) -> pd.DataFrame:
    """Compute the feature matrix described by ``recipe`` (before fitting filters)."""
    return compute_feature_blocks(
        smiles,
        blocks=list(recipe.blocks),
        fingerprint_bits=recipe.fingerprint_bits,
        fingerprint_radius=recipe.fingerprint_radius,
    )


def murcko_scaffold(smiles: str) -> str:
    """Bemis-Murcko scaffold SMILES, empty string when it cannot be derived."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return ""
    try:
        from rdkit.Chem.Scaffolds import MurckoScaffold

        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
        return Chem.MolToSmiles(scaffold) if scaffold is not None else ""
    except Exception:
        return ""


def morgan_fingerprint_matrix(
    smiles: list[str], n_bits: int = 1024, radius: int = 2
) -> np.ndarray:
    """Convenience binary fingerprint matrix, used for similarity-based diagnostics."""
    _ = AllChem  # kept imported for downstream users of this module
    mols = mols_from_smiles(smiles)
    return compute_morgan(mols, n_bits, radius).to_numpy(dtype=float)
