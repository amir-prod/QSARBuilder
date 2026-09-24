"""Generate the sample datasets used to demo the agentic workflow.

Builds a combinatorial library from real scaffolds and substituents, then assigns
activities from a deliberately **nonlinear** function of RDKit descriptors plus a
substructure term and noise. A linear model therefore underfits while tree
ensembles succeed, which is what makes the iteration loop visible in the demo.

Run from the repository root:

    python scripts/generate_sample_datasets.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors

RDLogger.DisableLog("rdApp.*")

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "example"
RANDOM_SEED = 20240917

SCAFFOLDS = [
    "c1ccccc1{R}",
    "c1ccc(cc1){R}",
    "c1ccc2ccccc2c1{R}",
    "c1ccncc1{R}",
    "c1cc2ccccc2cn1{R}",
    "C1CCCCC1{R}",
    "C1CCNCC1{R}",
    "c1cc(sc1){R}",
    "c1cc(oc1){R}",
    "c1cnc2ccccc2n1{R}",
    "C1COCCN1{R}",
    "c1ccc(cc1)C(=O){R}",
    "c1ccc(cc1)S(=O)(=O){R}",
    "c1ccc(cc1)Oc1ccccc1{R}",
    "C1=CC(=O)NC1{R}",
]

SUBSTITUENTS = [
    "C",
    "CC",
    "CCC",
    "C(C)C",
    "CCCC",
    "O",
    "OC",
    "OCC",
    "N",
    "NC",
    "N(C)C",
    "Cl",
    "Br",
    "F",
    "C(=O)O",
    "C(=O)N",
    "C#N",
    "S(=O)(=O)N",
    "CO",
    "CCO",
    "CN",
    "c1ccccc1",
    "C(F)(F)F",
    "N1CCOCC1",
]


def build_library() -> list[str]:
    """Enumerate scaffold/substituent combinations, keeping valid unique molecules."""
    seen: set[str] = set()
    smiles: list[str] = []
    for scaffold in SCAFFOLDS:
        for substituent in SUBSTITUENTS:
            candidate = scaffold.replace("{R}", substituent)
            mol = Chem.MolFromSmiles(candidate)
            if mol is None:
                continue
            canonical = Chem.MolToSmiles(mol)
            if canonical in seen:
                continue
            seen.add(canonical)
            smiles.append(canonical)
    return smiles


def descriptor_frame(smiles: list[str]) -> pd.DataFrame:
    rows = []
    for smi in smiles:
        mol = Chem.MolFromSmiles(smi)
        rows.append(
            {
                "logp": Descriptors.MolLogP(mol),
                "tpsa": Descriptors.TPSA(mol),
                "mw": Descriptors.MolWt(mol),
                "hbd": Descriptors.NumHDonors(mol),
                "aromatic_rings": Descriptors.NumAromaticRings(mol),
                "rotatable": Descriptors.NumRotatableBonds(mol),
                "has_halogen": float(
                    any(atom.GetSymbol() in {"F", "Cl", "Br", "I"} for atom in mol.GetAtoms())
                ),
                "has_carbonyl": float(mol.HasSubstructMatch(Chem.MolFromSmarts("C=O"))),
            }
        )
    return pd.DataFrame(rows)


def latent_activity(features: pd.DataFrame, rng: np.random.Generator) -> np.ndarray:
    """A nonlinear structure-activity relationship with interactions and noise.

    Every term here is chosen to be invisible to a linear model on descriptors: a
    narrow Gaussian optimum in logP, a halogen effect that only applies to
    polycyclic compounds, and a step change above a polarity cutoff. Tree
    ensembles recover them, which is what makes the iteration loop meaningful.
    """
    logp = features["logp"].to_numpy()
    tpsa = features["tpsa"].to_numpy()
    aromatic = features["aromatic_rings"].to_numpy()
    halogen = features["has_halogen"].to_numpy()
    carbonyl = features["has_carbonyl"].to_numpy()
    rotatable = features["rotatable"].to_numpy()

    lipophilic_optimum = 2.2 * np.exp(-(((logp - 2.6) / 0.9) ** 2))
    conditional_halogen = 0.9 * halogen * (aromatic >= 2)
    polarity_step = -1.1 * (tpsa > 55).astype(float)
    flexibility_penalty = -0.18 * np.minimum(rotatable, 6)
    carbonyl_term = -0.5 * carbonyl * (1.0 - 0.5 * halogen)

    signal = (
        5.5
        + lipophilic_optimum
        + conditional_halogen
        + polarity_step
        + flexibility_penalty
        + carbonyl_term
    )
    noise = rng.normal(0.0, 0.30, size=len(signal))
    return signal + noise


def write_regression(smiles: list[str], activity: np.ndarray) -> Path:
    frame = pd.DataFrame(
        {
            "compound_id": [f"CMP{i:04d}" for i in range(len(smiles))],
            "smiles": smiles,
            "pIC50": np.round(activity, 3),
        }
    )
    path = OUTPUT_DIR / "agentic_regression_sample.csv"
    frame.to_csv(path, index=False)
    return path


def write_classification(smiles: list[str], activity: np.ndarray) -> Path:
    # Split at the 60th percentile so the classes are unbalanced but learnable.
    threshold = float(np.quantile(activity, 0.60))
    frame = pd.DataFrame(
        {
            "compound_id": [f"CMP{i:04d}" for i in range(len(smiles))],
            "smiles": smiles,
            "active": (activity >= threshold).astype(int),
        }
    )
    path = OUTPUT_DIR / "agentic_classification_sample.csv"
    frame.to_csv(path, index=False)
    return path


def main() -> None:
    rng = np.random.default_rng(RANDOM_SEED)
    smiles = build_library()
    features = descriptor_frame(smiles)
    activity = latent_activity(features, rng)

    regression_path = write_regression(smiles, activity)
    classification_path = write_classification(smiles, activity)

    print(f"{len(smiles)} unique compounds generated.")
    print(f"regression:     {regression_path}")
    print(
        f"  pIC50 range {activity.min():.2f} to {activity.max():.2f}, "
        f"mean {activity.mean():.2f}, std {activity.std():.2f}"
    )
    print(f"classification: {classification_path}")
    counts = pd.read_csv(classification_path)["active"].value_counts().to_dict()
    print(f"  class counts {counts}")


if __name__ == "__main__":
    main()
