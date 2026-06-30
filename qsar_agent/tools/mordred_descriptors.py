"""Mordred descriptor calculation tool."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from mordred import Calculator, descriptors
from rdkit import Chem
from rdkit import RDLogger

from qsar_agent.config import DescriptorConfig
from qsar_agent.schemas.descriptors import MordredCalculationResult
from qsar_agent.services.artifact_manager import save_json

RDLogger.DisableLog("rdApp.*")

META_COLUMNS = {"compound_id", "canonical_smiles", "activity", "original_row_index"}


def _get_versions() -> tuple[str, str]:
    import mordred

    try:
        from rdkit import __version__ as rdkit_version
    except ImportError:
        rdkit_version = "unknown"
    return mordred.__version__, rdkit_version


def calculate_mordred_descriptors(
    cleaned_dataset_path: str | Path,
    run_dir: Path,
    descriptor_config: DescriptorConfig | None = None,
) -> MordredCalculationResult:
    """Calculate Mordred 2D descriptors for cleaned compounds."""
    cfg = descriptor_config or DescriptorConfig()
    if cfg.enable_3d:
        raise NotImplementedError(
            "3D Mordred descriptors require valid 3D conformers and are disabled by default."
        )

    df = pd.read_csv(cleaned_dataset_path)
    mols = []
    valid_indices = []
    for i, smi in enumerate(df["canonical_smiles"]):
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            mols.append(mol)
            valid_indices.append(i)

    calc = Calculator(descriptors, ignore_3D=True)
    raw_result = calc.pandas(mols)

    desc_df = raw_result.copy()
    failed_records = []

    for col in desc_df.columns:
        numeric_col = pd.to_numeric(desc_df[col], errors="coerce")
        failures = desc_df[col].apply(
            lambda x: isinstance(x, Exception) or (isinstance(x, str) and x == "error")
        )
        if failures.any():
            for idx in desc_df.index[failures]:
                failed_records.append({"descriptor": col, "compound_index": int(idx)})
        desc_df[col] = numeric_col

    desc_df = desc_df.replace([np.inf, -np.inf], np.nan)

    meta = df.iloc[valid_indices][
        ["compound_id", "canonical_smiles", "activity", "original_row_index"]
    ].reset_index(drop=True)
    full_df = pd.concat([meta, desc_df.reset_index(drop=True)], axis=1)

    raw_path = run_dir / "mordred_descriptors_raw.csv"
    full_df.to_csv(raw_path, index=False)

    missing_per_desc = desc_df.isna().sum()
    descriptors_with_missing = int((missing_per_desc > 0).sum())
    failed_count = len(failed_records)

    failed_path = None
    if failed_records:
        failed_path = str(run_dir / "failed_descriptor_values.csv")
        pd.DataFrame(failed_records).to_csv(failed_path, index=False)

    mordred_ver, rdkit_ver = _get_versions()
    report = {
        "compound_count": len(meta),
        "descriptor_count": desc_df.shape[1],
        "descriptors_with_missing": descriptors_with_missing,
        "failed_descriptor_values": failed_count,
        "mordred_version": mordred_ver,
        "rdkit_version": rdkit_ver,
        "enable_3d": cfg.enable_3d,
        "missing_per_descriptor": missing_per_desc.to_dict(),
    }
    report_path = run_dir / "mordred_calculation_report.json"
    save_json(report_path, report)

    return MordredCalculationResult(
        compound_count=len(meta),
        descriptor_count=desc_df.shape[1],
        descriptors_with_missing=descriptors_with_missing,
        failed_descriptor_values=failed_count,
        mordred_version=mordred_ver,
        rdkit_version=rdkit_ver,
        enable_3d=cfg.enable_3d,
        raw_descriptors_path=str(raw_path),
        calculation_report_path=str(report_path),
        failed_values_path=failed_path,
    )
