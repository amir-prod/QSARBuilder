"""Dataset validation tool."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from rdkit import Chem

from qsar_agent.schemas.dataset import ActivityStats, DatasetValidationResult
from qsar_agent.services.artifact_manager import save_json


META_COLUMNS = {"compound_id", "canonical_smiles", "activity", "original_row_index"}


def validate_dataset(
    dataset_path: str | Path,
    smiles_column: str,
    activity_column: str,
    id_column: str | None,
    run_dir: Path,
    min_valid_compounds: int = 20,
) -> DatasetValidationResult:
    """Validate and clean a QSAR dataset without silently discarding records."""
    df = pd.read_csv(dataset_path)
    original_row_count = len(df)
    warnings: list[str] = []

    for col in [smiles_column, activity_column]:
        if col not in df.columns:
            raise ValueError(f"Required column not found: {col}")
    if id_column and id_column not in df.columns:
        raise ValueError(f"ID column not found: {id_column}")

    work = df.copy()
    work["original_row_index"] = work.index

    if id_column:
        work["compound_id"] = work[id_column].astype(str)
    else:
        work["compound_id"] = [f"compound_{i}" for i in range(len(work))]

    work["raw_smiles"] = work[smiles_column].astype(str)
    work["raw_activity"] = work[activity_column]

    numeric_activity = pd.to_numeric(work["raw_activity"], errors="coerce")
    invalid_activity_mask = numeric_activity.isna()
    invalid_activity_count = int(invalid_activity_mask.sum())

    valid_smiles: list[str | None] = []
    invalid_smiles_mask = []
    for smi in work["raw_smiles"]:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            valid_smiles.append(None)
            invalid_smiles_mask.append(True)
        else:
            valid_smiles.append(Chem.MolToSmiles(mol, canonical=True))
            invalid_smiles_mask.append(False)

    work["canonical_smiles"] = valid_smiles
    work["invalid_smiles"] = invalid_smiles_mask
    invalid_smiles_count = int(sum(invalid_smiles_mask))

    invalid_rows_path = None
    invalid_mask = invalid_activity_mask | work["invalid_smiles"]
    if invalid_mask.any():
        invalid_df = work.loc[invalid_mask].copy()
        invalid_rows_path = str(run_dir / "invalid_rows.csv")
        invalid_df.to_csv(invalid_rows_path, index=False)

    valid_mask = ~invalid_mask
    valid_df = work.loc[valid_mask].copy()
    valid_df["activity"] = pd.to_numeric(valid_df["raw_activity"], errors="coerce")

    duplicate_mask = valid_df.duplicated(subset=["canonical_smiles"], keep="first")
    duplicate_count = int(duplicate_mask.sum())
    duplicate_compounds_path = None
    if duplicate_count > 0:
        dup_df = valid_df.loc[duplicate_mask].copy()
        duplicate_compounds_path = str(run_dir / "duplicate_compounds.csv")
        dup_df.to_csv(duplicate_compounds_path, index=False)
        valid_df = valid_df.loc[~duplicate_mask].copy()
        warnings.append(f"Removed {duplicate_count} duplicate SMILES (kept first occurrence).")

    if len(valid_df) < min_valid_compounds:
        raise ValueError(
            f"Too few valid compounds for modeling: {len(valid_df)} "
            f"(minimum required: {min_valid_compounds})."
        )

    cleaned = valid_df[
        ["compound_id", "canonical_smiles", "activity", "original_row_index"]
    ].reset_index(drop=True)

    cleaned_path = run_dir / "cleaned_dataset.csv"
    cleaned.to_csv(cleaned_path, index=False)

    activity_stats = ActivityStats(
        min=float(cleaned["activity"].min()),
        max=float(cleaned["activity"].max()),
        mean=float(cleaned["activity"].mean()),
        median=float(cleaned["activity"].median()),
        std=float(cleaned["activity"].std()),
    )

    report = {
        "original_row_count": original_row_count,
        "valid_compound_count": len(cleaned),
        "invalid_smiles_count": invalid_smiles_count,
        "missing_or_invalid_activity_count": invalid_activity_count,
        "duplicate_compound_count": duplicate_count,
        "activity_stats": activity_stats.model_dump(),
        "smiles_column": smiles_column,
        "activity_column": activity_column,
        "id_column": id_column,
    }
    report_path = run_dir / "dataset_validation.json"
    save_json(report_path, report)

    return DatasetValidationResult(
        original_row_count=original_row_count,
        valid_compound_count=len(cleaned),
        invalid_smiles_count=invalid_smiles_count,
        missing_or_invalid_activity_count=invalid_activity_count,
        duplicate_compound_count=duplicate_count,
        activity_stats=activity_stats,
        cleaned_dataset_path=str(cleaned_path),
        invalid_rows_path=invalid_rows_path,
        duplicate_compounds_path=duplicate_compounds_path,
        validation_report_path=str(report_path),
        warnings=warnings,
    )
