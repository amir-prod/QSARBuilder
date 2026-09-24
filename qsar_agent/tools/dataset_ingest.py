"""Dataset ingestion for the agentic loop (regression and classification).

Distinct from :mod:`qsar_agent.tools.dataset_validation`, which assumes a
continuous endpoint and coerces the activity column to numeric. This ingester
keeps categorical labels intact so classification datasets survive, and resolves
duplicate structures by averaging (regression) or majority vote (classification)
rather than discarding them.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from pydantic import BaseModel, Field
from rdkit import Chem, RDLogger

from qsar_agent.schemas.agentic import TaskDetection
from qsar_agent.services.artifact_manager import save_json
from qsar_agent.tools.task_detection import detect_task

RDLogger.DisableLog("rdApp.*")

MIN_COMPOUNDS = 20


class IngestResult(BaseModel):
    """Cleaned dataset plus an account of everything that was changed."""

    cleaned_dataset_path: str
    n_input_rows: int
    n_compounds: int
    n_invalid_smiles: int
    n_missing_activity: int
    n_duplicate_structures: int
    task_detection: TaskDetection
    smiles_column: str
    activity_column: str
    id_column: str | None = None
    duplicate_resolution: str = ""
    invalid_rows_path: str | None = None
    report_path: str = ""
    warnings: list[str] = Field(default_factory=list)


def ingest_dataset(
    dataset_path: str | Path,
    smiles_column: str,
    activity_column: str,
    run_dir: Path,
    id_column: str | None = None,
    forced_task: str | None = None,
    min_compounds: int = MIN_COMPOUNDS,
) -> IngestResult:
    """Read, canonicalise, deduplicate and characterise a QSAR dataset."""
    frame = pd.read_csv(dataset_path)
    n_input_rows = len(frame)
    warnings: list[str] = []

    for column in (smiles_column, activity_column):
        if column not in frame.columns:
            raise ValueError(
                f"Column {column!r} not found. Available columns: {list(frame.columns)}"
            )
    if id_column and id_column not in frame.columns:
        raise ValueError(f"ID column {id_column!r} not found.")

    work = frame.copy()
    work["original_row_index"] = range(len(work))
    work["compound_id"] = (
        work[id_column].astype(str) if id_column else [f"cmp_{i:05d}" for i in range(len(work))]
    )

    canonical: list[str | None] = []
    for raw in work[smiles_column].astype(str):
        mol = Chem.MolFromSmiles(raw)
        canonical.append(Chem.MolToSmiles(mol, canonical=True) if mol is not None else None)
    work["canonical_smiles"] = canonical

    invalid_smiles = work["canonical_smiles"].isna()
    missing_activity = work[activity_column].isna()
    bad = invalid_smiles | missing_activity

    invalid_rows_path = None
    if bad.any():
        invalid_rows_path = str(run_dir / "agentic_invalid_rows.csv")
        work.loc[bad].to_csv(invalid_rows_path, index=False)
        warnings.append(
            f"Excluded {int(bad.sum())} rows: {int(invalid_smiles.sum())} unparseable SMILES, "
            f"{int(missing_activity.sum())} missing activities. See {invalid_rows_path}."
        )

    clean = work.loc[~bad].copy()
    if clean.empty:
        raise ValueError("No usable rows remain after removing invalid SMILES and activities.")

    detection = detect_task(clean[activity_column], forced=forced_task)

    if detection.task == "regression":
        clean["activity"] = pd.to_numeric(clean[activity_column], errors="coerce")
        clean = clean.dropna(subset=["activity"])
    else:
        clean["activity"] = clean[activity_column].astype(str)

    n_before_dedup = len(clean)
    duplicated = clean["canonical_smiles"].duplicated(keep=False)
    n_duplicate_structures = int(clean.loc[duplicated, "canonical_smiles"].nunique())

    if duplicated.any():
        if detection.task == "regression":
            resolution = "averaged activities of duplicate structures"
            aggregated = (
                clean.groupby("canonical_smiles", as_index=False)
                .agg(
                    compound_id=("compound_id", "first"),
                    activity=("activity", "mean"),
                    original_row_index=("original_row_index", "first"),
                )
            )
        else:
            resolution = "kept the majority label for duplicate structures"
            aggregated = (
                clean.groupby("canonical_smiles", as_index=False)
                .agg(
                    compound_id=("compound_id", "first"),
                    activity=("activity", lambda s: s.mode().iat[0]),
                    original_row_index=("original_row_index", "first"),
                )
            )
        clean = aggregated
        warnings.append(
            f"Collapsed {n_before_dedup - len(clean)} duplicate rows covering "
            f"{n_duplicate_structures} structures; {resolution}."
        )
    else:
        resolution = "no duplicate structures found"
        clean = clean[["canonical_smiles", "compound_id", "activity", "original_row_index"]]

    clean = clean.sort_values("original_row_index").reset_index(drop=True)

    if len(clean) < min_compounds:
        raise ValueError(
            f"Only {len(clean)} usable compounds; at least {min_compounds} are needed for a "
            "train/test split with cross-validation."
        )

    # Re-detect on the deduplicated activities so class counts match the model's view.
    detection = detect_task(clean["activity"], forced=forced_task)

    cleaned_path = run_dir / "agentic_cleaned_dataset.csv"
    clean.to_csv(cleaned_path, index=False)

    report_path = run_dir / "agentic_ingest_report.json"
    save_json(
        report_path,
        {
            "n_input_rows": n_input_rows,
            "n_compounds": len(clean),
            "n_invalid_smiles": int(invalid_smiles.sum()),
            "n_missing_activity": int(missing_activity.sum()),
            "n_duplicate_structures": n_duplicate_structures,
            "duplicate_resolution": resolution,
            "task_detection": detection.model_dump(),
            "warnings": warnings,
        },
    )

    return IngestResult(
        cleaned_dataset_path=str(cleaned_path),
        n_input_rows=n_input_rows,
        n_compounds=len(clean),
        n_invalid_smiles=int(invalid_smiles.sum()),
        n_missing_activity=int(missing_activity.sum()),
        n_duplicate_structures=n_duplicate_structures,
        task_detection=detection,
        smiles_column=smiles_column,
        activity_column=activity_column,
        id_column=id_column,
        duplicate_resolution=resolution,
        invalid_rows_path=invalid_rows_path,
        report_path=str(report_path),
        warnings=warnings,
    )
