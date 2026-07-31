"""DescJocky-powered molecular descriptor calculation and external merge."""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit import RDLogger
from rdkit.Chem import AllChem

from qsar_agent.config import DescriptorConfig
from qsar_agent.schemas.descriptors import BackendInfo, DescriptorCalculationResult
from qsar_agent.services.artifact_manager import save_json

RDLogger.DisableLog("rdApp.*")
logger = logging.getLogger(__name__)

META_COLUMNS = {"compound_id", "canonical_smiles", "activity", "original_row_index"}
META_COLUMN_ORDER = ["compound_id", "canonical_smiles", "activity", "original_row_index"]
DESCJOCKY_META = {"mol_id", "smiles", "error"}

# Fallback metadata when DescJocky registry is unavailable (e.g. unit tests).
_BACKEND_3D_FALLBACK: dict[str, bool] = {
    "RDKit": False,
    "Mordred": True,
    "Native": True,
    "Pybel": False,
}


def _rdkit_version() -> str:
    try:
        from rdkit import __version__ as rdkit_version

        return str(rdkit_version)
    except Exception:
        return "unknown"


def _descjocky_version() -> str:
    try:
        import descjocky

        return getattr(descjocky, "__version__", "unknown")
    except Exception:
        return "unavailable"


def _mol_id_for_index(i: int) -> str:
    return f"mol_{i + 1:04d}"


def write_smiles_and_id_map(
    cleaned_df: pd.DataFrame,
    descjocky_dir: Path,
) -> tuple[Path, dict[str, dict[str, Any]]]:
    """Write one-SMILES-per-line file and mol_id → compound meta map."""
    descjocky_dir.mkdir(parents=True, exist_ok=True)
    smiles_path = descjocky_dir / "smiles.txt"
    mol_id_map: dict[str, dict[str, Any]] = {}
    lines: list[str] = []
    for pos, (_, row) in enumerate(cleaned_df.iterrows()):
        mol_id = _mol_id_for_index(pos)
        smi = str(row["canonical_smiles"])
        lines.append(smi)
        mol_id_map[mol_id] = {
            "compound_id": row["compound_id"],
            "canonical_smiles": smi,
            "activity": row["activity"],
            "original_row_index": int(row["original_row_index"]),
            "row_pos": pos,
        }

    smiles_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    map_path = descjocky_dir / "mol_id_map.json"
    save_json(map_path, mol_id_map)
    return smiles_path, mol_id_map


def write_light_sdfs(
    cleaned_df: pd.DataFrame,
    mol_dir: Path,
) -> list[str]:
    """Write RDKit SDFs (no xtb) so DescJocky can run with skip_phase1=True."""
    opt_dir = mol_dir / "optimized"
    opt_dir.mkdir(parents=True, exist_ok=True)
    warnings: list[str] = []

    for pos, (_, row) in enumerate(cleaned_df.iterrows()):
        mol_id = _mol_id_for_index(pos)
        smi = str(row["canonical_smiles"])
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            warnings.append(f"Could not parse SMILES for {mol_id} ({smi}); skipped SDF.")
            continue
        mol = Chem.AddHs(mol)
        try:
            params = AllChem.ETKDGv3()
            params.randomSeed = 42
            status = AllChem.EmbedMolecule(mol, params)
            if status != 0:
                # Fall back to 2D coordinates if embedding fails.
                AllChem.Compute2DCoords(mol)
                warnings.append(
                    f"3D embed failed for {mol_id}; wrote 2D coordinates instead."
                )
            else:
                try:
                    AllChem.MMFFOptimizeMolecule(mol, maxIters=200)
                except Exception:
                    pass
        except Exception as exc:
            AllChem.Compute2DCoords(mol)
            warnings.append(f"Geometry generation failed for {mol_id}: {exc}")

        sdf_path = opt_dir / f"{mol_id}.sdf"
        writer = Chem.SDWriter(str(sdf_path))
        writer.write(mol)
        writer.close()

    return warnings


def merge_external_descriptors(
    generated_df: pd.DataFrame,
    external_path: str | Path,
) -> tuple[pd.DataFrame, list[str], int]:
    """
    Left-join external descriptors onto generated rows by compound_id.

    Returns (merged_df, warnings, n_external_feature_cols).
    """
    warnings: list[str] = []
    external_path = Path(external_path)
    if not external_path.exists():
        raise FileNotFoundError(f"External descriptors file not found: {external_path}")

    ext = pd.read_csv(external_path)
    if "compound_id" not in ext.columns:
        raise ValueError("External descriptors CSV must include a 'compound_id' column.")

    ext["compound_id"] = ext["compound_id"].astype(str)
    generated = generated_df.copy()
    generated["compound_id"] = generated["compound_id"].astype(str)

    # Drop external meta collisions (except join key).
    drop_meta = [c for c in ext.columns if c in META_COLUMNS and c != "compound_id"]
    if drop_meta:
        ext = ext.drop(columns=drop_meta)
        warnings.append(
            f"Dropped external columns colliding with meta: {', '.join(drop_meta)}"
        )

    gen_features = [c for c in generated.columns if c not in META_COLUMNS]
    rename_map: dict[str, str] = {}
    for col in ext.columns:
        if col == "compound_id":
            continue
        if col in gen_features or col in generated.columns:
            rename_map[col] = f"ext__{col}"
    if rename_map:
        ext = ext.rename(columns=rename_map)
        warnings.append(
            "Renamed colliding external descriptor columns with 'ext__' prefix: "
            + ", ".join(f"{k}->{v}" for k, v in rename_map.items())
        )

    feature_cols = [c for c in ext.columns if c != "compound_id"]
    for col in feature_cols:
        ext[col] = pd.to_numeric(ext[col], errors="coerce")
    ext[feature_cols] = ext[feature_cols].replace([np.inf, -np.inf], np.nan)

    # Deduplicate external IDs (keep first).
    if ext["compound_id"].duplicated().any():
        n_dup = int(ext["compound_id"].duplicated().sum())
        ext = ext.drop_duplicates(subset=["compound_id"], keep="first")
        warnings.append(f"Dropped {n_dup} duplicate external compound_id row(s).")

    gen_ids = set(generated["compound_id"])
    ext_ids = set(ext["compound_id"])
    unmatched_ext = sorted(ext_ids - gen_ids)
    missing_ext = sorted(gen_ids - ext_ids)
    if unmatched_ext:
        warnings.append(
            f"{len(unmatched_ext)} external compound_id(s) not in generated set "
            f"(examples: {', '.join(unmatched_ext[:5])})."
        )
    if missing_ext:
        warnings.append(
            f"{len(missing_ext)} generated compound(s) have no external descriptors "
            f"(examples: {', '.join(missing_ext[:5])})."
        )

    merged = generated.merge(ext, on="compound_id", how="left")
    return merged, warnings, len(feature_cols)


def _align_descjocky_csv(
    descjocky_csv: Path,
    mol_id_map: dict[str, dict[str, Any]],
    cleaned_df: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str]]:
    """Build meta + numeric descriptor frame aligned to cleaned compounds."""
    warnings: list[str] = []
    raw = pd.read_csv(descjocky_csv)
    if "mol_id" not in raw.columns:
        raise ValueError("DescJocky output CSV missing required 'mol_id' column.")

    if "error" in raw.columns:
        errored = raw["error"].fillna("").astype(str).str.len() > 0
        if errored.any():
            bad_ids = raw.loc[errored, "mol_id"].astype(str).tolist()
            warnings.append(
                f"DescJocky reported errors for {len(bad_ids)} molecule(s) "
                f"(examples: {', '.join(bad_ids[:5])})."
            )

    feature_cols = [c for c in raw.columns if c not in DESCJOCKY_META]
    for col in feature_cols:
        raw[col] = pd.to_numeric(raw[col], errors="coerce")
    if feature_cols:
        raw[feature_cols] = raw[feature_cols].replace([np.inf, -np.inf], np.nan)

    rows: list[dict[str, Any]] = []
    desc_by_mol = raw.set_index("mol_id", drop=False)
    for pos, (_, crow) in enumerate(cleaned_df.iterrows()):
        mol_id = _mol_id_for_index(pos)
        meta = {
            "compound_id": crow["compound_id"],
            "canonical_smiles": crow["canonical_smiles"],
            "activity": crow["activity"],
            "original_row_index": int(crow["original_row_index"]),
        }
        if mol_id not in desc_by_mol.index:
            warnings.append(f"No DescJocky row for {mol_id}; descriptors set to NaN.")
            feat = {c: np.nan for c in feature_cols}
        else:
            rec = desc_by_mol.loc[mol_id]
            if isinstance(rec, pd.DataFrame):
                rec = rec.iloc[0]
            feat = {c: rec[c] for c in feature_cols}
        rows.append({**meta, **feat})

    out = pd.DataFrame(rows)
    # Preserve meta column order then features
    ordered = META_COLUMN_ORDER + [c for c in out.columns if c not in META_COLUMNS]
    return out[ordered], warnings


def resolve_backend_details(backends: list[str]) -> list[dict[str, Any]]:
    """Resolve which backends are available and whether they support 3D descriptors."""
    details: list[dict[str, Any]] = []
    registry = None
    try:
        from descjocky.core.registry import BackendRegistry

        registry = BackendRegistry
    except Exception:
        registry = None

    for name in backends:
        supports_3d = _BACKEND_3D_FALLBACK.get(name, False)
        available = name in _BACKEND_3D_FALLBACK
        if registry is not None:
            try:
                cls = registry.get(name)
                available = bool(cls.available())
                supports_3d = bool(getattr(cls, "supports_3d", False))
            except Exception:
                available = False
        details.append(
            {
                "name": name,
                "available": available,
                "supports_3d": supports_3d,
                "used": available,
            }
        )
    return details


def _write_descriptor_report_md(path: Path, report: dict[str, Any]) -> None:
    backends = report.get("backends_detail", [])
    backend_lines = []
    for b in backends:
        status = "used" if b.get("used") else "requested but unavailable"
        d3 = "yes" if b.get("supports_3d") else "no"
        backend_lines.append(f"- **{b.get('name')}** ({status}); supports_3d={d3}")

    cols = report.get("generated_descriptor_columns", [])
    col_preview = ", ".join(cols[:40])
    if len(cols) > 40:
        col_preview += f", … ({len(cols) - 40} more)"

    md = "\n".join(
        [
            "# Descriptor Calculation Report",
            "",
            f"- **Compounds:** {report.get('compound_count')}",
            f"- **Generated descriptors:** {report.get('generated_descriptor_count')}",
            f"- **External descriptors merged:** {report.get('external_descriptor_count')}",
            f"- **Total descriptors (combined):** {report.get('descriptor_count')}",
            "",
            "## Backends calculated",
            *(backend_lines or ["- (none)"]),
            "",
            "## 3D status",
            f"- **Geometry source:** {report.get('geometry_source')}",
            f"- **xtb geometry optimization:** {report.get('run_geometry_optimization')}",
            f"- **3D geometries used:** {report.get('three_d_geometries_used')}",
            f"- **3D descriptors included:** {report.get('three_d_descriptors_included')}",
            "",
            "3D descriptors are marked included only when **Run geometry optimization (xtb)** "
            "is enabled and at least one used backend declares `supports_3d=True` "
            "(e.g. Mordred, Native).",
            "",
            "## Generated descriptor columns",
            f"Count: {len(cols)}",
            "",
            col_preview or "(none)",
            "",
            "## Artifacts",
            f"- Generated CSV: `{report.get('generated_descriptors_path', '')}`",
            f"- Combined CSV: `{report.get('raw_descriptors_path', '')}`",
            "",
            "## Warnings",
        ]
    )
    warnings = report.get("warnings") or []
    if warnings:
        md += "\n" + "\n".join(f"- {w}" for w in warnings) + "\n"
    else:
        md += "\n- None\n"
    path.write_text(md, encoding="utf-8")


def _run_descjocky_pipeline(
    *,
    smiles_path: Path,
    mol_dir: Path,
    csv_output: Path,
    backends: list[str],
    skip_phase1: bool,
    num_workers: int,
    xtb_timeout: int,
) -> None:
    from descjocky import Pipeline

    if not skip_phase1 and shutil.which("xtb") is None:
        raise RuntimeError(
            "Geometry optimization is enabled but 'xtb' was not found on PATH. "
            "Install xtb or disable run_geometry_optimization."
        )

    Pipeline(
        {
            "input_file": str(smiles_path),
            "mol_dir": str(mol_dir),
            "csv_output": str(csv_output),
            "num_workers": num_workers,
            "xtb_timeout": xtb_timeout,
            "remove_temp": False,
            "backends": backends,
            "skip_phase1": skip_phase1,
        }
    ).run()


def calculate_descriptors(
    cleaned_dataset_path: str | Path,
    run_dir: Path,
    descriptor_config: DescriptorConfig | None = None,
    pipeline_runner=_run_descjocky_pipeline,
) -> DescriptorCalculationResult:
    """
    Calculate molecular descriptors with DescJocky and optionally merge external CSV.

    ``pipeline_runner`` is injectable for unit tests.
    """
    cfg = descriptor_config or DescriptorConfig()
    run_dir = Path(run_dir)
    cleaned_df = pd.read_csv(cleaned_dataset_path)
    required = {"compound_id", "canonical_smiles", "activity", "original_row_index"}
    missing = required - set(cleaned_df.columns)
    if missing:
        raise ValueError(f"Cleaned dataset missing columns: {sorted(missing)}")

    warnings: list[str] = []
    descjocky_dir = run_dir / "descjocky"
    mol_dir = descjocky_dir / "mols"
    smiles_path, mol_id_map = write_smiles_and_id_map(cleaned_df, descjocky_dir)

    skip_phase1 = not cfg.run_geometry_optimization
    if skip_phase1:
        warnings.extend(write_light_sdfs(cleaned_df, mol_dir))
    else:
        mol_dir.mkdir(parents=True, exist_ok=True)

    dj_csv = descjocky_dir / "descriptors.csv"
    backends = list(cfg.backends) if cfg.backends else ["RDKit", "Mordred"]
    pipeline_runner(
        smiles_path=smiles_path,
        mol_dir=mol_dir,
        csv_output=dj_csv,
        backends=backends,
        skip_phase1=skip_phase1,
        num_workers=cfg.num_workers,
        xtb_timeout=cfg.xtb_timeout,
    )

    if not dj_csv.exists():
        raise RuntimeError(f"DescJocky did not produce output CSV at {dj_csv}")

    generated_df, align_warnings = _align_descjocky_csv(dj_csv, mol_id_map, cleaned_df)
    warnings.extend(align_warnings)

    # Final generated-only CSV (primary artifact) + raw-named copy for compatibility.
    generated_path = run_dir / "generated_descriptors.csv"
    generated_raw_path = run_dir / "generated_descriptors_raw.csv"
    generated_df.to_csv(generated_path, index=False)
    generated_df.to_csv(generated_raw_path, index=False)
    generated_cols = [c for c in generated_df.columns if c not in META_COLUMNS]
    generated_count = len(generated_cols)

    # Also keep DescJocky's native CSV under a stable report-facing name.
    descjocky_final = run_dir / "descjocky_descriptors.csv"
    if dj_csv.exists() and dj_csv.resolve() != descjocky_final.resolve():
        shutil.copy2(dj_csv, descjocky_final)

    external_count = 0
    external_copy_path: str | None = None
    combined = generated_df
    if cfg.external_descriptors_path:
        src = Path(cfg.external_descriptors_path)
        external_copy = run_dir / "external_descriptors.csv"
        if src.resolve() != external_copy.resolve():
            shutil.copy2(src, external_copy)
        external_copy_path = str(external_copy)
        combined, merge_warnings, external_count = merge_external_descriptors(
            generated_df, external_copy
        )
        warnings.extend(merge_warnings)

    raw_path = run_dir / "descriptors_raw.csv"
    combined.to_csv(raw_path, index=False)

    feature_cols = [c for c in combined.columns if c not in META_COLUMNS]
    descriptors_with_missing = int(combined[feature_cols].isna().any().sum()) if feature_cols else 0

    backends_detail = resolve_backend_details(backends)
    used_backends = [b for b in backends_detail if b.get("used")]
    # 3D is reported only when the user enabled xtb geometry optimization.
    geometry_source = (
        "xtb" if cfg.run_geometry_optimization else "rdkit_light_sdf_no_xtb"
    )
    three_d_geometries_used = bool(cfg.run_geometry_optimization)
    three_d_descriptors_included = bool(cfg.run_geometry_optimization) and any(
        b.get("supports_3d") for b in used_backends
    )

    report = {
        "compound_count": len(combined),
        "descriptor_count": len(feature_cols),
        "generated_descriptor_count": generated_count,
        "external_descriptor_count": external_count,
        "descriptors_with_missing": descriptors_with_missing,
        "backends": backends,
        "backends_requested": backends,
        "backends_detail": backends_detail,
        "backends_used": [b["name"] for b in used_backends],
        "run_geometry_optimization": cfg.run_geometry_optimization,
        "geometry_source": geometry_source,
        "three_d_geometries_used": three_d_geometries_used,
        "three_d_descriptors_included": three_d_descriptors_included,
        "generated_descriptor_columns": generated_cols,
        "generated_descriptors_path": str(generated_path),
        "generated_descriptors_raw_path": str(generated_raw_path),
        "descjocky_native_csv_path": str(descjocky_final),
        "raw_descriptors_path": str(raw_path),
        "rdkit_version": _rdkit_version(),
        "descjocky_version": _descjocky_version(),
        "warnings": warnings,
    }
    report_path = run_dir / "descriptor_calculation_report.json"
    report_md_path = run_dir / "descriptor_calculation_report.md"
    save_json(report_path, report)
    _write_descriptor_report_md(report_md_path, report)
    save_json(run_dir / "generated_descriptor_columns.json", {"columns": generated_cols})

    return DescriptorCalculationResult(
        compound_count=len(combined),
        descriptor_count=len(feature_cols),
        generated_descriptor_count=generated_count,
        external_descriptor_count=external_count,
        descriptors_with_missing=descriptors_with_missing,
        backends=backends,
        backends_detail=[BackendInfo(**b) for b in backends_detail],
        run_geometry_optimization=cfg.run_geometry_optimization,
        geometry_source=geometry_source,
        three_d_geometries_used=three_d_geometries_used,
        three_d_descriptors_included=three_d_descriptors_included,
        generated_descriptor_columns=generated_cols,
        rdkit_version=_rdkit_version(),
        descjocky_version=_descjocky_version(),
        raw_descriptors_path=str(raw_path),
        generated_descriptors_path=str(generated_path),
        calculation_report_path=str(report_path),
        calculation_report_md_path=str(report_md_path),
        external_descriptors_path=external_copy_path,
        warnings=warnings,
    )


# Backward-compatible aliases
def calculate_mordred_descriptors(
    cleaned_dataset_path: str | Path,
    run_dir: Path,
    descriptor_config: DescriptorConfig | None = None,
):
    """Deprecated alias — use calculate_descriptors."""
    return calculate_descriptors(cleaned_dataset_path, run_dir, descriptor_config)
