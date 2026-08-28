"""Persistent-outlier aggregation from out-of-fold residuals (never auto-deletes)."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable

import numpy as np
import pandas as pd

from qsar_agent.schemas.agentic import ExclusionProposal, PersistentOutlierReport


def _zscores(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    std = np.nanstd(values)
    if std == 0 or not np.isfinite(std):
        return np.zeros_like(values)
    return (values - np.nanmean(values)) / std


def persistent_outliers_from_oof(
    oof_tables: Iterable[pd.DataFrame],
    *,
    structural_flags: dict[str, float] | None = None,
    model_family: str = "",
    residual_z_threshold: float = 3.0,
) -> list[PersistentOutlierReport]:
    """Aggregate OOF residual-outlier frequency across tables.

    ``oof_tables`` must contain ``compound_id`` and ``residual`` (or activity/prediction).
    """
    hits: dict[str, int] = defaultdict(int)
    seen: dict[str, int] = defaultdict(int)
    families: dict[str, set[str]] = defaultdict(set)
    structural_flags = structural_flags or {}

    for table in oof_tables:
        df = table.copy()
        if "residual" not in df.columns:
            df["residual"] = df["activity"] - df["predicted_activity"]
        z = np.abs(_zscores(df["residual"].to_numpy()))
        df = df.assign(_z=z)
        for _, row in df.iterrows():
            cid = str(row["compound_id"])
            seen[cid] += 1
            if float(row["_z"]) > residual_z_threshold:
                hits[cid] += 1
                if model_family:
                    families[cid].add(model_family)

    reports: list[PersistentOutlierReport] = []
    for cid, n_seen in seen.items():
        oof_freq = hits[cid] / n_seen if n_seen else 0.0
        struct_freq = float(structural_flags.get(cid, 0.0))
        if oof_freq <= 0 and struct_freq <= 0:
            continue
        if oof_freq >= 0.7 and struct_freq >= 0.5:
            action = "propose_exclusion"
        elif oof_freq >= 0.5:
            action = "audit"
        elif struct_freq >= 0.5:
            action = "restrict_domain"
        else:
            action = "retain"
        reports.append(
            PersistentOutlierReport(
                compound_id=cid,
                oof_response_outlier_frequency=oof_freq,
                structural_outlier_frequency=struct_freq,
                model_families_flagging=sorted(families.get(cid, set()) or ({model_family} if model_family and oof_freq else [])),
                possible_data_quality_issue=None,
                recommended_action=action,  # type: ignore[arg-type]
            )
        )
    reports.sort(key=lambda r: r.oof_response_outlier_frequency, reverse=True)
    return reports


def proposal_from_report(report: PersistentOutlierReport, *, case: str = "difficult_valid") -> ExclusionProposal:
    return ExclusionProposal(
        compound_id=report.compound_id,
        proposed_reason=report.recommended_action,
        evidence=[
            f"OOF residual outlier frequency={report.oof_response_outlier_frequency:.3f}",
            f"structural outlier frequency={report.structural_outlier_frequency:.3f}",
        ],
        source_of_verification="oof_residuals_and_leverage",
        models_and_runs_flagging=list(report.model_families_flagging),
        oof_residual_frequency=report.oof_response_outlier_frequency,
        structural_outlier_frequency=report.structural_outlier_frequency,
        expected_scientific_effect="Unknown until sensitivity analysis on approved exclusions.",
        required_approval=True,
        case=case,  # type: ignore[arg-type]
    )


def mask_compounds(df: pd.DataFrame, compound_ids: list[str]) -> pd.DataFrame:
    """Return a copy without the listed IDs. Does not write the curated dataset."""
    return df[~df["compound_id"].astype(str).isin({str(c) for c in compound_ids})].copy()
