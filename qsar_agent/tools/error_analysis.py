"""Error analysis for the modeling handoff (winner predictions + AD flags)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from qsar_agent.schemas.handoff import (
    DomainPerformance,
    ErrorAnalysis,
    LargestErrorCompound,
    RangePerformance,
    ResidualDiagnostics,
)

_TOP_N_ERRORS = 10


def _safe_regression_metrics(y_true, y_pred) -> tuple[float | None, float | None, float | None]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    n = len(y_true)
    if n == 0:
        return None, None, None
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    if n < 2 or np.allclose(y_true, y_true[0]):
        return None, rmse, mae
    return float(r2_score(y_true, y_pred)), rmse, mae


def _domain_performance(df: pd.DataFrame) -> DomainPerformance:
    if df.empty:
        return DomainPerformance(n=0)
    r2, rmse, mae = _safe_regression_metrics(df["activity"], df["predicted_activity"])
    return DomainPerformance(n=int(len(df)), r2=r2, rmse=rmse, mae=mae)


def analyze_errors(
    predictions_path: str | Path,
    ad_classifications_path: str | Path | None,
    winner_run_id: str,
    top_n: int = _TOP_N_ERRORS,
) -> ErrorAnalysis:
    pred = pd.read_csv(predictions_path)
    if "residual" not in pred.columns:
        pred["residual"] = pred["activity"] - pred["predicted_activity"]
    pred["abs_residual"] = pred["residual"].abs()

    ad_col = "applicability_domain"
    in_domain_col = "in_domain"
    if ad_classifications_path and Path(ad_classifications_path).exists():
        ad = pd.read_csv(ad_classifications_path)
        keep = ["compound_id"]
        if ad_col in ad.columns:
            keep.append(ad_col)
        if in_domain_col in ad.columns:
            keep.append(in_domain_col)
        pred = pred.merge(ad[keep], on="compound_id", how="left")
    if ad_col not in pred.columns:
        pred[ad_col] = ""
    if in_domain_col not in pred.columns:
        pred[in_domain_col] = True

    largest: list[LargestErrorCompound] = []
    ranked = pred.sort_values("abs_residual", ascending=False).head(top_n)
    for _, row in ranked.iterrows():
        largest.append(
            LargestErrorCompound(
                compound_id=str(row["compound_id"]),
                split=str(row.get("split", "")),
                activity=float(row["activity"]),
                predicted_activity=float(row["predicted_activity"]),
                residual=float(row["residual"]),
                abs_residual=float(row["abs_residual"]),
                applicability_domain=str(row.get(ad_col, "") or ""),
            )
        )

    range_rows: list[RangePerformance] = []
    if len(pred) >= 3:
        try:
            pred = pred.copy()
            pred["_tertile"] = pd.qcut(
                pred["activity"], q=3, labels=["low", "mid", "high"], duplicates="drop"
            )
            for label, grp in pred.groupby("_tertile", observed=True):
                r2, rmse, mae = _safe_regression_metrics(
                    grp["activity"], grp["predicted_activity"]
                )
                range_rows.append(
                    RangePerformance(
                        range_label=str(label),
                        n=int(len(grp)),
                        r2=r2,
                        rmse=rmse,
                        mae=mae,
                        activity_min=float(grp["activity"].min()),
                        activity_max=float(grp["activity"].max()),
                    )
                )
        except ValueError:
            r2, rmse, mae = _safe_regression_metrics(pred["activity"], pred["predicted_activity"])
            range_rows.append(
                RangePerformance(
                    range_label="all",
                    n=int(len(pred)),
                    r2=r2,
                    rmse=rmse,
                    mae=mae,
                    activity_min=float(pred["activity"].min()),
                    activity_max=float(pred["activity"].max()),
                )
            )
    elif not pred.empty:
        r2, rmse, mae = _safe_regression_metrics(pred["activity"], pred["predicted_activity"])
        range_rows.append(
            RangePerformance(
                range_label="all",
                n=int(len(pred)),
                r2=r2,
                rmse=rmse,
                mae=mae,
                activity_min=float(pred["activity"].min()),
                activity_max=float(pred["activity"].max()),
            )
        )

    in_mask = pred[in_domain_col].fillna(True).astype(bool)
    inside = _domain_performance(pred[in_mask])
    outside = _domain_performance(pred[~in_mask])

    residuals = pred["residual"].to_numpy(dtype=float) if not pred.empty else np.array([])
    predicted = pred["predicted_activity"].to_numpy(dtype=float) if not pred.empty else np.array([])
    corr = None
    if len(residuals) >= 2 and float(np.std(residuals)) > 0 and float(np.std(predicted)) > 0:
        corr = float(np.corrcoef(residuals, predicted)[0, 1])
    diagnostics = ResidualDiagnostics(
        mean=float(np.mean(residuals)) if len(residuals) else None,
        std=float(np.std(residuals, ddof=1)) if len(residuals) > 1 else None,
        residual_vs_predicted_correlation=corr,
    )
    return ErrorAnalysis(
        winner_run_id=winner_run_id,
        largest_error_compounds=largest,
        target_range_performance=range_rows,
        inside_domain=inside,
        outside_domain=outside,
        residual_diagnostics=diagnostics,
    )
