"""Feature count selection using one-standard-error rule."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from qsar_agent.schemas.feature_selection import FeatureCountSelection, SFSResult, SFSResultRow
from qsar_agent.tools.combined_score import combined_r2


def _row_combined(row: SFSResultRow) -> float:
    if row.combined_r2 is not None:
        return float(row.combined_r2)
    return combined_r2(row.mean_cv_r2, row.val_r2)


def select_feature_count_one_se_rule(sfs_result: SFSResult) -> FeatureCountSelection:
    """
    Select descriptor count using combined CV+val R² and the one-SE rule.

    Combined score is 0.5 * mean CV R² + 0.5 * validation R² (CV-only if val is
    missing). Returns the smallest feature count whose combined score is within
    one CV standard error of the best combined score.
    """
    results = sfs_result.results
    if not results:
        raise ValueError("No SFS results available for feature count selection.")

    best = max(results, key=_row_combined)
    best_combo = _row_combined(best)
    threshold = best_combo - best.std_cv_r2

    candidates = [r for r in results if _row_combined(r) >= threshold]
    selected = min(candidates, key=lambda r: r.n_features)
    selected_combo = _row_combined(selected)

    overfitting = (
        selected.mean_train_r2 - selected.mean_cv_r2 > 0.15
        if selected.mean_train_r2 > selected.mean_cv_r2
        else False
    )

    val_text = (
        f", val R² = {selected.val_r2:.4f}"
        if selected.val_r2 is not None
        else ", no validation R² (CV-only)"
    )
    explanation = (
        f"The highest combined R² (0.5·CV + 0.5·val) was {best_combo:.4f} "
        f"at {best.n_features} descriptor(s) "
        f"(CV R² = {best.mean_cv_r2:.4f}"
        f"{f', val R² = {best.val_r2:.4f}' if best.val_r2 is not None else ''}). "
        f"Applying the one-standard-error rule with CV std as SE "
        f"(threshold = {threshold:.4f}), "
        f"the smallest feature count within one SE of the best is "
        f"{selected.n_features} descriptor(s) with combined R² = {selected_combo:.4f} "
        f"(CV R² = {selected.mean_cv_r2:.4f}{val_text}). "
    )
    if overfitting:
        explanation += (
            "Training R² exceeds CV R², suggesting some overfitting may be present."
        )
    else:
        explanation += (
            "Training and cross-validation performance are reasonably aligned."
        )

    return FeatureCountSelection(
        best_cv_r2=best.mean_cv_r2,
        best_feature_count=best.n_features,
        selected_feature_count=selected.n_features,
        selected_cv_r2=selected.mean_cv_r2,
        best_combined_r2=best_combo,
        selected_combined_r2=selected_combo,
        selected_val_r2=selected.val_r2,
        explanation=explanation,
        selection_json_path="",
        explanation_md_path="",
    )


def save_feature_count_selection(
    selection: FeatureCountSelection,
    run_dir: Path,
) -> FeatureCountSelection:
    json_path = run_dir / "selected_feature_count.json"
    md_path = run_dir / "feature_count_selection_explanation.md"

    data = {
        "best_cv_r2": selection.best_cv_r2,
        "best_feature_count": selection.best_feature_count,
        "selected_feature_count": selection.selected_feature_count,
        "selected_cv_r2": selection.selected_cv_r2,
        "best_combined_r2": selection.best_combined_r2,
        "selected_combined_r2": selection.selected_combined_r2,
        "selected_val_r2": selection.selected_val_r2,
        "explanation": selection.explanation,
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Feature Count Selection\n\n")
        f.write(selection.explanation + "\n")

    return selection.model_copy(
        update={
            "selection_json_path": str(json_path),
            "explanation_md_path": str(md_path),
        }
    )


def select_feature_count_from_csv(
    sfs_results_path: str | Path,
    run_dir: Path,
) -> FeatureCountSelection:
    df = pd.read_csv(sfs_results_path)
    results = []
    for _, row in df.iterrows():
        val_r2 = None
        if "val_r2" in df.columns and pd.notna(row.get("val_r2")):
            val_r2 = float(row["val_r2"])
        combo = None
        if "combined_r2" in df.columns and pd.notna(row.get("combined_r2")):
            combo = float(row["combined_r2"])
        results.append(
            SFSResultRow(
                n_features=int(row["n_features"]),
                mean_train_r2=float(row["mean_train_r2"]),
                mean_cv_r2=float(row["mean_cv_r2"]),
                std_cv_r2=float(row["std_cv_r2"]),
                selected_features=[],
                val_r2=val_r2,
                combined_r2=combo,
            )
        )
    sfs_result = SFSResult(
        results=results,
        max_features_evaluated=len(results),
        results_csv_path=str(sfs_results_path),
        selected_features_json_path="",
        plot_png_path="",
        plot_svg_path="",
    )
    selection = select_feature_count_one_se_rule(sfs_result)
    return save_feature_count_selection(selection, run_dir)
