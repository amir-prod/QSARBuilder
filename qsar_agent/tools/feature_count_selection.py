"""Feature count selection using one-standard-error rule."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from qsar_agent.schemas.feature_selection import FeatureCountSelection, SFSResult


def select_feature_count_one_se_rule(sfs_result: SFSResult) -> FeatureCountSelection:
    """
    Select descriptor count using highest CV R² and one-standard-error rule.

    Returns the smallest feature count whose mean CV R² is within one SE of the best.
    """
    results = sfs_result.results
    if not results:
        raise ValueError("No SFS results available for feature count selection.")

    best = max(results, key=lambda r: r.mean_cv_r2)
    threshold = best.mean_cv_r2 - best.std_cv_r2

    candidates = [r for r in results if r.mean_cv_r2 >= threshold]
    selected = min(candidates, key=lambda r: r.n_features)

    overfitting = (
        selected.mean_train_r2 - selected.mean_cv_r2 > 0.15
        if selected.mean_train_r2 > selected.mean_cv_r2
        else False
    )

    explanation = (
        f"The highest mean cross-validation R² was {best.mean_cv_r2:.4f} "
        f"at {best.n_features} descriptor(s). "
        f"Applying the one-standard-error rule (threshold = {threshold:.4f}), "
        f"the smallest feature count within one SE of the best is "
        f"{selected.n_features} descriptor(s) with CV R² = {selected.mean_cv_r2:.4f}. "
    )
    if overfitting:
        explanation += (
            "Training R² exceeds validation R², suggesting some overfitting may be present."
        )
    else:
        explanation += (
            "Training and validation performance are reasonably aligned."
        )

    return FeatureCountSelection(
        best_cv_r2=best.mean_cv_r2,
        best_feature_count=best.n_features,
        selected_feature_count=selected.n_features,
        selected_cv_r2=selected.mean_cv_r2,
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
    results = [
        __import__(
            "qsar_agent.schemas.feature_selection", fromlist=["SFSResultRow"]
        ).SFSResultRow(
            n_features=int(row["n_features"]),
            mean_train_r2=float(row["mean_train_r2"]),
            mean_cv_r2=float(row["mean_cv_r2"]),
            std_cv_r2=float(row["std_cv_r2"]),
            selected_features=[],
        )
        for _, row in df.iterrows()
    ]
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
