"""Sequential feature selection tool."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold

from qsar_agent.config import ModelConfig
from qsar_agent.schemas.feature_selection import SFSResult, SFSResultRow
from qsar_agent.services.plotting import plot_sfs_r2
from qsar_agent.services import build_estimator
from qsar_agent.tools.combined_score import combined_r2
from qsar_agent.tools.descriptor_calculation import META_COLUMNS

logger = logging.getLogger(__name__)


def _get_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    feature_cols = [c for c in df.columns if c not in META_COLUMNS]
    return df[feature_cols], df["activity"]


def run_sequential_feature_selection(
    train_path: str | Path,
    run_dir: Path,
    max_features: int = 20,
    cv_folds: int = 5,
    model_config: ModelConfig | None = None,
    random_seed: int = 42,
    n_jobs: int = -1,
    val_path: str | Path | None = None,
) -> SFSResult:
    """
    Run forward SFS for feature counts 1..min(max_features, n_descriptors).

    Uses a single mlxtend SFS fit (as in examples/utils.py build_each_model) and
    reads intermediate subsets from sfs.subsets_, rather than re-fitting SFS
    separately for every feature count. Search uses K-fold CV on train; each
    subset is also scored on the held-out validation set when ``val_path`` is set.
    """
    df = pd.read_csv(train_path)
    X, y = _get_xy(df)
    n_descriptors = X.shape[1]
    max_eval = min(max_features, n_descriptors)

    X_val = y_val = None
    if val_path is not None:
        val_df = pd.read_csv(val_path)
        X_val, y_val = _get_xy(val_df)

    logger.info(
        "Starting sequential feature selection: %d descriptors, evaluating 1..%d",
        n_descriptors,
        max_eval,
    )

    estimator = build_estimator(model_config)
    cv = KFold(n_splits=cv_folds, shuffle=True, random_state=random_seed)

    sfs = SFS(
        estimator,
        k_features=max_eval,
        forward=True,
        floating=False,
        verbose=2,
        scoring="r2",
        cv=cv,
        n_jobs=n_jobs,
    )
    sfs.fit(X.values, y.values.ravel())

    results: list[SFSResultRow] = []
    selected_by_count: dict[str, list[str]] = {}

    for k in range(1, max_eval + 1):
        if k not in sfs.subsets_:
            raise RuntimeError(f"SFS did not produce a subset for k={k}")

        subset = sfs.subsets_[k]
        idx = list(subset["feature_idx"])
        selected_names = [X.columns[i] for i in idx]
        selected_by_count[str(k)] = selected_names

        X_sel = X.iloc[:, idx]
        estimator_train = build_estimator(model_config)
        estimator_train.fit(X_sel, y)
        y_pred_train = estimator_train.predict(X_sel)
        train_r2 = float(r2_score(y, y_pred_train))

        mean_cv_r2 = float(subset["avg_score"])
        cv_scores = np.asarray(subset["cv_scores"], dtype=float)
        std_cv_r2 = float(cv_scores.std()) if len(cv_scores) > 1 else 0.0

        val_score = None
        if X_val is not None and y_val is not None:
            missing = [c for c in selected_names if c not in X_val.columns]
            if missing:
                raise ValueError(f"Selected features missing from validation data: {missing}")
            y_val_pred = estimator_train.predict(X_val[selected_names])
            val_score = float(r2_score(y_val, y_val_pred))

        combo = combined_r2(mean_cv_r2, val_score)

        results.append(
            SFSResultRow(
                n_features=k,
                mean_train_r2=train_r2,
                mean_cv_r2=mean_cv_r2,
                std_cv_r2=std_cv_r2,
                selected_features=selected_names,
                val_r2=val_score,
                combined_r2=combo,
            )
        )
        logger.info(
            "SFS k=%d: train R²=%.4f, CV R²=%.4f (+/- %.4f), val R²=%s, combined R²=%.4f",
            k,
            train_r2,
            mean_cv_r2,
            std_cv_r2,
            f"{val_score:.4f}" if val_score is not None else "n/a",
            combo,
        )

    results_df = pd.DataFrame([r.model_dump() for r in results])
    results_csv = run_dir / "sfs_results.csv"
    results_df.to_csv(results_csv, index=False)

    features_json = run_dir / "sfs_selected_features_by_count.json"
    with open(features_json, "w", encoding="utf-8") as f:
        json.dump(selected_by_count, f, indent=2)

    png_path = run_dir / "sfs_r2_vs_feature_count.png"
    svg_path = run_dir / "sfs_r2_vs_feature_count.svg"
    plot_sfs_r2(results_df, png_path, svg_path)

    logger.info("Sequential feature selection complete.")
    return SFSResult(
        results=results,
        max_features_evaluated=max_eval,
        results_csv_path=str(results_csv),
        selected_features_json_path=str(features_json),
        plot_png_path=str(png_path),
        plot_svg_path=str(svg_path),
    )
