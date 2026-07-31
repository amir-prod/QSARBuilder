"""Sequential feature selection tool."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold, cross_val_score

from qsar_agent.config import ModelConfig, SFSConfig
from qsar_agent.schemas.feature_selection import SFSResult, SFSResultRow
from qsar_agent.services.plotting import plot_sfs_r2
from qsar_agent.services import build_estimator
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
) -> SFSResult:
    """
    Run forward SFS for feature counts 1..min(max_features, n_descriptors).

    Uses a single mlxtend SFS fit (as in examples/utils.py build_each_model) and
    reads intermediate subsets from sfs.subsets_, rather than re-fitting SFS
    separately for every feature count.
    """
    df = pd.read_csv(train_path)
    X, y = _get_xy(df)
    n_descriptors = X.shape[1]
    max_eval = min(max_features, n_descriptors)

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

        results.append(
            SFSResultRow(
                n_features=k,
                mean_train_r2=train_r2,
                mean_cv_r2=mean_cv_r2,
                std_cv_r2=std_cv_r2,
                selected_features=selected_names,
            )
        )
        logger.info(
            "SFS k=%d: train R²=%.4f, CV R²=%.4f (+/- %.4f)",
            k,
            train_r2,
            mean_cv_r2,
            std_cv_r2,
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
