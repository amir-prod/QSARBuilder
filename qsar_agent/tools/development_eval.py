"""Train/CV/val evaluation that never opens the sealed external test set."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold

from qsar_agent.config import ModelConfig
from qsar_agent.services import build_estimator
from qsar_agent.services.artifact_manager import save_json
from qsar_agent.tools.descriptor_calculation import META_COLUMNS
from qsar_agent.tools.hyperparameter_optimization import evaluate_baseline_model_cv


def _feature_frame(df: pd.DataFrame, selected_features: list[str]) -> tuple[pd.DataFrame, pd.Series]:
    missing = [c for c in selected_features if c not in df.columns]
    if missing:
        raise ValueError(f"Selected features missing: {missing}")
    return df[selected_features], df["activity"]


def score_holdout(train_path: str | Path, holdout_path: str | Path, features: list[str], model_config: ModelConfig | None = None) -> dict[str, float | None]:
    train = pd.read_csv(train_path)
    holdout = pd.read_csv(holdout_path)
    X_tr, y_tr = _feature_frame(train, features)
    X_ho, y_ho = _feature_frame(holdout, features)
    model = build_estimator(model_config)
    model.fit(X_tr, y_tr)
    pred = model.predict(X_ho)
    if len(y_ho) < 2 or np.allclose(y_ho, y_ho.iloc[0]):
        r2 = None
    else:
        r2 = float(r2_score(y_ho, pred))
    return {
        "r2": r2,
        "rmse": float(np.sqrt(mean_squared_error(y_ho, pred))),
        "mae": float(mean_absolute_error(y_ho, pred)),
        "n": float(len(y_ho)),
    }


def oof_predictions(
    train_path: str | Path,
    selected_features: list[str],
    model_config: ModelConfig | None = None,
    cv_folds: int = 5,
    random_seed: int = 42,
) -> pd.DataFrame:
    """Out-of-fold predictions on the training partition only."""
    df = pd.read_csv(train_path)
    X, y = _feature_frame(df, selected_features)
    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=random_seed)
    records: list[dict[str, Any]] = []
    oof = np.full(len(df), np.nan)
    for fold_idx, (tr_idx, val_idx) in enumerate(kf.split(X), start=1):
        model = build_estimator(model_config)
        model.fit(X.iloc[tr_idx], y.iloc[tr_idx])
        pred = model.predict(X.iloc[val_idx])
        oof[val_idx] = pred
        for i, row_i in enumerate(val_idx):
            records.append(
                {
                    "compound_id": str(df.iloc[row_i]["compound_id"]),
                    "fold": fold_idx,
                    "activity": float(y.iloc[row_i]),
                    "predicted_activity": float(pred[i]),
                    "residual": float(y.iloc[row_i] - pred[i]),
                    "split": "oof",
                }
            )
    return pd.DataFrame(records)


def evaluate_feature_subset(
    train_path: str | Path,
    val_path: str | Path | None,
    selected_features: list[str],
    out_dir: Path,
    model_config: ModelConfig | None = None,
    cv_folds: int = 5,
    random_seed: int = 42,
) -> dict[str, Any]:
    """Inner-fold CV on train plus optional outer validation holdout. Never reads test."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cv = evaluate_baseline_model_cv(
        train_path,
        selected_features,
        model_config=model_config,
        cv_folds=cv_folds,
        random_seed=random_seed,
        run_dir=out_dir,
    )
    oof = oof_predictions(train_path, selected_features, model_config, cv_folds, random_seed)
    oof_path = out_dir / "oof_predictions.csv"
    oof.to_csv(oof_path, index=False)
    val_metrics = None
    if val_path is not None:
        val_metrics = score_holdout(train_path, val_path, selected_features, model_config)
        save_json(out_dir / "val_metrics.json", val_metrics)
    summary = cv.summary
    metrics = {
        "mean_cv_fold_train_r2": summary.mean_train_r2,
        "oof_cv_r2": summary.mean_cv_r2,
        "cv_r2": summary.mean_cv_r2,
        "cv_r2_std": summary.std_cv_r2,
        "cv_rmse": summary.mean_cv_rmse,
        "cv_mae": summary.mean_cv_mae,
        "train_cv_r2_gap": summary.train_cv_r2_gap,
        "cv_fold_train_val_gap": summary.train_cv_r2_gap,
        "val_r2": None if not val_metrics else val_metrics["r2"],
        "val_rmse": None if not val_metrics else val_metrics["rmse"],
        "val_mae": None if not val_metrics else val_metrics["mae"],
        "feature_count": len(selected_features),
        "refit_train_r2": None,
        "refit_train_cv_gap": None,
    }
    # Refit train R² for interpretation only.
    train = pd.read_csv(train_path)
    X, y = _feature_frame(train, selected_features)
    model = build_estimator(model_config)
    model.fit(X, y)
    refit_pred = model.predict(X)
    metrics["refit_train_r2"] = float(r2_score(y, refit_pred))
    metrics["refit_train_cv_gap"] = metrics["refit_train_r2"] - float(summary.mean_cv_r2)
    save_json(out_dir / "development_metrics.json", metrics)
    save_json(out_dir / "selected_features.json", {"selected_features": list(selected_features)})
    return {
        "metrics": metrics,
        "selected_features": list(selected_features),
        "fold_metrics": [f.model_dump() for f in cv.fold_metrics],
        "oof_predictions_path": str(oof_path),
        "cv_summary": summary.model_dump(),
    }
