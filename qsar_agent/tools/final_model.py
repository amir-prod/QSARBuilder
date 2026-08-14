"""Final model training and external evaluation."""

from __future__ import annotations

import platform
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from qsar_agent.config import ModelConfig
from qsar_agent.schemas.modeling import Metrics, ModelingResult
from qsar_agent.services import build_estimator
from qsar_agent.services.artifact_manager import save_json
from qsar_agent.services.plotting import plot_prediction_scatter


def _compute_metrics(y_true, y_pred) -> Metrics:
    return Metrics(
        r2=float(r2_score(y_true, y_pred)),
        rmse=float(np.sqrt(mean_squared_error(y_true, y_pred))),
        mae=float(mean_absolute_error(y_true, y_pred)),
        n_samples=len(y_true),
    )


def train_and_evaluate_final_model(
    train_path: str | Path,
    test_path: str | Path,
    run_dir: Path,
    selected_features: list[str],
    model_config: ModelConfig | None = None,
    activity_label: str = "activity",
    dataset_hash: str = "",
    config_snapshot: dict | None = None,
    hpo_metadata: dict[str, Any] | None = None,
    val_path: str | Path | None = None,
) -> ModelingResult:
    """Train final model on train only; score val (development) and external test."""
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    val_df = pd.read_csv(val_path) if val_path is not None else None

    for feat in selected_features:
        if feat not in train_df.columns:
            raise ValueError(f"Selected feature not in training data: {feat}")

    X_train = train_df[selected_features]
    y_train = train_df["activity"]
    X_test = test_df[selected_features]
    y_test = test_df["activity"]

    model = build_estimator(model_config)
    model.fit(X_train, y_train)

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    train_metrics = _compute_metrics(y_train, y_train_pred)
    test_metrics = _compute_metrics(y_test, y_test_pred)

    val_metrics = None
    y_val = y_val_pred = None
    if val_df is not None:
        missing = [f for f in selected_features if f not in val_df.columns]
        if missing:
            raise ValueError(f"Selected features missing from validation data: {missing}")
        X_val = val_df[selected_features]
        y_val = val_df["activity"]
        y_val_pred = model.predict(X_val)
        val_metrics = _compute_metrics(y_val, y_val_pred)

    predictions = []
    split_rows = [("train", train_df, y_train, y_train_pred)]
    if val_df is not None and y_val is not None and y_val_pred is not None:
        split_rows.append(("val", val_df, y_val, y_val_pred))
    split_rows.append(("test", test_df, y_test, y_test_pred))
    for split, df, y_true, y_pred in split_rows:
        for i in range(len(df)):
            predictions.append(
                {
                    "compound_id": df.iloc[i]["compound_id"],
                    "canonical_smiles": df.iloc[i]["canonical_smiles"],
                    "activity": float(y_true.iloc[i]),
                    "predicted_activity": float(y_pred[i]),
                    "split": split,
                    "residual": float(y_true.iloc[i] - y_pred[i]),
                }
            )

    pred_path = run_dir / "predictions.csv"
    pd.DataFrame(predictions).to_csv(pred_path, index=False)

    hpo_meta = hpo_metadata or {}
    metrics_data = {
        "train": train_metrics.model_dump(),
        "test": test_metrics.model_dump(),
        "selected_features": selected_features,
        "hyperparameter_optimization": hpo_meta,
    }
    if val_metrics is not None:
        metrics_data["val"] = val_metrics.model_dump()
    metrics_path = run_dir / "model_metrics.json"
    save_json(metrics_path, metrics_data)

    model_path = run_dir / "final_model.joblib"
    joblib.dump(model, model_path)

    png_path = run_dir / "prediction_scatter.png"
    svg_path = run_dir / "prediction_scatter.svg"
    plot_prediction_scatter(
        y_train.values,
        y_train_pred,
        y_test.values,
        y_test_pred,
        train_metrics.model_dump(),
        test_metrics.model_dump(),
        activity_label,
        png_path,
        svg_path,
        val_true=None if y_val is None else y_val.values,
        val_pred=y_val_pred,
        val_metrics=None if val_metrics is None else val_metrics.model_dump(),
    )

    import sklearn

    manifest = {
        "python_version": sys.version,
        "platform": platform.platform(),
        "sklearn_version": sklearn.__version__,
        "model_config": (model_config or ModelConfig()).model_dump(),
        "selected_features": selected_features,
        "dataset_hash": dataset_hash,
        "workflow_config": config_snapshot or {},
        "hyperparameter_optimization": hpo_meta,
    }
    manifest_path = run_dir / "run_manifest.json"
    save_json(manifest_path, manifest)

    return ModelingResult(
        train_metrics=train_metrics,
        val_metrics=val_metrics,
        test_metrics=test_metrics,
        selected_features=selected_features,
        predictions_path=str(pred_path),
        metrics_path=str(metrics_path),
        model_path=str(model_path),
        scatter_png_path=str(png_path),
        scatter_svg_path=str(svg_path),
        manifest_path=str(manifest_path),
        hpo_enabled=bool(hpo_meta.get("enabled", False)),
        hpo_rounds_completed=int(hpo_meta.get("rounds_completed", 0)),
        final_model_source=str(hpo_meta.get("final_model_source", "baseline")),
    )
