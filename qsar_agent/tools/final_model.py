"""Final model training and external evaluation."""

from __future__ import annotations

import json
import platform
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from qsar_agent.config import ModelConfig
from qsar_agent.schemas.modeling import Metrics, ModelingResult
from qsar_agent.services import build_estimator
from qsar_agent.services.artifact_manager import file_hash, save_json
from qsar_agent.services.plotting import plot_prediction_scatter
from qsar_agent.tools.mordred_descriptors import META_COLUMNS


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
) -> ModelingResult:
    """Train final model on GA-selected features; evaluate on external test."""
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

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

    predictions = []
    for split, df, y_true, y_pred in [
        ("train", train_df, y_train, y_train_pred),
        ("test", test_df, y_test, y_test_pred),
    ]:
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

    metrics_data = {
        "train": train_metrics.model_dump(),
        "test": test_metrics.model_dump(),
        "selected_features": selected_features,
    }
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
    }
    manifest_path = run_dir / "run_manifest.json"
    save_json(manifest_path, manifest)

    return ModelingResult(
        train_metrics=train_metrics,
        test_metrics=test_metrics,
        selected_features=selected_features,
        predictions_path=str(pred_path),
        metrics_path=str(metrics_path),
        model_path=str(model_path),
        scatter_png_path=str(png_path),
        scatter_svg_path=str(svg_path),
        manifest_path=str(manifest_path),
    )
