"""Deterministic model compatibility validation against registry specs."""

from __future__ import annotations

from typing import Any

from qsar_agent.models.registry import (
    count_grid_combinations,
    get_fallback_grid,
    get_model_specification,
    list_registered_estimators,
)
from qsar_agent.schemas.agentic import ModelCompatibilityResult


def validate_model_compatibility(
    estimator_name: str,
    *,
    n_train_samples: int,
    n_features: int,
    is_sparse: bool = False,
    has_missing: bool = False,
    max_candidates: int = 60,
    compute_budget: str = "medium",
) -> ModelCompatibilityResult:
    if estimator_name not in list_registered_estimators(include_unavailable=True):
        return ModelCompatibilityResult(
            estimator_name=estimator_name,
            compatible=False,
            blocking_reasons=[f"Estimator '{estimator_name}' is not in the model catalog."],
            estimated_cost="n/a",
        )

    spec = get_model_specification(estimator_name)
    blocking: list[str] = []
    warnings: list[str] = []
    required_prep: list[str] = []

    if not spec.available:
        blocking.append(
            f"dependency_not_available: {spec.missing_dependency or 'unknown dependency'}"
        )

    if is_sparse and not spec.supports_sparse:
        if n_features > 5000:
            blocking.append("Sparse high-dimensional matrix incompatible with dense-only estimator.")
        else:
            warnings.append("Sparse inputs may be densified; ensure this is intentional.")

    if not is_sparse and not spec.supports_dense:
        blocking.append("Dense matrices are not supported by this estimator.")

    if spec.minimum_training_samples and n_train_samples < spec.minimum_training_samples:
        blocking.append(
            f"Need at least {spec.minimum_training_samples} training samples "
            f"(have {n_train_samples})."
        )

    if spec.maximum_recommended_features_ratio and n_features > 0:
        ratio = n_train_samples / float(n_features)
        if ratio < spec.maximum_recommended_features_ratio:
            warnings.append(
                f"Samples-per-feature ratio {ratio:.2f} is below recommended "
                f"{spec.maximum_recommended_features_ratio}."
            )

    if spec.requires_scaling:
        required_prep.append("feature_scaling")

    if estimator_name == "PLSRegression":
        max_comp = max(1, min(n_features, max(1, n_train_samples - 1)))
        if max_comp < 1:
            blocking.append("PLS n_components cannot be formed with current data shape.")

    if estimator_name == "KNeighborsRegressor" and n_train_samples < 2:
        blocking.append("KNN requires at least 2 training samples.")

    if has_missing:
        warnings.append("Missing values should be imputed before fitting.")

    grid = get_fallback_grid(estimator_name, "default")
    # Clamp PLS/KNN bounds in estimated candidate count
    if estimator_name == "PLSRegression" and "n_components" in grid:
        max_comp = max(1, min(n_features, max(1, n_train_samples - 1)))
        grid = {**grid, "n_components": [c for c in grid["n_components"] if c <= max_comp] or [1]}
    if estimator_name == "KNeighborsRegressor" and "n_neighbors" in grid:
        grid = {
            **grid,
            "n_neighbors": [k for k in grid["n_neighbors"] if k < n_train_samples] or [1],
        }
    if estimator_name == "SVR":
        n_comb = count_grid_combinations(grid)
        if n_comb > max_candidates:
            warnings.append(f"SVR default grid has {n_comb} candidates; will be shrunk to {max_candidates}.")

    n_comb = count_grid_combinations(grid)
    cost = spec.computational_cost
    if compute_budget == "low" and cost == "high":
        warnings.append("Estimator computational cost is high relative to compute budget.")

    return ModelCompatibilityResult(
        estimator_name=estimator_name,
        compatible=len(blocking) == 0,
        blocking_reasons=blocking,
        warnings=warnings,
        required_preprocessing=required_prep,
        estimated_candidate_count=min(n_comb, max_candidates),
        estimated_cost=cost,
    )


def reject_arbitrary_import_path(estimator_or_path: str) -> None:
    if "." in estimator_or_path and estimator_or_path not in list_registered_estimators(
        include_unavailable=True
    ):
        # Registry names never contain dots; import paths do
        if any(part in estimator_or_path for part in ("sklearn", "xgboost", "catboost", "lightgbm", "import")):
            raise ValueError(
                "Arbitrary import paths are not allowed; use a registered estimator_name."
            )
    if "/" in estimator_or_path or "\\" in estimator_or_path:
        raise ValueError("Arbitrary import paths are not allowed.")
