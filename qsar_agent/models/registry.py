"""Per-estimator defaults, HPO spaces, and factory."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR

from qsar_agent.config import ModelConfig
from qsar_agent.schemas.hyperparameter_optimization import GridSanitizationResult


class AdaptivePLSRegression(PLSRegression):
    """PLSRegression that clamps n_components to fit-time data shape.

    Required for SFS/GA, which evaluate subsets with fewer features than the
    configured n_components (sklearn requires n_components <= min(n_features, n_samples-1)).
    """

    def fit(self, X, y=None):
        X_arr = np.asarray(X)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(-1, 1)
        n_samples, n_features = X_arr.shape
        max_comp = max(1, min(int(n_features), max(1, int(n_samples) - 1)))
        self.n_components = min(int(self.n_components), max_comp)
        return super().fit(X, y)


class AdaptiveKNeighborsRegressor(KNeighborsRegressor):
    """KNN that clamps n_neighbors to the number of training samples."""

    def fit(self, X, y):
        X_arr = np.asarray(X)
        n_samples = X_arr.shape[0]
        self.n_neighbors = min(int(self.n_neighbors), max(1, int(n_samples)))
        return super().fit(X, y)


DEFAULT_FALLBACK_ESTIMATORS = [
    "PLSRegression",
    "ExtraTreesRegressor",
    "SVR",
    "KNeighborsRegressor",
]

SUPPORTED_ESTIMATORS = ["RandomForestRegressor", *DEFAULT_FALLBACK_ESTIMATORS]


def normalize_estimator_name(estimator: str | None) -> str:
    """Map display labels such as ``PLSRegression (sfs_fixed_ga_plus2)`` to registry names."""
    if not estimator:
        return ""
    name = str(estimator).strip()
    if name in SUPPORTED_ESTIMATORS:
        return name
    base = name.split("(", 1)[0].strip()
    if base in SUPPORTED_ESTIMATORS:
        return base
    return name

_RF_DEFAULTS: dict[str, Any] = {
    "n_estimators": 100,
    "max_depth": 10,
    "min_samples_split": 2,
    "min_samples_leaf": 1,
    "max_features": "sqrt",
    "bootstrap": True,
    "max_samples": None,
    "criterion": "squared_error",
}

_PLS_DEFAULTS: dict[str, Any] = {
    "n_components": 10,
    "scale": False,
    "max_iter": 500,
}

_EXTRA_TREES_DEFAULTS: dict[str, Any] = {
    "n_estimators": 100,
    "max_depth": 10,
    "min_samples_split": 2,
    "min_samples_leaf": 1,
    "max_features": "sqrt",
    "bootstrap": False,
}

_SVR_DEFAULTS: dict[str, Any] = {
    "C": 1.0,
    "epsilon": 0.1,
    "gamma": "scale",
    "kernel": "rbf",
}

_KNN_DEFAULTS: dict[str, Any] = {
    "n_neighbors": 5,
    "weights": "uniform",
    "p": 2,
    "metric": "minkowski",
}

_DEFAULT_PARAMS: dict[str, dict[str, Any]] = {
    "RandomForestRegressor": _RF_DEFAULTS,
    "PLSRegression": _PLS_DEFAULTS,
    "ExtraTreesRegressor": _EXTRA_TREES_DEFAULTS,
    "SVR": _SVR_DEFAULTS,
    "KNeighborsRegressor": _KNN_DEFAULTS,
}

_RF_SPACE: dict[str, set[Any]] = {
    "n_estimators": set(range(100, 1001)),
    "max_depth": set(range(2, 51)) | {None},
    "min_samples_split": set(range(2, 31)),
    "min_samples_leaf": set(range(1, 21)),
    "max_features": {"sqrt", "log2", 0.3, 0.5, 0.7, 1.0},
    "bootstrap": {True, False},
    "max_samples": {None} | {round(v, 2) for v in np.arange(0.5, 1.01, 0.05)},
    "criterion": {"squared_error", "absolute_error", "poisson"},
}

_PLS_SPACE: dict[str, set[Any]] = {
    "n_components": set(range(1, 51)),
    "scale": {True, False},
    "max_iter": {100, 200, 500, 1000},
}

_EXTRA_TREES_SPACE: dict[str, set[Any]] = {
    "n_estimators": set(range(100, 1001)),
    "max_depth": set(range(2, 51)) | {None},
    "min_samples_split": set(range(2, 31)),
    "min_samples_leaf": set(range(1, 21)),
    "max_features": {"sqrt", "log2", 0.3, 0.5, 0.7, 1.0},
    "bootstrap": {True, False},
}

_SVR_SPACE: dict[str, set[Any]] = {
    "C": {0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0},
    "epsilon": {0.01, 0.05, 0.1, 0.2, 0.5},
    "gamma": {"scale", "auto", 0.001, 0.01, 0.1, 1.0},
    "kernel": {"rbf", "linear", "poly"},
}

_KNN_SPACE: dict[str, set[Any]] = {
    "n_neighbors": set(range(1, 31)),
    "weights": {"uniform", "distance"},
    "p": {1, 2},
    "metric": {"minkowski", "euclidean", "manhattan"},
}

_ALLOWED_SPACE: dict[str, dict[str, set[Any]]] = {
    "RandomForestRegressor": _RF_SPACE,
    "PLSRegression": _PLS_SPACE,
    "ExtraTreesRegressor": _EXTRA_TREES_SPACE,
    "SVR": _SVR_SPACE,
    "KNeighborsRegressor": _KNN_SPACE,
}

_RF_FALLBACK_OVERFIT = {
    "n_estimators": [200, 500],
    "max_depth": [3, 5, 8, 12],
    "min_samples_split": [4, 8, 12],
    "min_samples_leaf": [2, 4, 8],
    "max_features": ["sqrt", 0.3, 0.5],
    "bootstrap": [True],
    "max_samples": [0.7, 0.9],
}

_RF_FALLBACK_UNDERFIT = {
    "n_estimators": [300, 500, 800],
    "max_depth": [None, 15, 25, 40],
    "min_samples_split": [2, 4],
    "min_samples_leaf": [1, 2],
    "max_features": ["sqrt", 0.5, 0.7, 1.0],
    "bootstrap": [True, False],
}

_RF_FALLBACK_UNSTABLE = {
    "n_estimators": [500, 800, 1000],
    "max_depth": [5, 8, 12, 20],
    "min_samples_split": [4, 8],
    "min_samples_leaf": [2, 4, 6],
    "max_features": ["sqrt", 0.3, 0.5],
    "bootstrap": [True],
    "max_samples": [0.7, 0.85, 1.0],
}

_RF_FALLBACK_DEFAULT = {
    "n_estimators": [200, 400],
    "max_depth": [5, 10, 15],
    "min_samples_split": [2, 4, 8],
    "min_samples_leaf": [1, 2, 4],
    "max_features": ["sqrt", 0.5],
    "bootstrap": [True],
    "max_samples": [0.8, 1.0],
}

_PLS_FALLBACK_OVERFIT = {
    "n_components": [2, 3, 5, 8],
    "scale": [False, True],
    "max_iter": [500, 1000],
}

_PLS_FALLBACK_UNDERFIT = {
    "n_components": [10, 15, 20, 30],
    "scale": [False, True],
    "max_iter": [500, 1000],
}

_PLS_FALLBACK_UNSTABLE = {
    "n_components": [5, 8, 10, 15],
    "scale": [True],
    "max_iter": [1000],
}

_PLS_FALLBACK_DEFAULT = {
    "n_components": [5, 10, 15],
    "scale": [False, True],
    "max_iter": [500],
}

_EXTRA_TREES_FALLBACK_OVERFIT = {
    "n_estimators": [200, 500],
    "max_depth": [3, 5, 8, 12],
    "min_samples_split": [4, 8, 12],
    "min_samples_leaf": [2, 4, 8],
    "max_features": ["sqrt", 0.3, 0.5],
}

_EXTRA_TREES_FALLBACK_UNDERFIT = {
    "n_estimators": [300, 500, 800],
    "max_depth": [None, 15, 25, 40],
    "min_samples_split": [2, 4],
    "min_samples_leaf": [1, 2],
    "max_features": ["sqrt", 0.5, 0.7, 1.0],
}

_EXTRA_TREES_FALLBACK_UNSTABLE = {
    "n_estimators": [500, 800, 1000],
    "max_depth": [5, 8, 12, 20],
    "min_samples_split": [4, 8],
    "min_samples_leaf": [2, 4, 6],
    "max_features": ["sqrt", 0.3, 0.5],
}

_EXTRA_TREES_FALLBACK_DEFAULT = {
    "n_estimators": [200, 400],
    "max_depth": [5, 10, 15],
    "min_samples_split": [2, 4, 8],
    "min_samples_leaf": [1, 2, 4],
    "max_features": ["sqrt", 0.5],
}

_SVR_FALLBACK_OVERFIT = {
    "C": [0.1, 0.5, 1.0],
    "epsilon": [0.1, 0.2, 0.5],
    "gamma": ["scale", 0.01, 0.1],
    "kernel": ["rbf", "linear"],
}

_SVR_FALLBACK_UNDERFIT = {
    "C": [10.0, 50.0, 100.0],
    "epsilon": [0.01, 0.05, 0.1],
    "gamma": ["scale", 0.1, 1.0],
    "kernel": ["rbf", "poly"],
}

_SVR_FALLBACK_UNSTABLE = {
    "C": [1.0, 5.0, 10.0],
    "epsilon": [0.05, 0.1, 0.2],
    "gamma": ["scale", 0.01],
    "kernel": ["rbf"],
}

_SVR_FALLBACK_DEFAULT = {
    "C": [0.5, 1.0, 5.0, 10.0],
    "epsilon": [0.05, 0.1, 0.2],
    "gamma": ["scale", 0.01, 0.1],
    "kernel": ["rbf", "linear"],
}

_KNN_FALLBACK_OVERFIT = {
    "n_neighbors": [15, 20, 25],
    "weights": ["distance"],
    "p": [2],
    "metric": ["minkowski"],
}

_KNN_FALLBACK_UNDERFIT = {
    "n_neighbors": [1, 3, 5],
    "weights": ["uniform", "distance"],
    "p": [1, 2],
    "metric": ["minkowski", "euclidean"],
}

_KNN_FALLBACK_UNSTABLE = {
    "n_neighbors": [5, 7, 9, 11],
    "weights": ["uniform"],
    "p": [2],
    "metric": ["minkowski"],
}

_KNN_FALLBACK_DEFAULT = {
    "n_neighbors": [3, 5, 7, 9],
    "weights": ["uniform", "distance"],
    "p": [2],
    "metric": ["minkowski"],
}

_FALLBACK_GRIDS: dict[str, dict[str, dict[str, list[Any]]]] = {
    "RandomForestRegressor": {
        "overfit": _RF_FALLBACK_OVERFIT,
        "underfit": _RF_FALLBACK_UNDERFIT,
        "unstable": _RF_FALLBACK_UNSTABLE,
        "default": _RF_FALLBACK_DEFAULT,
    },
    "PLSRegression": {
        "overfit": _PLS_FALLBACK_OVERFIT,
        "underfit": _PLS_FALLBACK_UNDERFIT,
        "unstable": _PLS_FALLBACK_UNSTABLE,
        "default": _PLS_FALLBACK_DEFAULT,
    },
    "ExtraTreesRegressor": {
        "overfit": _EXTRA_TREES_FALLBACK_OVERFIT,
        "underfit": _EXTRA_TREES_FALLBACK_UNDERFIT,
        "unstable": _EXTRA_TREES_FALLBACK_UNSTABLE,
        "default": _EXTRA_TREES_FALLBACK_DEFAULT,
    },
    "SVR": {
        "overfit": _SVR_FALLBACK_OVERFIT,
        "underfit": _SVR_FALLBACK_UNDERFIT,
        "unstable": _SVR_FALLBACK_UNSTABLE,
        "default": _SVR_FALLBACK_DEFAULT,
    },
    "KNeighborsRegressor": {
        "overfit": _KNN_FALLBACK_OVERFIT,
        "underfit": _KNN_FALLBACK_UNDERFIT,
        "unstable": _KNN_FALLBACK_UNSTABLE,
        "default": _KNN_FALLBACK_DEFAULT,
    },
}

_HPO_PROMPTS: dict[str, str] = {
    "RandomForestRegressor": (
        "Allowed params: n_estimators (100-1000), max_depth (2-50 or null), "
        "min_samples_split (2-30), min_samples_leaf (1-20), "
        "max_features (sqrt, log2, 0.3, 0.5, 0.7, 1.0), bootstrap (true/false), "
        "max_samples (null or 0.5-1.0, only with bootstrap=true), "
        "criterion (squared_error, absolute_error, poisson)."
    ),
    "PLSRegression": (
        "Allowed params: n_components (1 to min(n_features, n_train-1)), "
        "scale (true/false), max_iter (100, 200, 500, 1000)."
    ),
    "ExtraTreesRegressor": (
        "Allowed params: n_estimators (100-1000), max_depth (2-50 or null), "
        "min_samples_split (2-30), min_samples_leaf (1-20), "
        "max_features (sqrt, log2, 0.3, 0.5, 0.7, 1.0), bootstrap (true/false)."
    ),
    "SVR": (
        "Allowed params: C (0.1-100), epsilon (0.01-0.5), "
        "gamma (scale, auto, or numeric 0.001-1.0), kernel (rbf, linear, poly)."
    ),
    "KNeighborsRegressor": (
        "Allowed params: n_neighbors (1-30), weights (uniform, distance), "
        "p (1 or 2), metric (minkowski, euclidean, manhattan)."
    ),
}


def estimator_slug(estimator: str) -> str:
    return {
        "RandomForestRegressor": "random_forest",
        "PLSRegression": "pls_regression",
        "ExtraTreesRegressor": "extra_trees_regressor",
        "SVR": "svr",
        "KNeighborsRegressor": "k_neighbors_regressor",
    }.get(estimator, estimator.lower())


def get_tunable_params(estimator: str) -> set[str]:
    return set(_ALLOWED_SPACE.get(estimator, {}).keys())


def get_allowed_param_space(estimator: str) -> dict[str, set[Any]]:
    return _ALLOWED_SPACE.get(estimator, {})


def get_hpo_prompt_spec(estimator: str) -> str:
    return _HPO_PROMPTS.get(estimator, "")


def get_default_model_config(estimator: str, random_state: int = 42, n_jobs: int = -1) -> ModelConfig:
    estimator = normalize_estimator_name(estimator)
    if estimator not in SUPPORTED_ESTIMATORS:
        raise ValueError(f"Unsupported estimator: {estimator}")
    params = dict(_DEFAULT_PARAMS[estimator])
    cfg_data: dict[str, Any] = {
        "estimator": estimator,
        "random_state": random_state,
        "n_jobs": n_jobs,
        "params": params,
    }
    if estimator == "RandomForestRegressor":
        cfg_data.update(params)
    return ModelConfig(**cfg_data)


def resolve_params(config: ModelConfig) -> dict[str, Any]:
    """Merge ModelConfig into a flat param dict for sklearn."""
    defaults = dict(_DEFAULT_PARAMS.get(config.estimator, {}))
    if config.estimator == "RandomForestRegressor":
        for key in _RF_DEFAULTS:
            val = getattr(config, key, None)
            if val is not None:
                defaults[key] = val
    if config.params:
        defaults.update(config.params)
    # sklearn forbids max_samples when bootstrap=False
    if config.estimator == "RandomForestRegressor" and not defaults.get("bootstrap", True):
        defaults["max_samples"] = None
    return defaults


def baseline_params_from_config(config: ModelConfig) -> dict[str, Any]:
    tunable = get_tunable_params(config.estimator)
    resolved = resolve_params(config)
    return {k: resolved[k] for k in tunable if k in resolved}


def build_estimator_from_config(config: ModelConfig | dict[str, Any] | None = None):
    if isinstance(config, dict):
        cfg = ModelConfig(**{**ModelConfig().model_dump(), **config})
    else:
        cfg = config or ModelConfig()

    estimator = normalize_estimator_name(cfg.estimator)
    if estimator != cfg.estimator:
        cfg = cfg.model_copy(update={"estimator": estimator})
    params = resolve_params(cfg)

    if estimator == "RandomForestRegressor":
        kwargs: dict[str, Any] = {
            "n_estimators": params["n_estimators"],
            "max_depth": params["max_depth"],
            "min_samples_split": params["min_samples_split"],
            "min_samples_leaf": params["min_samples_leaf"],
            "max_features": params["max_features"],
            "bootstrap": params["bootstrap"],
            "criterion": params["criterion"],
            "random_state": cfg.random_state,
            "n_jobs": cfg.n_jobs,
        }
        if params.get("bootstrap") and params.get("max_samples") is not None:
            kwargs["max_samples"] = params["max_samples"]
        return RandomForestRegressor(**kwargs)

    if estimator == "PLSRegression":
        return AdaptivePLSRegression(
            n_components=params["n_components"],
            scale=params["scale"],
            max_iter=params["max_iter"],
        )

    if estimator == "ExtraTreesRegressor":
        return ExtraTreesRegressor(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            min_samples_split=params["min_samples_split"],
            min_samples_leaf=params["min_samples_leaf"],
            max_features=params["max_features"],
            bootstrap=params["bootstrap"],
            random_state=cfg.random_state,
            n_jobs=cfg.n_jobs,
        )

    if estimator == "SVR":
        return SVR(
            C=params["C"],
            epsilon=params["epsilon"],
            gamma=params["gamma"],
            kernel=params["kernel"],
        )

    if estimator == "KNeighborsRegressor":
        return AdaptiveKNeighborsRegressor(
            n_neighbors=params["n_neighbors"],
            weights=params["weights"],
            p=params["p"],
            metric=params["metric"],
            n_jobs=cfg.n_jobs,
        )

    raise ValueError(f"Unsupported estimator: {estimator}")


def count_grid_combinations(grid: dict[str, list[Any]]) -> int:
    if not grid:
        return 0
    total = 1
    for values in grid.values():
        total *= max(len(values), 1)
    return total


def _normalize_value(estimator: str, param: str, value: Any) -> Any:
    if param in ("max_depth",) and value in ("None", "null"):
        return None
    if param in ("bootstrap", "scale") and isinstance(value, str):
        return value.lower() in ("true", "1", "yes")
    if param == "max_features" and isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    if param == "max_samples" and value is not None:
        return float(value)
    if param == "gamma" and isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    if param == "C" and value is not None:
        return float(value)
    if param == "epsilon" and value is not None:
        return float(value)
    if param == "n_components" and value is not None:
        return int(value)
    if param == "n_neighbors" and value is not None:
        return int(value)
    if param == "p" and value is not None:
        return int(value)
    return value


def _value_allowed(estimator: str, param: str, value: Any) -> bool:
    value = _normalize_value(estimator, param, value)
    allowed = get_allowed_param_space(estimator).get(param)
    if allowed is None:
        return False
    if value in allowed:
        return True
    if param == "max_samples" and value is not None:
        return 0.5 <= float(value) <= 1.0
    if param == "n_components" and isinstance(value, int):
        return 1 <= value <= 50
    if param == "gamma" and isinstance(value, (int, float)):
        return 0.001 <= float(value) <= 1.0
    if param == "C" and isinstance(value, (int, float)):
        return 0.1 <= float(value) <= 100.0
    return False


def sanitize_param_grid(
    estimator: str,
    param_grid: dict[str, list[Any]],
    max_candidates: int = 120,
    random_seed: int = 42,
    n_features: int | None = None,
    n_train_samples: int | None = None,
) -> GridSanitizationResult:
    """Validate, filter, and shrink a hyperparameter grid deterministically."""
    allowed_params = get_tunable_params(estimator)
    removed_params: list[str] = []
    removed_values: dict[str, list[Any]] = {}
    shrink_steps: list[str] = []
    warnings: list[str] = []

    sanitized: dict[str, list[Any]] = {}
    for param, values in param_grid.items():
        if param not in allowed_params:
            removed_params.append(param)
            continue
        clean: list[Any] = []
        rejected: list[Any] = []
        for v in values:
            norm = _normalize_value(estimator, param, v)
            if param == "n_components" and isinstance(norm, int):
                max_comp = 50
                if n_features is not None:
                    max_comp = min(max_comp, n_features)
                if n_train_samples is not None:
                    max_comp = min(max_comp, max(1, n_train_samples - 1))
                if norm > max_comp:
                    rejected.append(v)
                    continue
            if _value_allowed(estimator, param, norm):
                if norm not in clean:
                    clean.append(norm)
            else:
                rejected.append(v)
        if rejected:
            removed_values[param] = rejected
        if clean:
            sanitized[param] = clean

    # sklearn forbids max_samples when bootstrap=False; drop max_samples whenever
    # False is in the bootstrap list so GridSearchCV never tries invalid combos.
    if (
        estimator == "RandomForestRegressor"
        and "max_samples" in sanitized
        and False in sanitized.get("bootstrap", [True])
    ):
        removed_params.append("max_samples")
        removed_values.setdefault("max_samples", sanitized.pop("max_samples", []))
        warnings.append(
            "Removed max_samples because bootstrap includes False "
            "(incompatible with sklearn RandomForestRegressor)."
        )

    default_grid = get_fallback_grid(estimator, "default")
    if not sanitized:
        warnings.append("Empty grid after sanitization; using default fallback grid.")
        sanitized = {k: list(v) for k, v in default_grid.items()}

    def shrink_step(grid: dict[str, list[Any]]) -> dict[str, list[Any]]:
        nonlocal shrink_steps
        if count_grid_combinations(grid) <= max_candidates:
            return grid
        param = max(grid, key=lambda p: len(grid[p]))
        if len(grid[param]) <= 1:
            return grid
        values = list(grid[param])
        drop_idx = len(values) // 2
        dropped = values.pop(drop_idx)
        grid[param] = values
        shrink_steps.append(
            f"Removed {param}={dropped!r} to reduce candidates "
            f"(remaining estimate={count_grid_combinations(grid)})."
        )
        return shrink_step(grid)

    candidate_count = count_grid_combinations(sanitized)
    if candidate_count > max_candidates:
        warnings.append(
            f"Grid has {candidate_count} combinations; shrinking to <= {max_candidates}."
        )
        sanitized = shrink_step(dict(sanitized))
        candidate_count = count_grid_combinations(sanitized)

    used_randomized = candidate_count > max_candidates
    if used_randomized:
        warnings.append(
            f"Grid still exceeds cap after shrinking; RandomizedSearchCV will sample "
            f"{max_candidates} candidates (seed={random_seed})."
        )
        candidate_count = max_candidates

    return GridSanitizationResult(
        original_grid=param_grid,
        sanitized_grid=sanitized,
        removed_params=removed_params,
        removed_values=removed_values,
        shrink_steps=shrink_steps,
        candidate_count=candidate_count,
        used_randomized_search=used_randomized,
        warnings=warnings,
    )


def get_fallback_grid(estimator: str, assessment_status: str) -> dict[str, list[Any]]:
    grids = _FALLBACK_GRIDS.get(estimator, _FALLBACK_GRIDS["RandomForestRegressor"])
    if assessment_status == "overfit":
        template = grids["overfit"]
    elif assessment_status == "underfit":
        template = grids["underfit"]
    elif assessment_status == "unstable":
        template = grids["unstable"]
    else:
        template = grids["default"]
    return {k: list(v) for k, v in template.items()}


def model_simplicity_score(estimator: str, params: dict[str, Any]) -> float:
    """Lower score = simpler / more regularized (preferred on ties)."""
    if estimator in ("RandomForestRegressor", "ExtraTreesRegressor"):
        max_depth = params.get("max_depth")
        depth_score = 50.0 if max_depth is None else float(max_depth)
        min_leaf = float(params.get("min_samples_leaf", 1))
        min_split = float(params.get("min_samples_split", 2))
        n_est = float(params.get("n_estimators", 100))
        max_feat = params.get("max_features", "sqrt")
        if isinstance(max_feat, str):
            feat_score = {"sqrt": 0.3, "log2": 0.25}.get(max_feat, 0.5)
        else:
            feat_score = float(max_feat)
        bootstrap = params.get("bootstrap", True)
        max_samples = params.get("max_samples", 1.0)
        reg_score = 0.0 if not bootstrap else (1.0 - float(max_samples or 1.0))
        return depth_score - min_leaf - min_split + feat_score * 10 + n_est * 0.01 + reg_score * 5

    if estimator == "PLSRegression":
        return float(params.get("n_components", 10))

    if estimator == "SVR":
        c_score = float(params.get("C", 1.0))
        eps = float(params.get("epsilon", 0.1))
        return c_score - eps * 10

    if estimator == "KNeighborsRegressor":
        return float(params.get("n_neighbors", 5))

    return 0.0


def params_to_model_config(params: dict[str, Any], base: ModelConfig) -> ModelConfig:
    data = base.model_dump()
    merged_params = {**resolve_params(base), **params}
    if base.estimator == "RandomForestRegressor":
        for key in _RF_DEFAULTS:
            if key in params:
                data[key] = params[key]
        bootstrap = merged_params.get("bootstrap", data.get("bootstrap", True))
        if not bootstrap:
            merged_params["max_samples"] = None
            data["max_samples"] = None
    data["params"] = merged_params
    return ModelConfig(**{k: v for k, v in data.items() if k in ModelConfig.model_fields})
