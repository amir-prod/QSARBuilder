"""Schema-driven model catalog extensions (sklearn + optional boosters)."""

from __future__ import annotations

from typing import Any

from qsar_agent.schemas.agentic import ModelSpecification

# Defaults / spaces / grids for newly added estimators
_HIST_GB_DEFAULTS: dict[str, Any] = {
    "max_depth": 6,
    "learning_rate": 0.1,
    "max_iter": 100,
    "min_samples_leaf": 20,
    "l2_regularization": 0.0,
}

_GB_DEFAULTS: dict[str, Any] = {
    "n_estimators": 100,
    "max_depth": 3,
    "learning_rate": 0.1,
    "subsample": 1.0,
    "min_samples_leaf": 1,
}

_ADA_DEFAULTS: dict[str, Any] = {
    "n_estimators": 50,
    "learning_rate": 1.0,
    "loss": "linear",
}

_ELASTIC_DEFAULTS: dict[str, Any] = {
    "alpha": 1.0,
    "l1_ratio": 0.5,
    "max_iter": 2000,
}

_RIDGE_DEFAULTS: dict[str, Any] = {
    "alpha": 1.0,
}

_XGB_DEFAULTS: dict[str, Any] = {
    "n_estimators": 100,
    "max_depth": 4,
    "learning_rate": 0.1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_lambda": 1.0,
}

_CAT_DEFAULTS: dict[str, Any] = {
    "iterations": 100,
    "depth": 4,
    "learning_rate": 0.1,
    "verbose": False,
}

_LGBM_DEFAULTS: dict[str, Any] = {
    "n_estimators": 100,
    "max_depth": 4,
    "learning_rate": 0.1,
    "subsample": 0.8,
    "verbosity": -1,
}

EXTENDED_DEFAULT_PARAMS: dict[str, dict[str, Any]] = {
    "HistGradientBoostingRegressor": _HIST_GB_DEFAULTS,
    "GradientBoostingRegressor": _GB_DEFAULTS,
    "AdaBoostRegressor": _ADA_DEFAULTS,
    "ElasticNet": _ELASTIC_DEFAULTS,
    "Ridge": _RIDGE_DEFAULTS,
    "XGBRegressor": _XGB_DEFAULTS,
    "CatBoostRegressor": _CAT_DEFAULTS,
    "LGBMRegressor": _LGBM_DEFAULTS,
}

EXTENDED_SPACES: dict[str, dict[str, set[Any]]] = {
    "HistGradientBoostingRegressor": {
        "max_depth": set(range(2, 16)) | {None},
        "learning_rate": {0.01, 0.05, 0.1, 0.2},
        "max_iter": {50, 100, 200, 300},
        "min_samples_leaf": set(range(5, 51)),
        "l2_regularization": {0.0, 0.1, 1.0, 10.0},
    },
    "GradientBoostingRegressor": {
        "n_estimators": set(range(50, 401)),
        "max_depth": set(range(2, 9)),
        "learning_rate": {0.01, 0.05, 0.1, 0.2},
        "subsample": {0.6, 0.8, 1.0},
        "min_samples_leaf": set(range(1, 21)),
    },
    "AdaBoostRegressor": {
        "n_estimators": set(range(20, 201)),
        "learning_rate": {0.1, 0.5, 1.0, 1.5},
        "loss": {"linear", "square", "exponential"},
    },
    "ElasticNet": {
        "alpha": {0.01, 0.1, 0.5, 1.0, 5.0, 10.0},
        "l1_ratio": {0.1, 0.3, 0.5, 0.7, 0.9},
        "max_iter": {1000, 2000, 5000},
    },
    "Ridge": {
        "alpha": {0.01, 0.1, 1.0, 10.0, 100.0, 1000.0},
    },
    "XGBRegressor": {
        "n_estimators": set(range(50, 401)),
        "max_depth": set(range(2, 9)),
        "learning_rate": {0.01, 0.05, 0.1, 0.2},
        "subsample": {0.6, 0.8, 1.0},
        "colsample_bytree": {0.6, 0.8, 1.0},
        "reg_lambda": {0.1, 1.0, 5.0, 10.0},
    },
    "CatBoostRegressor": {
        "iterations": {50, 100, 200, 300},
        "depth": set(range(2, 9)),
        "learning_rate": {0.01, 0.05, 0.1, 0.2},
    },
    "LGBMRegressor": {
        "n_estimators": set(range(50, 401)),
        "max_depth": set(range(2, 9)),
        "learning_rate": {0.01, 0.05, 0.1, 0.2},
        "subsample": {0.6, 0.8, 1.0},
    },
}

EXTENDED_FALLBACK_GRIDS: dict[str, dict[str, dict[str, list[Any]]]] = {
    "HistGradientBoostingRegressor": {
        "overfit": {
            "max_depth": [2, 3, 4],
            "learning_rate": [0.05, 0.1],
            "max_iter": [50, 100],
            "min_samples_leaf": [20, 30],
            "l2_regularization": [1.0, 10.0],
        },
        "underfit": {
            "max_depth": [6, 10, None],
            "learning_rate": [0.1, 0.2],
            "max_iter": [200, 300],
            "min_samples_leaf": [5, 10],
            "l2_regularization": [0.0, 0.1],
        },
        "unstable": {
            "max_depth": [3, 5],
            "learning_rate": [0.05, 0.1],
            "max_iter": [100, 200],
            "min_samples_leaf": [15, 25],
            "l2_regularization": [0.1, 1.0],
        },
        "default": {
            "max_depth": [3, 6],
            "learning_rate": [0.05, 0.1],
            "max_iter": [100, 200],
            "min_samples_leaf": [10, 20],
            "l2_regularization": [0.0, 1.0],
        },
    },
    "GradientBoostingRegressor": {
        "overfit": {
            "n_estimators": [50, 100],
            "max_depth": [2, 3],
            "learning_rate": [0.05, 0.1],
            "subsample": [0.6, 0.8],
            "min_samples_leaf": [2, 5],
        },
        "underfit": {
            "n_estimators": [200, 300],
            "max_depth": [4, 6],
            "learning_rate": [0.1, 0.2],
            "subsample": [0.8, 1.0],
            "min_samples_leaf": [1, 2],
        },
        "unstable": {
            "n_estimators": [100, 200],
            "max_depth": [2, 3],
            "learning_rate": [0.05, 0.1],
            "subsample": [0.8],
            "min_samples_leaf": [2, 4],
        },
        "default": {
            "n_estimators": [100, 200],
            "max_depth": [2, 3],
            "learning_rate": [0.05, 0.1],
            "subsample": [0.8, 1.0],
            "min_samples_leaf": [1, 2],
        },
    },
    "AdaBoostRegressor": {
        "overfit": {"n_estimators": [20, 50], "learning_rate": [0.1, 0.5], "loss": ["linear"]},
        "underfit": {"n_estimators": [100, 200], "learning_rate": [1.0, 1.5], "loss": ["linear", "square"]},
        "unstable": {"n_estimators": [50, 100], "learning_rate": [0.5, 1.0], "loss": ["linear"]},
        "default": {"n_estimators": [50, 100], "learning_rate": [0.5, 1.0], "loss": ["linear"]},
    },
    "ElasticNet": {
        "overfit": {"alpha": [1.0, 5.0, 10.0], "l1_ratio": [0.3, 0.5, 0.7], "max_iter": [2000]},
        "underfit": {"alpha": [0.01, 0.1], "l1_ratio": [0.1, 0.5, 0.9], "max_iter": [2000]},
        "unstable": {"alpha": [0.5, 1.0, 5.0], "l1_ratio": [0.5], "max_iter": [2000]},
        "default": {"alpha": [0.1, 1.0, 5.0], "l1_ratio": [0.3, 0.5, 0.7], "max_iter": [2000]},
    },
    "Ridge": {
        "overfit": {"alpha": [10.0, 100.0, 1000.0]},
        "underfit": {"alpha": [0.01, 0.1, 1.0]},
        "unstable": {"alpha": [1.0, 10.0, 100.0]},
        "default": {"alpha": [0.1, 1.0, 10.0, 100.0]},
    },
    "XGBRegressor": {
        "overfit": {
            "n_estimators": [50, 100],
            "max_depth": [2, 3],
            "learning_rate": [0.05, 0.1],
            "subsample": [0.6, 0.8],
            "colsample_bytree": [0.6, 0.8],
            "reg_lambda": [5.0, 10.0],
        },
        "underfit": {
            "n_estimators": [200, 300],
            "max_depth": [5, 7],
            "learning_rate": [0.1, 0.2],
            "subsample": [0.8, 1.0],
            "colsample_bytree": [0.8, 1.0],
            "reg_lambda": [0.1, 1.0],
        },
        "unstable": {
            "n_estimators": [100, 200],
            "max_depth": [3, 4],
            "learning_rate": [0.05, 0.1],
            "subsample": [0.8],
            "colsample_bytree": [0.8],
            "reg_lambda": [1.0, 5.0],
        },
        "default": {
            "n_estimators": [100, 200],
            "max_depth": [3, 4],
            "learning_rate": [0.05, 0.1],
            "subsample": [0.8],
            "colsample_bytree": [0.8],
            "reg_lambda": [1.0],
        },
    },
    "CatBoostRegressor": {
        "overfit": {"iterations": [50, 100], "depth": [2, 3], "learning_rate": [0.05, 0.1]},
        "underfit": {"iterations": [200, 300], "depth": [5, 6], "learning_rate": [0.1, 0.2]},
        "unstable": {"iterations": [100, 200], "depth": [3, 4], "learning_rate": [0.05, 0.1]},
        "default": {"iterations": [100, 200], "depth": [3, 4], "learning_rate": [0.05, 0.1]},
    },
    "LGBMRegressor": {
        "overfit": {
            "n_estimators": [50, 100],
            "max_depth": [2, 3],
            "learning_rate": [0.05, 0.1],
            "subsample": [0.6, 0.8],
        },
        "underfit": {
            "n_estimators": [200, 300],
            "max_depth": [5, 7],
            "learning_rate": [0.1, 0.2],
            "subsample": [0.8, 1.0],
        },
        "unstable": {
            "n_estimators": [100, 200],
            "max_depth": [3, 4],
            "learning_rate": [0.05, 0.1],
            "subsample": [0.8],
        },
        "default": {
            "n_estimators": [100, 200],
            "max_depth": [3, 4],
            "learning_rate": [0.05, 0.1],
            "subsample": [0.8],
        },
    },
}


def _probe_optional(module_name: str, attr: str) -> tuple[bool, str | None]:
    try:
        mod = __import__(module_name, fromlist=[attr])
        getattr(mod, attr)
        return True, None
    except Exception:
        return False, module_name


def build_model_specifications() -> dict[str, ModelSpecification]:
    xgb_ok, xgb_dep = _probe_optional("xgboost", "XGBRegressor")
    cat_ok, cat_dep = _probe_optional("catboost", "CatBoostRegressor")
    lgb_ok, lgb_dep = _probe_optional("lightgbm", "LGBMRegressor")

    specs = [
        ModelSpecification(
            estimator_name="RandomForestRegressor",
            display_name="Random Forest",
            family="bagging",
            import_path="sklearn.ensemble.RandomForestRegressor",
            available=True,
            requires_scaling=False,
            default_parameters={},
            computational_cost="medium",
            interpretability_level="medium",
            expected_strengths=["nonlinear", "robust"],
            expected_limitations=["can overfit small n"],
        ),
        ModelSpecification(
            estimator_name="ExtraTreesRegressor",
            display_name="Extra Trees",
            family="bagging",
            import_path="sklearn.ensemble.ExtraTreesRegressor",
            available=True,
            requires_scaling=False,
            computational_cost="medium",
            interpretability_level="medium",
        ),
        ModelSpecification(
            estimator_name="HistGradientBoostingRegressor",
            display_name="Histogram Gradient Boosting",
            family="boosting",
            import_path="sklearn.ensemble.HistGradientBoostingRegressor",
            available=True,
            requires_scaling=False,
            computational_cost="medium",
            interpretability_level="low",
            default_parameters=dict(_HIST_GB_DEFAULTS),
            bounded_hpo_space={k: list(v) for k, v in EXTENDED_FALLBACK_GRIDS["HistGradientBoostingRegressor"]["default"].items()},
        ),
        ModelSpecification(
            estimator_name="GradientBoostingRegressor",
            display_name="Gradient Boosting",
            family="boosting",
            import_path="sklearn.ensemble.GradientBoostingRegressor",
            available=True,
            requires_scaling=False,
            computational_cost="medium",
            interpretability_level="low",
            default_parameters=dict(_GB_DEFAULTS),
            bounded_hpo_space={k: list(v) for k, v in EXTENDED_FALLBACK_GRIDS["GradientBoostingRegressor"]["default"].items()},
        ),
        ModelSpecification(
            estimator_name="AdaBoostRegressor",
            display_name="AdaBoost",
            family="boosting",
            import_path="sklearn.ensemble.AdaBoostRegressor",
            available=True,
            requires_scaling=False,
            computational_cost="medium",
            interpretability_level="low",
            default_parameters=dict(_ADA_DEFAULTS),
            bounded_hpo_space={k: list(v) for k, v in EXTENDED_FALLBACK_GRIDS["AdaBoostRegressor"]["default"].items()},
        ),
        ModelSpecification(
            estimator_name="SVR",
            display_name="Support Vector Regression",
            family="kernel",
            import_path="sklearn.svm.SVR",
            available=True,
            requires_scaling=True,
            computational_cost="medium",
            interpretability_level="low",
            maximum_recommended_features_ratio=5.0,
        ),
        ModelSpecification(
            estimator_name="KNeighborsRegressor",
            display_name="k-Nearest Neighbors",
            family="neighbor",
            import_path="sklearn.neighbors.KNeighborsRegressor",
            available=True,
            requires_scaling=True,
            computational_cost="low",
            interpretability_level="medium",
            minimum_training_samples=5,
        ),
        ModelSpecification(
            estimator_name="ElasticNet",
            display_name="Elastic Net",
            family="linear",
            import_path="sklearn.linear_model.ElasticNet",
            available=True,
            requires_scaling=True,
            computational_cost="low",
            interpretability_level="high",
            default_parameters=dict(_ELASTIC_DEFAULTS),
            bounded_hpo_space={k: list(v) for k, v in EXTENDED_FALLBACK_GRIDS["ElasticNet"]["default"].items()},
        ),
        ModelSpecification(
            estimator_name="Ridge",
            display_name="Ridge Regression",
            family="linear",
            import_path="sklearn.linear_model.Ridge",
            available=True,
            requires_scaling=True,
            computational_cost="low",
            interpretability_level="high",
            default_parameters=dict(_RIDGE_DEFAULTS),
            bounded_hpo_space={k: list(v) for k, v in EXTENDED_FALLBACK_GRIDS["Ridge"]["default"].items()},
        ),
        ModelSpecification(
            estimator_name="PLSRegression",
            display_name="Partial Least Squares",
            family="latent_variable",
            import_path="sklearn.cross_decomposition.PLSRegression",
            available=True,
            requires_scaling=True,
            computational_cost="low",
            interpretability_level="medium",
            maximum_recommended_features_ratio=3.0,
        ),
        ModelSpecification(
            estimator_name="XGBRegressor",
            display_name="XGBoost",
            family="boosting",
            import_path="xgboost.XGBRegressor",
            available=xgb_ok,
            missing_dependency=None if xgb_ok else xgb_dep,
            requires_scaling=False,
            computational_cost="high",
            interpretability_level="low",
            default_parameters=dict(_XGB_DEFAULTS),
            bounded_hpo_space={k: list(v) for k, v in EXTENDED_FALLBACK_GRIDS["XGBRegressor"]["default"].items()},
        ),
        ModelSpecification(
            estimator_name="CatBoostRegressor",
            display_name="CatBoost",
            family="boosting",
            import_path="catboost.CatBoostRegressor",
            available=cat_ok,
            missing_dependency=None if cat_ok else cat_dep,
            requires_scaling=False,
            computational_cost="high",
            interpretability_level="low",
            default_parameters=dict(_CAT_DEFAULTS),
            bounded_hpo_space={k: list(v) for k, v in EXTENDED_FALLBACK_GRIDS["CatBoostRegressor"]["default"].items()},
        ),
        ModelSpecification(
            estimator_name="LGBMRegressor",
            display_name="LightGBM",
            family="boosting",
            import_path="lightgbm.LGBMRegressor",
            available=lgb_ok,
            missing_dependency=None if lgb_ok else lgb_dep,
            requires_scaling=False,
            computational_cost="high",
            interpretability_level="low",
            default_parameters=dict(_LGBM_DEFAULTS),
            bounded_hpo_space={k: list(v) for k, v in EXTENDED_FALLBACK_GRIDS["LGBMRegressor"]["default"].items()},
        ),
    ]
    return {s.estimator_name: s for s in specs}


def build_extended_estimator(estimator: str, params: dict[str, Any], *, random_state: int, n_jobs: int):
    """Construct newly catalogued estimators. Raises ValueError if unknown here."""
    if estimator == "HistGradientBoostingRegressor":
        from sklearn.ensemble import HistGradientBoostingRegressor

        return HistGradientBoostingRegressor(
            max_depth=params.get("max_depth"),
            learning_rate=params.get("learning_rate", 0.1),
            max_iter=params.get("max_iter", 100),
            min_samples_leaf=params.get("min_samples_leaf", 20),
            l2_regularization=params.get("l2_regularization", 0.0),
            random_state=random_state,
        )
    if estimator == "GradientBoostingRegressor":
        from sklearn.ensemble import GradientBoostingRegressor

        return GradientBoostingRegressor(
            n_estimators=params.get("n_estimators", 100),
            max_depth=params.get("max_depth", 3),
            learning_rate=params.get("learning_rate", 0.1),
            subsample=params.get("subsample", 1.0),
            min_samples_leaf=params.get("min_samples_leaf", 1),
            random_state=random_state,
        )
    if estimator == "AdaBoostRegressor":
        from sklearn.ensemble import AdaBoostRegressor

        return AdaBoostRegressor(
            n_estimators=params.get("n_estimators", 50),
            learning_rate=params.get("learning_rate", 1.0),
            loss=params.get("loss", "linear"),
            random_state=random_state,
        )
    if estimator == "ElasticNet":
        from sklearn.linear_model import ElasticNet

        return ElasticNet(
            alpha=params.get("alpha", 1.0),
            l1_ratio=params.get("l1_ratio", 0.5),
            max_iter=params.get("max_iter", 2000),
            random_state=random_state,
        )
    if estimator == "Ridge":
        from sklearn.linear_model import Ridge

        return Ridge(alpha=params.get("alpha", 1.0), random_state=random_state)
    if estimator == "XGBRegressor":
        from xgboost import XGBRegressor

        return XGBRegressor(
            n_estimators=params.get("n_estimators", 100),
            max_depth=params.get("max_depth", 4),
            learning_rate=params.get("learning_rate", 0.1),
            subsample=params.get("subsample", 0.8),
            colsample_bytree=params.get("colsample_bytree", 0.8),
            reg_lambda=params.get("reg_lambda", 1.0),
            random_state=random_state,
            n_jobs=n_jobs,
            verbosity=0,
        )
    if estimator == "CatBoostRegressor":
        from catboost import CatBoostRegressor

        return CatBoostRegressor(
            iterations=params.get("iterations", 100),
            depth=params.get("depth", 4),
            learning_rate=params.get("learning_rate", 0.1),
            random_seed=random_state,
            verbose=False,
        )
    if estimator == "LGBMRegressor":
        from lightgbm import LGBMRegressor

        return LGBMRegressor(
            n_estimators=params.get("n_estimators", 100),
            max_depth=params.get("max_depth", 4),
            learning_rate=params.get("learning_rate", 0.1),
            subsample=params.get("subsample", 0.8),
            random_state=random_state,
            n_jobs=n_jobs,
            verbosity=-1,
        )
    raise ValueError(f"Not an extended catalog estimator: {estimator}")
