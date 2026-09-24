"""Task-aware estimator registry for the agentic loop.

The Strategist may only name estimators and hyperparameters that exist here, so
a plan is always executable. Unknown hyperparameters are rejected with an
explanatory error that is fed back to the model for repair.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import ElasticNet, LogisticRegression, Ridge
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR


class AdaptivePLSRegression(PLSRegression):
    """PLS that clamps ``n_components`` to what the fit-time data supports."""

    def fit(self, X, y=None):
        data = np.asarray(X)
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        n_samples, n_features = data.shape
        limit = max(1, min(int(n_features), max(1, int(n_samples) - 1)))
        self.n_components = min(int(self.n_components), limit)
        return super().fit(X, y)

    def predict(self, X):
        return super().predict(X).ravel()


class AdaptiveKNeighborsRegressor(KNeighborsRegressor):
    """kNN regressor that clamps ``n_neighbors`` to the training-set size."""

    def fit(self, X, y):
        self.n_neighbors = min(int(self.n_neighbors), max(1, np.asarray(X).shape[0]))
        return super().fit(X, y)


class AdaptiveKNeighborsClassifier(KNeighborsClassifier):
    """kNN classifier that clamps ``n_neighbors`` to the training-set size."""

    def fit(self, X, y):
        self.n_neighbors = min(int(self.n_neighbors), max(1, np.asarray(X).shape[0]))
        return super().fit(X, y)


_REGRESSORS: dict[str, dict[str, Any]] = {
    "RandomForest": {
        "cls": RandomForestRegressor,
        "defaults": {"n_estimators": 300, "max_depth": None, "min_samples_leaf": 1},
        "description": "Random forest; robust default, handles nonlinearity and many features.",
        "supports_seed": True,
        "parallel": True,
    },
    "ExtraTrees": {
        "cls": ExtraTreesRegressor,
        "defaults": {"n_estimators": 300, "min_samples_leaf": 1},
        "description": "Extremely randomised trees; higher variance reduction than RF.",
        "supports_seed": True,
        "parallel": True,
    },
    "GradientBoosting": {
        "cls": GradientBoostingRegressor,
        "defaults": {"n_estimators": 200, "learning_rate": 0.1, "max_depth": 3},
        "description": "Gradient-boosted trees; strong on structured descriptors.",
        "supports_seed": True,
    },
    "Ridge": {
        "cls": Ridge,
        "defaults": {"alpha": 1.0},
        "description": "L2-regularised linear model; a good low-variance baseline.",
        "supports_seed": False,
    },
    "ElasticNet": {
        "cls": ElasticNet,
        "defaults": {"alpha": 0.1, "l1_ratio": 0.5, "max_iter": 5000},
        "description": "L1+L2 linear model; performs implicit feature selection.",
        "supports_seed": True,
    },
    "SVR": {
        "cls": SVR,
        "defaults": {"C": 1.0, "gamma": "scale", "kernel": "rbf"},
        "description": "Support-vector regression; needs scaled features.",
        "supports_seed": False,
    },
    "KNN": {
        "cls": AdaptiveKNeighborsRegressor,
        "defaults": {"n_neighbors": 5, "weights": "distance"},
        "description": "k-nearest-neighbour regression; a pure similarity baseline.",
        "supports_seed": False,
        "parallel": True,
    },
    "PLS": {
        "cls": AdaptivePLSRegression,
        "defaults": {"n_components": 5, "scale": False},
        "description": "Partial least squares; classical QSAR choice for collinear descriptors.",
        "supports_seed": False,
    },
}

_CLASSIFIERS: dict[str, dict[str, Any]] = {
    "RandomForest": {
        "cls": RandomForestClassifier,
        "defaults": {"n_estimators": 300, "max_depth": None, "min_samples_leaf": 1},
        "description": "Random forest classifier; robust default.",
        "supports_seed": True,
        "parallel": True,
    },
    "ExtraTrees": {
        "cls": ExtraTreesClassifier,
        "defaults": {"n_estimators": 300, "min_samples_leaf": 1},
        "description": "Extremely randomised tree classifier.",
        "supports_seed": True,
        "parallel": True,
    },
    "GradientBoosting": {
        "cls": GradientBoostingClassifier,
        "defaults": {"n_estimators": 200, "learning_rate": 0.1, "max_depth": 3},
        "description": "Gradient-boosted classification trees.",
        "supports_seed": True,
    },
    "LogisticRegression": {
        "cls": LogisticRegression,
        "defaults": {"C": 1.0, "max_iter": 2000},
        "description": "Regularised logistic regression; interpretable linear baseline.",
        "supports_seed": True,
    },
    "SVC": {
        "cls": SVC,
        "defaults": {"C": 1.0, "gamma": "scale", "kernel": "rbf"},
        "description": (
            "Support-vector classifier; ROC-AUC uses its decision-function margins "
            "rather than calibrated probabilities."
        ),
        "supports_seed": True,
    },
    "KNN": {
        "cls": AdaptiveKNeighborsClassifier,
        "defaults": {"n_neighbors": 5, "weights": "distance"},
        "description": "k-nearest-neighbour classifier.",
        "supports_seed": False,
        "parallel": True,
    },
}

_ZOO = {"regression": _REGRESSORS, "classification": _CLASSIFIERS}

#: Estimators that accept ``class_weight`` for imbalance handling.
CLASS_WEIGHT_CAPABLE = {"RandomForest", "ExtraTrees", "LogisticRegression", "SVC"}


def list_estimators(task: str) -> list[str]:
    return sorted(_zoo_for(task))


def _zoo_for(task: str) -> dict[str, dict[str, Any]]:
    try:
        return _ZOO[task]
    except KeyError:
        raise ValueError(f"Unknown task: {task!r}") from None


def describe_zoo(task: str) -> list[dict[str, Any]]:
    """Machine-readable catalogue handed to the LLM as a tool result."""
    return [
        {
            "name": name,
            "description": spec["description"],
            "default_hyperparameters": dict(spec["defaults"]),
            "tunable_hyperparameters": sorted(_tunable_params(spec["cls"])),
        }
        for name, spec in sorted(_zoo_for(task).items())
    ]


def default_hyperparameters(task: str, estimator: str) -> dict[str, Any]:
    return dict(_lookup(task, estimator)["defaults"])


def _lookup(task: str, estimator: str) -> dict[str, Any]:
    zoo = _zoo_for(task)
    if estimator not in zoo:
        raise ValueError(
            f"Unknown {task} estimator {estimator!r}. Available: {', '.join(sorted(zoo))}"
        )
    return zoo[estimator]


def _tunable_params(cls: type) -> set[str]:
    params = set(cls().get_params().keys())
    return params - {"n_jobs", "verbose", "random_state", "warm_start"}


def validate_hyperparameters(task: str, estimator: str, params: dict[str, Any]) -> None:
    """Raise ``ValueError`` naming any hyperparameter the estimator does not accept."""
    spec = _lookup(task, estimator)
    accepted = set(spec["cls"]().get_params().keys())
    unknown = sorted(set(params) - accepted)
    if unknown:
        raise ValueError(
            f"{estimator} does not accept hyperparameter(s) {unknown}. "
            f"Accepted: {', '.join(sorted(accepted))}"
        )
    if "class_weight" in params and estimator not in CLASS_WEIGHT_CAPABLE:
        raise ValueError(f"{estimator} does not support class_weight.")


def build_estimator(
    task: str,
    estimator: str,
    hyperparameters: dict[str, Any] | None = None,
    random_state: int = 42,
    n_jobs: int = -1,
) -> BaseEstimator:
    """Instantiate an estimator, merging user hyperparameters over the defaults."""
    spec = _lookup(task, estimator)
    params = dict(spec["defaults"])
    params.update(hyperparameters or {})
    validate_hyperparameters(task, estimator, params)

    cls = spec["cls"]
    accepted = set(cls().get_params().keys())
    if spec["supports_seed"] and "random_state" in accepted:
        params.setdefault("random_state", random_state)
    # Only estimators that genuinely parallelise get n_jobs; sklearn warns on the
    # ones where it is a no-op (e.g. LogisticRegression since 1.8).
    if spec.get("parallel") and "n_jobs" in accepted:
        params.setdefault("n_jobs", n_jobs)
    return cls(**params)


def supports_proba(model: BaseEstimator) -> bool:
    return hasattr(model, "predict_proba") or hasattr(model, "decision_function")


def decision_scores(model: BaseEstimator, X) -> np.ndarray | None:
    """Probabilities when available, otherwise decision-function margins."""
    if hasattr(model, "predict_proba"):
        proba = np.asarray(model.predict_proba(X))
        return proba[:, -1] if proba.ndim == 2 and proba.shape[1] == 2 else proba
    if hasattr(model, "decision_function"):
        return np.asarray(model.decision_function(X))
    return None
