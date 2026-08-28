"""Additional feature-selection methods used by the modeling-improvement agent.

Search always uses the training partition. Callers must evaluate selected
subsets on untouched outer folds or the development validation holdout.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
from sklearn.feature_selection import RFE, SelectFromModel, SelectKBest, mutual_info_regression
from sklearn.linear_model import ElasticNetCV
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression

from qsar_agent.config import ModelConfig
from qsar_agent.services import build_estimator
from qsar_agent.services.artifact_manager import save_json
from qsar_agent.tools.descriptor_calculation import META_COLUMNS

MethodName = Literal[
    "rfe",
    "elastic_net",
    "mutual_information",
    "model_embedded",
    "stability_selection",
    "pca",
    "pls",
]


def _xy(train_path: str | Path) -> tuple[pd.DataFrame, np.ndarray, list[str]]:
    df = pd.read_csv(train_path)
    names = [c for c in df.columns if c not in META_COLUMNS]
    return df[names], df["activity"].values.ravel(), names


def select_features(
    train_path: str | Path,
    method: MethodName,
    n_features: int,
    out_dir: Path,
    model_config: ModelConfig | None = None,
    random_seed: int = 42,
    n_stability_draws: int = 8,
    allow_latent: bool = False,
) -> dict[str, Any]:
    """Fit a selector on **train only** and return selected feature names."""
    if method in {"pca", "pls"} and not allow_latent:
        raise ValueError("Latent-component representations are not permitted.")
    X, y, names = _xy(train_path)
    n_features = max(1, min(int(n_features), X.shape[1]))
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    estimator = build_estimator(model_config)
    selected: list[str]
    extra: dict[str, Any] = {"method": method, "n_features": n_features}

    if method == "rfe":
        rfe = RFE(estimator=estimator, n_features_to_select=n_features, step=1)
        rfe.fit(X, y)
        selected = [names[i] for i, flag in enumerate(rfe.support_) if flag]
    elif method == "elastic_net":
        enet = ElasticNetCV(
            l1_ratio=[0.1, 0.5, 0.9],
            cv=min(5, max(2, len(y) // 3)),
            random_state=random_seed,
            max_iter=5000,
        )
        selector = SelectFromModel(enet, max_features=n_features)
        selector.fit(X, y)
        mask = selector.get_support()
        selected = [names[i] for i, flag in enumerate(mask) if flag]
        if len(selected) < 1:
            coef = np.abs(getattr(selector.estimator_, "coef_", np.ones(len(names))))
            order = np.argsort(-coef)[:n_features]
            selected = [names[i] for i in order]
    elif method == "mutual_information":
        kbest = SelectKBest(mutual_info_regression, k=n_features)
        kbest.fit(X.fillna(0.0), y)
        mask = kbest.get_support()
        selected = [names[i] for i, flag in enumerate(mask) if flag]
        extra["scores"] = {
            names[i]: float(kbest.scores_[i]) for i in range(len(names)) if np.isfinite(kbest.scores_[i])
        }
    elif method == "model_embedded":
        estimator.fit(X, y)
        selector = SelectFromModel(estimator, max_features=n_features, prefit=True)
        mask = selector.get_support()
        selected = [names[i] for i, flag in enumerate(mask) if flag]
        if not selected:
            importances = getattr(estimator, "feature_importances_", None)
            if importances is None:
                selected = names[:n_features]
            else:
                order = np.argsort(-np.asarray(importances))[:n_features]
                selected = [names[i] for i in order]
    elif method == "stability_selection":
        rng = np.random.RandomState(random_seed)
        freq = np.zeros(len(names), dtype=float)
        n = len(y)
        draws = max(2, int(n_stability_draws))
        for _ in range(draws):
            idx = rng.choice(n, size=max(5, n // 2), replace=False)
            enet = ElasticNetCV(
                l1_ratio=0.5,
                cv=min(3, max(2, len(idx) // 4)),
                random_state=int(rng.randint(0, 10_000)),
                max_iter=4000,
            )
            enet.fit(X.iloc[idx], y[idx])
            coef = np.abs(enet.coef_)
            if coef.sum() == 0:
                continue
            top = np.argsort(-coef)[:n_features]
            freq[top] += 1
        freq /= draws
        order = np.argsort(-freq)[:n_features]
        selected = [names[i] for i in order]
        extra["selection_frequency"] = {names[i]: float(freq[i]) for i in range(len(names)) if freq[i] > 0}
    elif method == "pca":
        pca = PCA(n_components=n_features, random_state=random_seed)
        pca.fit(X.fillna(0.0))
        # Represent components as synthetic feature names; loadings stored for audit.
        selected = [f"pca_component_{i+1}" for i in range(n_features)]
        extra["explained_variance_ratio"] = [float(v) for v in pca.explained_variance_ratio_]
        extra["loadings"] = pca.components_.tolist()
        extra["source_features"] = names
    elif method == "pls":
        n_comp = min(n_features, max(1, X.shape[1]), max(1, len(y) - 1))
        pls = PLSRegression(n_components=n_comp, scale=False)
        pls.fit(X.fillna(0.0), y)
        selected = [f"pls_component_{i+1}" for i in range(n_comp)]
        extra["x_weights"] = np.asarray(pls.x_weights_).tolist()
        extra["source_features"] = names
    else:
        raise ValueError(f"Unknown feature-selection method: {method}")

    selected = list(dict.fromkeys(selected))
    save_json(out_dir / "selected_features.json", {"selected_features": selected, **extra})
    return {"selected_features": selected, **extra}
