"""k-nearest-neighbour distance-to-model applicability domain.

Works for both regression and classification, unlike the leverage-based Williams
plot in :mod:`qsar_agent.tools.applicability_domain`, which needs residuals. The
threshold follows the common kNN AD convention: mean plus ``z`` standard
deviations of the training compounds' own mean k-neighbour distance.
"""

from __future__ import annotations

import numpy as np
from sklearn.neighbors import NearestNeighbors

from qsar_agent.schemas.agentic import ADSummary

DEFAULT_K = 5
DEFAULT_Z = 3.0


def _mean_neighbour_distance(distances: np.ndarray) -> np.ndarray:
    return distances.mean(axis=1) if distances.size else np.zeros(len(distances))


def knn_ad_threshold(
    X_train: np.ndarray, k: int = DEFAULT_K, z_factor: float = DEFAULT_Z
) -> tuple[float, np.ndarray, NearestNeighbors]:
    """Fit the domain on training data and return ``(threshold, train_distances, index)``."""
    X_train = np.asarray(X_train, dtype=float)
    n_train = len(X_train)
    if n_train < 2:
        raise ValueError("At least 2 training compounds are required to define a domain.")

    effective_k = max(1, min(k, n_train - 1))
    index = NearestNeighbors(n_neighbors=effective_k + 1).fit(X_train)
    distances, _ = index.kneighbors(X_train)
    # Column 0 is the compound itself.
    train_distances = _mean_neighbour_distance(distances[:, 1:])
    threshold = float(train_distances.mean() + z_factor * train_distances.std(ddof=0))
    return threshold, train_distances, index


def evaluate_applicability_domain(
    X_train: np.ndarray,
    X_test: np.ndarray,
    test_ids: list[str] | None = None,
    k: int = DEFAULT_K,
    z_factor: float = DEFAULT_Z,
) -> tuple[ADSummary, np.ndarray]:
    """Return an :class:`ADSummary` plus the per-test-compound in-domain mask."""
    X_train = np.asarray(X_train, dtype=float)
    X_test = np.asarray(X_test, dtype=float)
    threshold, train_distances, index = knn_ad_threshold(X_train, k=k, z_factor=z_factor)

    effective_k = max(1, min(k, len(X_train)))
    test_distances, _ = index.kneighbors(X_test, n_neighbors=effective_k)
    test_mean_distance = _mean_neighbour_distance(test_distances)
    in_domain = test_mean_distance <= threshold

    ids = test_ids if test_ids is not None else [str(i) for i in range(len(X_test))]
    out_of_domain = [ids[i] for i in range(len(ids)) if not in_domain[i]]

    summary = ADSummary(
        k=effective_k,
        z_factor=z_factor,
        threshold=threshold,
        train_mean_distance=float(train_distances.mean()),
        test_in_domain_count=int(in_domain.sum()),
        test_total_count=int(len(X_test)),
        test_coverage_pct=float(100.0 * in_domain.mean()) if len(X_test) else 0.0,
        out_of_domain_ids=out_of_domain,
    )
    return summary, in_domain


def training_outlier_mask(
    X_train: np.ndarray, k: int = DEFAULT_K, z_factor: float = DEFAULT_Z
) -> np.ndarray:
    """Boolean mask of training compounds beyond the domain threshold.

    Used when the Strategist asks for ``drop_ad_outliers``.
    """
    threshold, train_distances, _ = knn_ad_threshold(X_train, k=k, z_factor=z_factor)
    return train_distances > threshold
