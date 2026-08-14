"""UMAP-based cluster-aware train/validation/test split."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import umap
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.model_selection import train_test_split

from qsar_agent.config import ClusteringConfig, UMAPConfig
from qsar_agent.schemas.split import ClusterInfo, SplitResult
from qsar_agent.services.artifact_manager import save_json
from qsar_agent.services.plotting import plot_sorted_split, plot_umap_split
from qsar_agent.tools.provisional_preprocessing import provisional_preprocess_for_umap

_SMALL_CLUSTER_SIZE = 5
_SMALL_HOLD_OUT_WARN = 5


def find_best_num_clusters(
    embedding: np.ndarray,
    clustering_config: ClusteringConfig,
) -> int:
    """Select cluster count using silhouette minus Davies-Bouldin (from examples)."""
    results = []
    for n_clusters in range(
        clustering_config.min_clusters, clustering_config.max_clusters + 1
    ):
        kmeans = KMeans(n_clusters=n_clusters, random_state=clustering_config.random_state)
        labels = kmeans.fit_predict(embedding)
        if len(set(labels)) < 2:
            continue
        sil = silhouette_score(embedding, labels)
        db = davies_bouldin_score(embedding, labels)
        results.append({"n_clusters": n_clusters, "score": sil - db})
    if not results:
        return clustering_config.min_clusters
    best = max(results, key=lambda r: r["score"])
    return int(best["n_clusters"])


def split_indices_three_way(
    idx: list[int],
    val_fraction: float,
    test_fraction: float,
    random_seed: int,
) -> tuple[list[int], list[int], list[int]]:
    """
    Split indices into train, val, and test.

    Peels test first, then validation from the remainder using
    ``val_relative = val_fraction / (1 - test_fraction)``.
    Clusters (or pools) of size <= 5 go entirely to train.
    """
    indices = list(idx)
    n = len(indices)
    if n <= _SMALL_CLUSTER_SIZE:
        return indices, [], []
    if val_fraction <= 0 or test_fraction <= 0:
        raise ValueError("val_fraction and test_fraction must be in (0, 1)")
    if val_fraction + test_fraction >= 1:
        raise ValueError("val_fraction + test_fraction must be < 1")

    test_size = max(test_fraction, 1.0 / n)
    remaining, test_idx = train_test_split(
        indices,
        test_size=test_size,
        random_state=random_seed,
        shuffle=True,
    )
    remaining = list(remaining)
    test_idx = list(test_idx)

    if len(remaining) <= 1:
        return remaining, [], test_idx

    val_relative = val_fraction / (1.0 - test_fraction)
    n_rem = len(remaining)
    val_size = max(val_relative, 1.0 / n_rem)
    if val_size >= 1.0 - (1.0 / n_rem):
        # Leave at least one compound in train.
        if n_rem < 2:
            return remaining, [], test_idx
        val_size = 1.0 / n_rem

    train_idx, val_idx = train_test_split(
        remaining,
        test_size=val_size,
        random_state=random_seed,
        shuffle=True,
    )
    train_idx = list(train_idx)
    val_idx = list(val_idx)

    _assert_no_overlap(train_idx, val_idx, test_idx)
    if set(train_idx) | set(val_idx) | set(test_idx) != set(indices):
        raise RuntimeError("Three-way split did not assign every index.")
    return train_idx, val_idx, test_idx


def _activity_mean_std(series: pd.Series) -> tuple[float, float]:
    if len(series) == 0:
        return float("nan"), float("nan")
    mean = float(series.mean())
    std = float(series.std()) if len(series) > 1 else 0.0
    return mean, std


def _holdout_size_warnings(
    val_count: int,
    test_count: int,
    warnings: list[str],
) -> None:
    if val_count < _SMALL_HOLD_OUT_WARN:
        warnings.append(
            f"Validation set has only {val_count} compound(s); validation R² will be noisy."
        )
    if test_count < _SMALL_HOLD_OUT_WARN:
        warnings.append(
            f"External test set has only {test_count} compound(s); test metrics will be noisy."
        )


def create_umap_cluster_split(
    descriptor_path: str | Path,
    run_dir: Path,
    test_fraction: float = 0.10,
    random_seed: int = 42,
    umap_config: UMAPConfig | None = None,
    clustering_config: ClusteringConfig | None = None,
    val_fraction: float = 0.10,
) -> SplitResult:
    """
    Create a cluster-aware train/validation/external-test split using UMAP + KMeans.

    UMAP produces a 2D embedding; KMeans clusters that embedding (per examples).
    Split is performed within each cluster to preserve chemical diversity.
    """
    umap_cfg = umap_config or UMAPConfig()
    cluster_cfg = clustering_config or ClusteringConfig()
    warnings: list[str] = []
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(descriptor_path)
    _, X_scaled = provisional_preprocess_for_umap(df)

    reducer = umap.UMAP(
        n_neighbors=umap_cfg.n_neighbors,
        n_components=umap_cfg.n_components,
        min_dist=umap_cfg.min_dist,
        metric=umap_cfg.metric,
        random_state=random_seed,
    )
    embedding = reducer.fit_transform(X_scaled)

    best_n = find_best_num_clusters(embedding, cluster_cfg)
    kmeans = KMeans(n_clusters=best_n, random_state=cluster_cfg.random_state)
    cluster_labels = kmeans.fit_predict(embedding)

    umap_df = pd.DataFrame(
        {
            "compound_id": df["compound_id"].values,
            "original_row_index": df["original_row_index"].values,
            "umap_1": embedding[:, 0],
            "umap_2": embedding[:, 1],
            "cluster": cluster_labels,
        }
    )

    train_indices: list[int] = []
    val_indices: list[int] = []
    test_indices: list[int] = []

    for cluster_id, group in umap_df.groupby("cluster"):
        idx = group.index.tolist()
        if len(idx) <= _SMALL_CLUSTER_SIZE:
            warnings.append(
                f"Cluster {cluster_id} has {len(idx)} samples; assigned entirely to training."
            )
            train_indices.extend(idx)
            continue
        tr, va, te = split_indices_three_way(
            idx, val_fraction, test_fraction, random_seed
        )
        train_indices.extend(tr)
        val_indices.extend(va)
        test_indices.extend(te)

    n_total = len(df)
    need_fallback = (len(test_indices) == 0 or len(val_indices) == 0) and n_total >= 3
    if need_fallback:
        missing = []
        if len(test_indices) == 0:
            missing.append("test")
        if len(val_indices) == 0:
            missing.append("validation")
        warnings.append(
            f"No per-cluster {'/'.join(missing)} compounds were assigned "
            "(all clusters were small or splits were empty). "
            f"Falling back to a global train/val/test split "
            f"(val_fraction={val_fraction}, test_fraction={test_fraction})."
        )
        train_indices, val_indices, test_indices = split_indices_three_way(
            list(range(n_total)), val_fraction, test_fraction, random_seed
        )
    if len(test_indices) == 0:
        raise RuntimeError(
            f"Cannot create an external test set with only {n_total} compound(s)."
        )
    if len(val_indices) == 0:
        raise RuntimeError(
            f"Cannot create a validation set with only {n_total} compound(s)."
        )

    umap_df["split"] = "train"
    umap_df.loc[val_indices, "split"] = "val"
    umap_df.loc[test_indices, "split"] = "test"

    _assert_no_overlap(train_indices, val_indices, test_indices)

    train_df = df.iloc[train_indices].copy()
    val_df = df.iloc[val_indices].copy()
    test_df = df.iloc[test_indices].copy()
    if len(test_df) == 0:
        raise RuntimeError(
            "External test set is empty after splitting; cannot continue preprocessing."
        )
    if len(val_df) == 0:
        raise RuntimeError(
            "Validation set is empty after splitting; cannot continue preprocessing."
        )

    _holdout_size_warnings(len(val_df), len(test_df), warnings)

    train_path, val_path, test_path = _write_raw_split_csvs(
        run_dir, train_df, val_df, test_df
    )

    assignments_path = run_dir / "split_assignments.csv"
    umap_df.to_csv(assignments_path, index=False)

    coords_path = run_dir / "umap_coordinates.csv"
    umap_df.to_csv(coords_path, index=False)

    png_path = run_dir / "umap_split.png"
    svg_path = run_dir / "umap_split.svg"
    plot_umap_split(umap_df, png_path, svg_path)

    cluster_sizes = [
        ClusterInfo(cluster_id=int(cid), size=int(len(g)))
        for cid, g in umap_df.groupby("cluster")
    ]

    train_mean, train_std = _activity_mean_std(train_df["activity"])
    val_mean, val_std = _activity_mean_std(val_df["activity"])
    test_mean, test_std = _activity_mean_std(test_df["activity"])

    report = {
        "train_count": len(train_df),
        "val_count": len(val_df),
        "test_count": len(test_df),
        "val_fraction_target": val_fraction,
        "test_fraction_target": test_fraction,
        "val_fraction_actual": len(val_df) / len(df),
        "test_fraction_actual": len(test_df) / len(df),
        "n_clusters": best_n,
        "cluster_sizes": [c.model_dump() for c in cluster_sizes],
        "train_activity_mean": train_mean,
        "train_activity_std": train_std,
        "val_activity_mean": val_mean,
        "val_activity_std": val_std,
        "test_activity_mean": test_mean,
        "test_activity_std": test_std,
        "random_seed": random_seed,
        "umap_config": umap_cfg.model_dump(),
        "clustering_config": cluster_cfg.model_dump(),
        "split_method": "umap_cluster",
        "warnings": warnings,
        "note": (
            "Activity distributions are reported for diagnostics only; "
            "split is based on descriptor-space clustering, not activity."
        ),
    }
    report_path = run_dir / "split_report.json"
    save_json(report_path, report)

    return SplitResult(
        train_count=len(train_df),
        val_count=len(val_df),
        test_count=len(test_df),
        val_fraction_actual=len(val_df) / len(df),
        test_fraction_actual=len(test_df) / len(df),
        n_clusters=best_n,
        cluster_sizes=cluster_sizes,
        train_activity_mean=train_mean,
        train_activity_std=train_std,
        val_activity_mean=val_mean,
        val_activity_std=val_std,
        test_activity_mean=test_mean,
        test_activity_std=test_std,
        train_path=str(train_path),
        val_path=str(val_path),
        test_path=str(test_path),
        split_assignments_path=str(assignments_path),
        umap_coordinates_path=str(coords_path),
        umap_plot_png=str(png_path),
        umap_plot_svg=str(svg_path),
        split_report_path=str(report_path),
        split_method="umap_cluster",
        warnings=warnings,
    )


def sorted_split_stride(fraction: float) -> int:
    """Stride k such that every k-th ranked compound is a holdout candidate (10% → 10)."""
    if fraction <= 0 or fraction >= 1:
        raise ValueError(f"fraction must be in (0, 1), got {fraction}")
    return max(2, int(round(1.0 / float(fraction))))


def _val_rank_offset(val_stride: int, test_stride: int) -> int:
    """Pick a 1-indexed rank modulus for val that avoids test ranks when possible."""
    if val_stride == test_stride:
        return max(1, val_stride // 2)
    gcd = math.gcd(val_stride, test_stride)
    for off in range(1, val_stride):
        # Collision with test (rank % test_stride == 0) exists iff gcd divides `off`.
        if off % gcd != 0:
            return off
    return max(1, val_stride // 2)


def assign_sorted_split_indices(
    activity: np.ndarray,
    val_fraction: float = 0.10,
    test_fraction: float = 0.10,
) -> tuple[list[int], list[int], list[int], int, int]:
    """
    Sort by activity and assign compounds to train, val, and test.

    Compounds with the minimum or maximum activity always stay in train.
    Test uses every ``test_stride``-th ranked compound. Validation uses every
    ``val_stride``-th ranked compound at an offset that avoids test ranks.
    If a rank would be both val and test, it is assigned to test.
    """
    activity = np.asarray(activity, dtype=float)
    n = len(activity)
    if n < 3:
        raise RuntimeError(
            f"Cannot create train/val/test splits with only {n} compound(s)."
        )

    test_stride = sorted_split_stride(test_fraction)
    val_stride = sorted_split_stride(val_fraction)
    val_offset = _val_rank_offset(val_stride, test_stride)

    order = np.argsort(activity, kind="mergesort")
    min_val = float(np.min(activity))
    max_val = float(np.max(activity))
    protected = set(
        int(i) for i in np.flatnonzero((activity == min_val) | (activity == max_val))
    )

    train_indices: list[int] = []
    val_indices: list[int] = []
    test_indices: list[int] = []
    for rank0, idx in enumerate(order):
        rank = rank0 + 1
        idx = int(idx)
        if idx in protected:
            train_indices.append(idx)
        elif rank % test_stride == 0:
            test_indices.append(idx)
        elif rank % val_stride == val_offset:
            val_indices.append(idx)
        else:
            train_indices.append(idx)

    if not test_indices or not val_indices:
        candidates = [int(i) for i in order if int(i) not in protected]
        if not candidates:
            raise RuntimeError(
                "Cannot create holdout sets because every compound has "
                "the minimum or maximum activity value."
            )
        assigned = set()
        if not test_indices:
            n_test = max(1, min(len(candidates), int(round(n * test_fraction))))
            pick_pos = np.unique(
                np.linspace(0, len(candidates) - 1, n_test).astype(int)
            )
            test_indices = [candidates[int(p)] for p in pick_pos]
            assigned.update(test_indices)
        if not val_indices:
            remaining = [i for i in candidates if i not in assigned]
            if not remaining:
                raise RuntimeError(
                    "Cannot create a validation set after assigning the test set."
                )
            n_val = max(1, min(len(remaining), int(round(n * val_fraction))))
            pick_pos = np.unique(
                np.linspace(0, len(remaining) - 1, n_val).astype(int)
            )
            val_indices = [remaining[int(p)] for p in pick_pos]
            assigned.update(val_indices)
        holdout = set(test_indices) | set(val_indices)
        train_indices = [i for i in range(n) if i not in holdout]

    _assert_no_overlap(train_indices, val_indices, test_indices)
    if set(train_indices) | set(val_indices) | set(test_indices) != set(range(n)):
        raise RuntimeError("Sorted split did not assign every compound.")
    return train_indices, val_indices, test_indices, test_stride, val_stride


def create_sorted_split(
    descriptor_path: str | Path,
    run_dir: Path,
    test_fraction: float = 0.10,
    val_fraction: float = 0.10,
) -> SplitResult:
    """
    Create a train/validation/external-test split by ranking compounds on the target.

    After sorting by activity, every k_test-th compound goes to test and every
    k_val-th (offset) compound goes to validation. Min and max activity stay in train.
    """
    warnings: list[str] = []
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(descriptor_path)
    if "activity" not in df.columns:
        raise ValueError("Descriptor table must include an 'activity' column.")

    activity = df["activity"].to_numpy(dtype=float)
    if np.isnan(activity).any():
        raise ValueError("Activity contains missing values; cannot perform sorted splitting.")

    train_indices, val_indices, test_indices, test_stride, val_stride = (
        assign_sorted_split_indices(activity, val_fraction, test_fraction)
    )

    assignments = pd.DataFrame(
        {
            "compound_id": df["compound_id"].values,
            "original_row_index": df["original_row_index"].values,
            "activity": activity,
            "activity_rank": np.empty(len(df), dtype=int),
            "split": "train",
        }
    )
    order = np.argsort(activity, kind="mergesort")
    assignments.loc[order, "activity_rank"] = np.arange(len(df))
    assignments.loc[val_indices, "split"] = "val"
    assignments.loc[test_indices, "split"] = "test"

    train_df = df.iloc[train_indices].copy()
    val_df = df.iloc[val_indices].copy()
    test_df = df.iloc[test_indices].copy()
    if len(test_df) == 0:
        raise RuntimeError(
            "External test set is empty after splitting; cannot continue preprocessing."
        )
    if len(val_df) == 0:
        raise RuntimeError(
            "Validation set is empty after splitting; cannot continue preprocessing."
        )

    _holdout_size_warnings(len(val_df), len(test_df), warnings)

    train_path, val_path, test_path = _write_raw_split_csvs(
        run_dir, train_df, val_df, test_df
    )

    assignments_path = run_dir / "split_assignments.csv"
    assignments.to_csv(assignments_path, index=False)

    png_path = run_dir / "sorted_split.png"
    svg_path = run_dir / "sorted_split.svg"
    plot_sorted_split(assignments, png_path, svg_path)

    train_mean, train_std = _activity_mean_std(train_df["activity"])
    val_mean, val_std = _activity_mean_std(val_df["activity"])
    test_mean, test_std = _activity_mean_std(test_df["activity"])

    report = {
        "split_method": "sorted",
        "train_count": len(train_df),
        "val_count": len(val_df),
        "test_count": len(test_df),
        "val_fraction_target": val_fraction,
        "test_fraction_target": test_fraction,
        "val_fraction_actual": len(val_df) / len(df),
        "test_fraction_actual": len(test_df) / len(df),
        "test_stride": test_stride,
        "val_stride": val_stride,
        "n_clusters": 0,
        "cluster_sizes": [],
        "train_activity_mean": train_mean,
        "train_activity_std": train_std,
        "val_activity_mean": val_mean,
        "val_activity_std": val_std,
        "test_activity_mean": test_mean,
        "test_activity_std": test_std,
        "train_activity_min": float(train_df["activity"].min()),
        "train_activity_max": float(train_df["activity"].max()),
        "val_activity_min": float(val_df["activity"].min()),
        "val_activity_max": float(val_df["activity"].max()),
        "test_activity_min": float(test_df["activity"].min()),
        "test_activity_max": float(test_df["activity"].max()),
        "warnings": warnings,
        "note": (
            f"Compounds ranked by activity; every {test_stride}th assigned to test, "
            f"every {val_stride}th (offset) assigned to validation. "
            "Compounds with minimum or maximum activity always remain in train. "
            "If a rank would be both val and test, it is assigned to test."
        ),
    }
    report_path = run_dir / "split_report.json"
    save_json(report_path, report)

    return SplitResult(
        train_count=len(train_df),
        val_count=len(val_df),
        test_count=len(test_df),
        val_fraction_actual=len(val_df) / len(df),
        test_fraction_actual=len(test_df) / len(df),
        n_clusters=0,
        cluster_sizes=[],
        train_activity_mean=train_mean,
        train_activity_std=train_std,
        val_activity_mean=val_mean,
        val_activity_std=val_std,
        test_activity_mean=test_mean,
        test_activity_std=test_std,
        train_path=str(train_path),
        val_path=str(val_path),
        test_path=str(test_path),
        split_assignments_path=str(assignments_path),
        umap_coordinates_path="",
        umap_plot_png=str(png_path),
        umap_plot_svg=str(svg_path),
        split_report_path=str(report_path),
        split_method="sorted",
        test_stride=test_stride,
        val_stride=val_stride,
        warnings=warnings,
    )


def create_split(
    descriptor_path: str | Path,
    run_dir: Path,
    test_fraction: float = 0.10,
    random_seed: int = 42,
    umap_config: UMAPConfig | None = None,
    clustering_config: ClusteringConfig | None = None,
    split_method: str = "umap_cluster",
    val_fraction: float = 0.10,
) -> SplitResult:
    """Dispatch to UMAP-cluster or activity-sorted splitting."""
    method = (split_method or "umap_cluster").strip().lower()
    if method == "sorted":
        return create_sorted_split(
            descriptor_path, run_dir, test_fraction, val_fraction=val_fraction
        )
    if method in {"umap_cluster", "umap"}:
        return create_umap_cluster_split(
            descriptor_path,
            run_dir,
            test_fraction,
            random_seed,
            umap_config,
            clustering_config,
            val_fraction=val_fraction,
        )
    raise ValueError(
        f"Unknown split_method {split_method!r}; expected 'umap_cluster' or 'sorted'."
    )


def _assert_no_overlap(
    train_indices: list[int],
    val_indices: list[int],
    test_indices: list[int],
) -> None:
    if set(train_indices) & set(test_indices):
        raise RuntimeError(
            f"Train/test overlap detected: {set(train_indices) & set(test_indices)}"
        )
    if set(train_indices) & set(val_indices):
        raise RuntimeError(
            f"Train/val overlap detected: {set(train_indices) & set(val_indices)}"
        )
    if set(val_indices) & set(test_indices):
        raise RuntimeError(
            f"Val/test overlap detected: {set(val_indices) & set(test_indices)}"
        )


def _write_raw_split_csvs(
    run_dir: Path,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> tuple[Path, Path, Path]:
    train_path = run_dir / "train_set_raw_descriptors.csv"
    val_path = run_dir / "val_set_raw_descriptors.csv"
    test_path = run_dir / "test_set_raw_descriptors.csv"
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)
    return train_path, val_path, test_path
