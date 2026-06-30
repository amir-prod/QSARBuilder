"""UMAP-based cluster-aware train/test split."""

from __future__ import annotations

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
from qsar_agent.services.plotting import plot_umap_split
from qsar_agent.tools.mordred_descriptors import META_COLUMNS
from qsar_agent.tools.provisional_preprocessing import provisional_preprocess_for_umap


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


def create_umap_cluster_split(
    descriptor_path: str | Path,
    run_dir: Path,
    test_fraction: float = 0.20,
    random_seed: int = 42,
    umap_config: UMAPConfig | None = None,
    clustering_config: ClusteringConfig | None = None,
) -> SplitResult:
    """
    Create a cluster-aware train/external-test split using UMAP + KMeans.

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
    test_indices: list[int] = []

    for cluster_id, group in umap_df.groupby("cluster"):
        idx = group.index.tolist()
        if len(idx) <= 5:
            warnings.append(
                f"Cluster {cluster_id} has {len(idx)} samples; assigned entirely to training."
            )
            train_indices.extend(idx)
            continue
        train_idx, test_idx = train_test_split(
            idx,
            test_size=test_fraction,
            random_state=random_seed,
            shuffle=True,
        )
        train_indices.extend(train_idx)
        test_indices.extend(test_idx)

    umap_df["split"] = "train"
    umap_df.loc[test_indices, "split"] = "test"

    overlap = set(train_indices) & set(test_indices)
    if overlap:
        raise RuntimeError(f"Train/test overlap detected: {overlap}")

    train_df = df.iloc[train_indices].copy()
    test_df = df.iloc[test_indices].copy()

    train_path = run_dir / "train_set_raw_descriptors.csv"
    test_path = run_dir / "test_set_raw_descriptors.csv"
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

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

    report = {
        "train_count": len(train_df),
        "test_count": len(test_df),
        "test_fraction_target": test_fraction,
        "test_fraction_actual": len(test_df) / len(df),
        "n_clusters": best_n,
        "cluster_sizes": [c.model_dump() for c in cluster_sizes],
        "train_activity_mean": float(train_df["activity"].mean()),
        "train_activity_std": float(train_df["activity"].std()),
        "test_activity_mean": float(test_df["activity"].mean()),
        "test_activity_std": float(test_df["activity"].std()),
        "random_seed": random_seed,
        "umap_config": umap_cfg.model_dump(),
        "clustering_config": cluster_cfg.model_dump(),
        "note": (
            "Activity distributions are reported for diagnostics only; "
            "split is based on descriptor-space clustering, not activity."
        ),
    }
    report_path = run_dir / "split_report.json"
    save_json(report_path, report)

    return SplitResult(
        train_count=len(train_df),
        test_count=len(test_df),
        test_fraction_actual=len(test_df) / len(df),
        n_clusters=best_n,
        cluster_sizes=cluster_sizes,
        train_activity_mean=float(train_df["activity"].mean()),
        train_activity_std=float(train_df["activity"].std()),
        test_activity_mean=float(test_df["activity"].mean()),
        test_activity_std=float(test_df["activity"].std()),
        train_path=str(train_path),
        test_path=str(test_path),
        split_assignments_path=str(assignments_path),
        umap_coordinates_path=str(coords_path),
        umap_plot_png=str(png_path),
        umap_plot_svg=str(svg_path),
        split_report_path=str(report_path),
        warnings=warnings,
    )
