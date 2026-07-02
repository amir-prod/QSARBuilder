"""Plotting utilities for QSAR Agent."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def save_figure(fig: plt.Figure, png_path: Path, svg_path: Path, dpi: int = 300) -> None:
    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, dpi=dpi, bbox_inches="tight")
    fig.savefig(svg_path, format="svg", bbox_inches="tight")
    plt.close(fig)


def plot_umap_split(
    umap_df,
    png_path: Path,
    svg_path: Path,
    title: str = "UMAP Cluster-Aware Train/Test Split",
) -> None:
    fig, ax = plt.subplots(figsize=(10, 8))
    clusters = sorted(umap_df["cluster"].unique())
    colors = sns.color_palette("tab10", n_colors=max(len(clusters), 1))
    cluster_colors = {c: colors[i % len(colors)] for i, c in enumerate(clusters)}

    for split, marker in [("train", "o"), ("test", "^")]:
        subset = umap_df[umap_df["split"] == split]
        for cluster_id in clusters:
            mask = subset["cluster"] == cluster_id
            if mask.sum() == 0:
                continue
            ax.scatter(
                subset.loc[mask, "umap_1"],
                subset.loc[mask, "umap_2"],
                c=[cluster_colors[cluster_id]],
                marker=marker,
                s=50,
                alpha=0.7,
                edgecolors="black",
                linewidths=0.3,
                label=f"Cluster {cluster_id} ({split})",
            )

    ax.set_xlabel("UMAP Dimension 1")
    ax.set_ylabel("UMAP Dimension 2")
    ax.set_title(title)
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
    fig.tight_layout()
    save_figure(fig, png_path, svg_path)


def plot_sfs_r2(
    results_df,
    png_path: Path,
    svg_path: Path,
    selected_count: int | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    x = results_df["n_features"]
    ax.plot(x, results_df["mean_train_r2"], "o-", label="Mean training R²", color="blue")
    ax.errorbar(
        x,
        results_df["mean_cv_r2"],
        yerr=results_df["std_cv_r2"],
        fmt="s-",
        label="Mean CV R²",
        color="green",
        capsize=3,
    )
    if selected_count is not None:
        ax.axvline(selected_count, color="red", linestyle="--", label=f"Selected ({selected_count})")
    ax.set_xlabel("Number of descriptors")
    ax.set_ylabel("R²")
    ax.set_title("Sequential Feature Selection: R² vs Descriptor Count")
    ax.set_xticks(range(int(x.min()), int(x.max()) + 1))
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    save_figure(fig, png_path, svg_path)


def plot_ga_convergence(history_df, png_path: Path, svg_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(history_df["generation"], history_df["best_fitness"], label="Best fitness", color="blue")
    ax.plot(history_df["generation"], history_df["avg_fitness"], label="Average fitness", color="orange")
    ax.set_xlabel("Generation")
    ax.set_ylabel("CV R²")
    ax.set_title("Genetic Algorithm Convergence")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    save_figure(fig, png_path, svg_path)


def plot_hpo_round_performance(
    candidates,
    png_path: Path,
    svg_path: Path,
    round_index: int,
) -> None:
    if not candidates:
        return
    fig, ax = plt.subplots(figsize=(10, 5))
    ranks = [c.rank for c in candidates]
    train_r2 = [c.mean_train_r2 for c in candidates]
    cv_r2 = [c.mean_cv_r2 for c in candidates]
    cv_std = [c.std_cv_r2 for c in candidates]
    ax.plot(ranks, train_r2, "o-", label="Mean train R²", color="blue", alpha=0.7)
    ax.errorbar(
        ranks,
        cv_r2,
        yerr=cv_std,
        fmt="s-",
        label="Mean CV R²",
        color="green",
        capsize=3,
        alpha=0.8,
    )
    best = next((c for c in candidates if c.is_best), candidates[0])
    ax.axvline(best.rank, color="red", linestyle="--", label=f"Best (rank {best.rank})")
    ax.set_xlabel("Candidate rank")
    ax.set_ylabel("R²")
    ax.set_title(f"HPO Round {round_index}: Train vs CV Performance")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    save_figure(fig, png_path, svg_path)


def plot_hpo_summary(
    summary_rows: list[dict],
    png_path: Path,
    svg_path: Path,
    selected_source: str,
) -> None:
    if not summary_rows:
        return
    fig, ax = plt.subplots(figsize=(10, 5))
    labels = [r["source"] for r in summary_rows]
    x = np.arange(len(labels))
    train_r2 = [r["mean_train_r2"] for r in summary_rows]
    cv_r2 = [r["mean_cv_r2"] for r in summary_rows]
    cv_std = [r["std_cv_r2"] for r in summary_rows]
    ax.plot(x, train_r2, "o-", label="Mean train R²", color="blue")
    ax.errorbar(x, cv_r2, yerr=cv_std, fmt="s-", label="Mean CV R²", color="green", capsize=3)
    for i, row in enumerate(summary_rows):
        if row["source"] == selected_source:
            ax.axvline(i, color="red", linestyle="--", alpha=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("R²")
    ax.set_title("HPO Summary: Baseline and Rounds")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    save_figure(fig, png_path, svg_path)


def plot_prediction_scatter(
    train_true,
    train_pred,
    test_true,
    test_pred,
    train_metrics: dict,
    test_metrics: dict,
    activity_label: str,
    png_path: Path,
    svg_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(train_true, train_pred, c="blue", alpha=0.7, s=50, label="Training")
    ax.scatter(test_true, test_pred, c="gold", alpha=0.7, s=50, marker="^", label="External test")

    all_vals = np.concatenate([train_true, test_true, train_pred, test_pred])
    vmin, vmax = float(np.min(all_vals)), float(np.max(all_vals))
    margin = (vmax - vmin) * 0.05 if vmax > vmin else 1.0
    lo, hi = vmin - margin, vmax + margin
    ax.plot([lo, hi], [lo, hi], "k--", linewidth=1.5, label="1:1 line")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel(f"Experimental {activity_label}")
    ax.set_ylabel(f"Predicted {activity_label}")
    ax.set_title("Predicted vs Experimental Activity")
    text = (
        f"Train: R²={train_metrics['r2']:.3f}, RMSE={train_metrics['rmse']:.3f}, "
        f"MAE={train_metrics['mae']:.3f}, n={train_metrics['n_samples']}\n"
        f"Test: R²={test_metrics['r2']:.3f}, RMSE={test_metrics['rmse']:.3f}, "
        f"MAE={test_metrics['mae']:.3f}, n={test_metrics['n_samples']}"
    )
    ax.text(0.05, 0.95, text, transform=ax.transAxes, va="top", fontsize=9,
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
    ax.legend()
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    save_figure(fig, png_path, svg_path)


def plot_williams(
    leverage,
    std_residuals,
    splits,
    h_star: float,
    png_path: Path,
    svg_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 8))
    for split, color, marker in [("train", "blue", "o"), ("test", "gold", "^")]:
        mask = np.array(splits) == split
        ax.scatter(
            np.array(leverage)[mask],
            np.array(std_residuals)[mask],
            c=color,
            marker=marker,
            s=50,
            alpha=0.7,
            label=split.capitalize(),
        )
    ax.axhline(3, color="red", linestyle="--", linewidth=1, label="±3 residual")
    ax.axhline(-3, color="red", linestyle="--", linewidth=1)
    ax.axvline(h_star, color="purple", linestyle="--", linewidth=1, label=f"h*={h_star:.3f}")
    ax.set_xlabel("Leverage")
    ax.set_ylabel("Standardized residual")
    ax.set_title("Williams Plot (Applicability Domain)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    save_figure(fig, png_path, svg_path)
