"""Deterministic best-experiment selection using internal evidence only."""

from __future__ import annotations

from typing import Any

from qsar_agent.schemas.agentic import ExperimentRecord


def select_best_experiment(
    records: list[ExperimentRecord],
    *,
    practical_equivalence_tolerance: float = 0.01,
    acceptance_lookup: dict[str, bool] | None = None,
) -> tuple[ExperimentRecord | None, str]:
    """Select winner by ordered internal policy. LLM cannot override."""
    if not records:
        return None, "No experiment records available."

    completed = [r for r in records if r.status in ("completed", "accepted", "locked")]
    if not completed:
        completed = list(records)

    def is_acceptable(r: ExperimentRecord) -> bool:
        if acceptance_lookup and r.experiment_id in acceptance_lookup:
            return acceptance_lookup[r.experiment_id]
        return bool(r.internal_metrics.get("accepted", False))

    def mean_cv(r: ExperimentRecord) -> float:
        return float(r.internal_metrics.get("mean_cv_r2", float("-inf")))

    def gap(r: ExperimentRecord) -> float:
        return float(r.internal_metrics.get("train_cv_gap", float("inf")))

    def cv_std(r: ExperimentRecord) -> float:
        return float(r.internal_metrics.get("cv_r2_std", float("inf")))

    def complexity(r: ExperimentRecord) -> float:
        # Prefer fewer features; fall back to feature_count / simplicity score
        fc = r.feature_count
        if fc is None:
            fc = r.internal_metrics.get("feature_count")
        if fc is None:
            return float("inf")
        return float(fc)

    acceptable = [r for r in completed if is_acceptable(r)]
    pool = acceptable if acceptable else completed
    pool_label = "acceptable models" if acceptable else "all completed models (none met acceptance)"

    # Sort: higher CV R2, smaller gap, lower std, simpler
    ranked = sorted(
        pool,
        key=lambda r: (-mean_cv(r), gap(r), cv_std(r), complexity(r), r.experiment_id),
    )
    best = ranked[0]

    # Practical equivalence: among near-tied CV R2, prefer simpler
    best_cv = mean_cv(best)
    tied = [
        r
        for r in ranked
        if abs(mean_cv(r) - best_cv) <= practical_equivalence_tolerance
    ]
    if len(tied) > 1:
        tied_sorted = sorted(tied, key=lambda r: (complexity(r), gap(r), cv_std(r), r.experiment_id))
        if tied_sorted[0].experiment_id != best.experiment_id:
            rationale = (
                f"Selected {tied_sorted[0].experiment_id} from {pool_label}: "
                f"CV R² within practical equivalence ({practical_equivalence_tolerance}) of "
                f"best {best.experiment_id} (CV R²={best_cv:.4f}); preferred simpler model "
                f"(features={complexity(tied_sorted[0])})."
            )
            return tied_sorted[0], rationale

    rationale = (
        f"Selected {best.experiment_id} from {pool_label}: "
        f"mean_cv_r2={mean_cv(best):.4f}, train_cv_gap={gap(best):.4f}, "
        f"cv_r2_std={cv_std(best):.4f}, feature_count={complexity(best)}."
    )
    return best, rationale


def compare_to_parent(
    child: ExperimentRecord,
    parent: ExperimentRecord | None,
) -> dict[str, Any]:
    if parent is None:
        return {"parent_experiment_id": None, "delta_mean_cv_r2": None}
    c = float(child.internal_metrics.get("mean_cv_r2", float("nan")))
    p = float(parent.internal_metrics.get("mean_cv_r2", float("nan")))
    return {
        "parent_experiment_id": parent.experiment_id,
        "parent_mean_cv_r2": p,
        "child_mean_cv_r2": c,
        "delta_mean_cv_r2": c - p if c == c and p == p else None,
        "improved": (c > p) if c == c and p == p else None,
    }
