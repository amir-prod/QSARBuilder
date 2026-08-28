"""Fixed candidate-ranking policy (must match the Markdown explanation)."""

from __future__ import annotations

from typing import Any

from qsar_agent.agentic.requirements import evaluate_requirements, requirements_from_config
from qsar_agent.config import ModelingRequirements, WorkflowConfig
from qsar_agent.models.registry import SUPPORTED_ESTIMATORS, model_simplicity_score
from qsar_agent.schemas.agentic import CandidateRanking


def _estimator_rank(name: str) -> int:
    try:
        return SUPPORTED_ESTIMATORS.index(name)
    except ValueError:
        return len(SUPPORTED_ESTIMATORS) + 1


def rank_candidates(
    experiments: list[dict[str, Any]],
    requirements: ModelingRequirements,
) -> list[CandidateRanking]:
    """Rank experiments: hard-requirement passers first, then scientific tie-breaks.

    Among eligible candidates prefer:
    1. better outer-CV primary metric (R²)
    2. lower error (RMSE)
    3. lower variability
    4. smaller train–CV gap
    5. fewer features
    6. simpler model
    """
    scored: list[tuple[CandidateRanking, tuple]] = []
    for exp in experiments:
        metrics = exp.get("metrics") or exp
        n_feat = exp.get("feature_count")
        if n_feat is None:
            feats = exp.get("selected_features") or []
            n_feat = len(feats) if feats else None
        evaluation = evaluate_requirements(
            metrics,
            requirements,
            feature_count=None if n_feat is None else int(n_feat),
            ad_coverage=exp.get("ad_coverage"),
            runtime_seconds=exp.get("runtime_seconds"),
        )
        eligible = evaluation.acceptance_status == "passed"
        cv_r2 = evaluation.metrics.get("cv_r2")
        rmse = evaluation.metrics.get("cv_rmse")
        std = evaluation.metrics.get("cv_r2_std")
        gap = evaluation.train_cv_gap
        estimator = str(exp.get("model") or exp.get("estimator") or "")
        simplicity = model_simplicity_score(estimator, exp.get("hyperparameters") or exp.get("params") or {})
        # Eligible models always rank above ineligible. Within a bucket, higher is better
        # for R²; lower is better for error/std/gap/features/simplicity.
        sort_key = (
            0 if eligible else 1,
            -(cv_r2 if cv_r2 is not None else float("-inf")),
            rmse if rmse is not None else float("inf"),
            std if std is not None else float("inf"),
            gap if gap is not None else float("inf"),
            int(n_feat) if n_feat is not None else 10**9,
            simplicity,
            _estimator_rank(estimator),
        )
        score = (
            (100.0 if eligible else 0.0)
            + (cv_r2 or 0.0)
            - 0.01 * (rmse or 0.0)
            - 0.05 * (std or 0.0)
            - 0.1 * (gap or 0.0)
            - 0.001 * (int(n_feat) if n_feat is not None else 0)
        )
        reason_bits = []
        if eligible:
            reason_bits.append("satisfies every hard requirement")
        else:
            names = ", ".join(f.name for f in evaluation.failed_requirements)
            reason_bits.append(f"fails hard requirements: {names}")
        if cv_r2 is not None:
            reason_bits.append(f"outer-CV R²={cv_r2:.4f}")
        ranking = CandidateRanking(
            experiment_id=str(exp.get("experiment_id") or exp.get("run_id") or ""),
            eligible=eligible,
            failed_requirements=[f.name for f in evaluation.failed_requirements],
            selection_score=float(score),
            rank=0,
            selection_reason="; ".join(reason_bits),
        )
        scored.append((ranking, sort_key))

    scored.sort(key=lambda item: item[1])
    out: list[CandidateRanking] = []
    for i, (ranking, _key) in enumerate(scored, start=1):
        out.append(ranking.model_copy(update={"rank": i}))
    return out


def rank_from_workflow(
    experiments: list[dict[str, Any]],
    config: WorkflowConfig,
) -> list[CandidateRanking]:
    return rank_candidates(experiments, requirements_from_config(config))
