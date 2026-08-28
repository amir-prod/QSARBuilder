"""Feature-subset stability summaries."""

from __future__ import annotations

from typing import Iterable, Literal

from qsar_agent.schemas.agentic import FeatureStabilityReport

StabilityStatus = Literal["stable", "mixed", "unstable"]


def pairwise_jaccard(subsets: list[set[str]]) -> float:
    if len(subsets) < 2:
        return 1.0 if subsets else 0.0
    scores: list[float] = []
    for i, a in enumerate(subsets):
        for b in subsets[i + 1 :]:
            union = a | b
            scores.append(0.0 if not union else len(a & b) / len(union))
    return float(sum(scores) / len(scores)) if scores else 0.0


def selection_frequencies(subsets: Iterable[Iterable[str]]) -> dict[str, float]:
    lists = [list(s) for s in subsets]
    n = len(lists)
    if n == 0:
        return {}
    counts: dict[str, int] = {}
    for subset in lists:
        for name in set(subset):
            counts[name] = counts.get(name, 0) + 1
    return {k: v / n for k, v in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))}


def summarize_stability(
    subsets: list[list[str]],
    *,
    stable_threshold: float = 0.8,
    unstable_threshold: float = 0.4,
) -> FeatureStabilityReport:
    freq = selection_frequencies(subsets)
    jaccard = pairwise_jaccard([set(s) for s in subsets])
    stable = [name for name, f in freq.items() if f >= stable_threshold]
    unstable = [name for name, f in freq.items() if f < unstable_threshold]
    if jaccard >= 0.6 and len(stable) >= max(1, int(0.5 * max(len(freq), 1))):
        status: StabilityStatus = "stable"
    elif jaccard <= 0.25:
        status = "unstable"
    else:
        status = "mixed"
    return FeatureStabilityReport(
        selection_frequency=freq,
        mean_pairwise_jaccard=jaccard,
        stable_features=stable,
        unstable_features=unstable,
        stability_status=status,
    )


def consensus_subset(subsets: list[list[str]], min_frequency: float = 0.5, max_size: int | None = None) -> list[str]:
    freq = selection_frequencies(subsets)
    names = [name for name, f in freq.items() if f >= min_frequency]
    if max_size is not None:
        names = names[: max(0, max_size)]
    return names
