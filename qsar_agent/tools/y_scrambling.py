"""y-randomization (y-scrambling) control against chance correlation.

The activity vector is shuffled repeatedly and the whole cross-validation is
re-run. A model that scores well on scrambled labels is fitting noise, so this
is a mandatory gate rather than a diagnostic afterthought.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from sklearn.base import BaseEstimator, clone
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score

from qsar_agent.schemas.agentic import YScramblingResult
from qsar_agent.tools.metrics import CV_SCORER

DEFAULT_N_REPEATS = 10


def _cv_splitter(task: str, folds: int, seed: int):
    if task == "classification":
        return StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    return KFold(n_splits=folds, shuffle=True, random_state=seed)


def run_y_scrambling(
    X: np.ndarray,
    y: np.ndarray,
    estimator: BaseEstimator | Callable[[], BaseEstimator],
    task: str,
    real_cv_score: float,
    n_repeats: int = DEFAULT_N_REPEATS,
    cv_folds: int = 5,
    random_seed: int = 42,
    n_jobs: int = 1,
) -> YScramblingResult:
    """Shuffle ``y`` ``n_repeats`` times and re-score by cross-validation.

    ``real_cv_score`` is passed in rather than recomputed so the comparison uses
    exactly the score the loop already reported.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)
    if len(y) != len(X):
        raise ValueError(f"X has {len(X)} rows but y has {len(y)} values.")

    scorer = CV_SCORER[task]
    rng = np.random.default_rng(random_seed)
    scores: list[float] = []

    for repeat in range(n_repeats):
        shuffled = y.copy()
        rng.shuffle(shuffled)
        if task == "classification" and len(np.unique(shuffled)) < 2:
            continue
        model = clone(estimator) if isinstance(estimator, BaseEstimator) else estimator()
        splitter = _cv_splitter(task, cv_folds, random_seed + repeat)
        try:
            fold_scores = cross_val_score(
                model, X, shuffled, cv=splitter, scoring=scorer, n_jobs=n_jobs,
                error_score=np.nan,
            )
        except ValueError:
            continue
        mean_score = float(np.nanmean(fold_scores))
        if np.isfinite(mean_score):
            scores.append(mean_score)

    if not scores:
        raise ValueError(
            "y-scrambling produced no valid scores; the dataset is likely too small "
            "or too imbalanced for cross-validation."
        )

    array = np.asarray(scores, dtype=float)
    mean_scrambled = float(array.mean())
    return YScramblingResult(
        n_repeats=len(scores),
        primary_metric=scorer,
        real_cv_score=float(real_cv_score),
        scrambled_scores=[float(s) for s in array],
        mean_scrambled_score=mean_scrambled,
        std_scrambled_score=float(array.std(ddof=0)),
        max_scrambled_score=float(array.max()),
        margin=float(real_cv_score) - mean_scrambled,
        fraction_scrambled_at_least_real=float(np.mean(array >= float(real_cv_score))),
    )
