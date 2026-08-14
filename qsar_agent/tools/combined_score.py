"""Combined CV + held-out validation scoring."""

from __future__ import annotations

CV_VAL_WEIGHT = 0.5


def combined_r2(mean_cv_r2: float, val_r2: float | None) -> float:
    """Equal-weight blend of training CV R² and held-out validation R².

    Falls back to CV-only when ``val_r2`` is missing (tiny datasets or no val set).
    """
    if val_r2 is None:
        return float(mean_cv_r2)
    return CV_VAL_WEIGHT * float(mean_cv_r2) + (1.0 - CV_VAL_WEIGHT) * float(val_r2)
