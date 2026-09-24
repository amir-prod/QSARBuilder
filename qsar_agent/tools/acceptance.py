"""User-specified acceptance criteria and failure diagnosis.

Thresholds are **not** library defaults. The user declares what "good enough"
means for their dataset and decision; the values in :data:`CRITERION_SPECS` are
only *suggestions* that the CLI displays while asking. Once a run starts the
criteria are frozen — the Strategist can read them but has no way to relax them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Literal

from pydantic import BaseModel, Field

from qsar_agent.schemas.agentic import (
    AcceptanceVerdict,
    CriterionCheck,
    FailureDiagnosis,
    TaskDetection,
    ValidationReport,
)


class AcceptanceCriteria(BaseModel):
    """Thresholds the user requires before a model is called validated.

    ``None`` disables a criterion. Fields are grouped by task; only the ones
    applicable to the detected task are evaluated.
    """

    model_config = {"frozen": True}

    # Regression
    min_test_r2: float | None = None
    min_cv_q2: float | None = None
    max_train_cv_gap: float | None = None
    max_test_rmse: float | None = None
    max_test_mae: float | None = None

    # Classification
    min_test_roc_auc: float | None = None
    min_test_mcc: float | None = None
    min_test_balanced_accuracy: float | None = None
    min_cv_roc_auc: float | None = None

    # Shared
    max_scramble_score: float | None = None
    min_scramble_margin: float | None = None
    min_ad_coverage_pct: float | None = None

    def enabled_names(self, task: str) -> list[str]:
        return [
            spec.field
            for spec in specs_for_task(task)
            if getattr(self, spec.field) is not None
        ]

    def describe(self, task: str) -> dict[str, Any]:
        return {
            spec.field: {
                "label": spec.label,
                "threshold": getattr(self, spec.field),
                "direction": spec.direction,
                "enabled": getattr(self, spec.field) is not None,
            }
            for spec in specs_for_task(task)
        }


@dataclass(frozen=True)
class CriterionSpec:
    """Describes one criterion: how to ask for it, and how to check it."""

    field: str
    label: str
    description: str
    direction: Literal["min", "max"]
    suggested: float
    source: str
    tasks: tuple[str, ...]
    extract: Callable[[ValidationReport], float | None]
    cli_flag: str


def _test_metric(key: str) -> Callable[[ValidationReport], float | None]:
    def extractor(report: ValidationReport) -> float | None:
        return report.test_metrics.get(key)

    return extractor


def _cv_mean(report: ValidationReport) -> float | None:
    return report.cv.mean_cv_score


def _cv_gap(report: ValidationReport) -> float | None:
    return report.cv.train_cv_gap


def _scramble_mean(report: ValidationReport) -> float | None:
    return report.y_scrambling.mean_scrambled_score if report.y_scrambling else None


def _scramble_margin(report: ValidationReport) -> float | None:
    return report.y_scrambling.margin if report.y_scrambling else None


def _ad_coverage(report: ValidationReport) -> float | None:
    return report.applicability_domain.test_coverage_pct if report.applicability_domain else None


#: Single source of truth shared by the CLI prompt, the criteria file loader,
#: and the evaluator.
CRITERION_SPECS: tuple[CriterionSpec, ...] = (
    CriterionSpec(
        field="min_test_r2",
        label="Hold-out test R2 at least",
        description="Coefficient of determination on the untouched test set.",
        direction="min",
        suggested=0.60,
        source="Tropsha external-validation guidance (R2_test > 0.6)",
        tasks=("regression",),
        extract=_test_metric("r2"),
        cli_flag="--min-test-r2",
    ),
    CriterionSpec(
        field="min_cv_q2",
        label="Cross-validated Q2 (train) at least",
        description="Mean K-fold R2 on the training set.",
        direction="min",
        suggested=0.50,
        source="OECD internal-validation floor (q2 > 0.5)",
        tasks=("regression",),
        extract=_cv_mean,
        cli_flag="--min-cv-q2",
    ),
    CriterionSpec(
        field="max_train_cv_gap",
        label="Train minus CV R2 gap at most",
        description="Overfitting guard: how much better the model fits training folds.",
        direction="max",
        suggested=0.30,
        source="common overfitting heuristic",
        tasks=("regression",),
        extract=_cv_gap,
        cli_flag="--max-train-cv-gap",
    ),
    CriterionSpec(
        field="max_test_rmse",
        label="Hold-out test RMSE at most",
        description="Root-mean-square error in activity units (dataset-specific).",
        direction="max",
        suggested=0.0,
        source="no general value exists; depends on your activity scale",
        tasks=("regression",),
        extract=_test_metric("rmse"),
        cli_flag="--max-test-rmse",
    ),
    CriterionSpec(
        field="max_test_mae",
        label="Hold-out test MAE at most",
        description="Mean absolute error in activity units (dataset-specific).",
        direction="max",
        suggested=0.0,
        source="no general value exists; depends on your activity scale",
        tasks=("regression",),
        extract=_test_metric("mae"),
        cli_flag="--max-test-mae",
    ),
    CriterionSpec(
        field="min_test_roc_auc",
        label="Hold-out ROC-AUC at least",
        description="Ranking quality on the untouched test set.",
        direction="min",
        suggested=0.70,
        source="common QSAR classification acceptability floor",
        tasks=("classification",),
        extract=_test_metric("roc_auc"),
        cli_flag="--min-roc-auc",
    ),
    CriterionSpec(
        field="min_test_mcc",
        label="Hold-out Matthews correlation at least",
        description="Imbalance-robust agreement between prediction and truth.",
        direction="min",
        suggested=0.30,
        source="imbalance-robust agreement convention",
        tasks=("classification",),
        extract=_test_metric("mcc"),
        cli_flag="--min-mcc",
    ),
    CriterionSpec(
        field="min_test_balanced_accuracy",
        label="Hold-out balanced accuracy at least",
        description="Mean of sensitivity and specificity.",
        direction="min",
        suggested=0.65,
        source="imbalance-robust accuracy convention",
        tasks=("classification",),
        extract=_test_metric("balanced_accuracy"),
        cli_flag="--min-balanced-accuracy",
    ),
    CriterionSpec(
        field="min_cv_roc_auc",
        label="Cross-validated ROC-AUC (train) at least",
        description="Mean K-fold ROC-AUC on the training set.",
        direction="min",
        suggested=0.70,
        source="internal-validation counterpart of the hold-out threshold",
        tasks=("classification",),
        extract=_cv_mean,
        cli_flag="--min-cv-roc-auc",
    ),
    CriterionSpec(
        field="max_scramble_score",
        label="Mean y-scrambled CV score at most",
        description="How well the model does on randomly shuffled activities.",
        direction="max",
        suggested=0.20,
        source="chance-correlation control (y-randomisation)",
        tasks=("regression", "classification"),
        extract=_scramble_mean,
        cli_flag="--max-scramble-score",
    ),
    CriterionSpec(
        field="min_scramble_margin",
        label="Real minus scrambled CV score at least",
        description="How far real performance exceeds chance.",
        direction="min",
        suggested=0.25,
        source="chance-correlation control (y-randomisation)",
        tasks=("regression", "classification"),
        extract=_scramble_margin,
        cli_flag="--min-scramble-margin",
    ),
    CriterionSpec(
        field="min_ad_coverage_pct",
        label="Test compounds inside applicability domain, at least (%)",
        description="k-NN distance-to-model coverage of the test set.",
        direction="min",
        suggested=80.0,
        source="OECD applicability-domain principle",
        tasks=("regression", "classification"),
        extract=_ad_coverage,
        cli_flag="--min-ad-coverage",
    ),
)

#: Suggested values that differ by task (chance-correlation scales differ:
#: a scrambled R2 near zero is expected, a scrambled ROC-AUC near 0.5 is).
_TASK_SUGGESTION_OVERRIDES: dict[str, dict[str, float]] = {
    "classification": {"max_scramble_score": 0.60, "min_scramble_margin": 0.10},
}

#: Criteria that are only meaningful once the user supplies a dataset-specific
#: number, so the CLI leaves them off unless asked for.
OPTIONAL_BY_DEFAULT = frozenset({"max_test_rmse", "max_test_mae"})


def specs_for_task(task: str) -> list[CriterionSpec]:
    return [spec for spec in CRITERION_SPECS if task in spec.tasks]


def suggested_value(spec: CriterionSpec, task: str) -> float:
    return _TASK_SUGGESTION_OVERRIDES.get(task, {}).get(spec.field, spec.suggested)


def spec_by_field(field: str) -> CriterionSpec:
    for spec in CRITERION_SPECS:
        if spec.field == field:
            return spec
    raise KeyError(f"Unknown acceptance criterion: {field!r}")


def suggested_criteria(task: str) -> AcceptanceCriteria:
    """The literature-typical starting point the CLI offers for confirmation."""
    values = {
        spec.field: suggested_value(spec, task)
        for spec in specs_for_task(task)
        if spec.field not in OPTIONAL_BY_DEFAULT
    }
    return AcceptanceCriteria(**values)


def evaluate_acceptance(
    criteria: AcceptanceCriteria, report: ValidationReport
) -> AcceptanceVerdict:
    """Check every enabled criterion for the report's task.

    A criterion whose observed value is missing (for example AD coverage when the
    domain could not be computed) counts as a failure: the loop never treats an
    unmeasured requirement as satisfied.
    """
    checks: list[CriterionCheck] = []
    for spec in specs_for_task(report.task):
        threshold = getattr(criteria, spec.field)
        if threshold is None:
            continue
        observed = spec.extract(report)
        if observed is None or observed != observed:  # None or NaN
            passed = False
        elif spec.direction == "min":
            passed = observed >= threshold
        else:
            passed = observed <= threshold
        checks.append(
            CriterionCheck(
                name=spec.field,
                description=spec.label,
                observed=None if observed is None or observed != observed else float(observed),
                threshold=float(threshold),
                direction=spec.direction,
                passed=passed,
            )
        )
    return AcceptanceVerdict(accepted=bool(checks) and all(c.passed for c in checks), checks=checks)


class DiagnosisThresholds(BaseModel):
    """Sensitivity of the failure classifier (not acceptance thresholds)."""

    overfit_gap: float = 0.20
    unstable_cv_std: float = 0.15
    generalisation_drop: float = 0.20
    imbalance_ratio: float = 3.0
    min_samples_per_feature: float = 0.5
    scramble_margin: float = 0.10
    extra: dict[str, float] = Field(default_factory=dict)


def diagnose_failure(
    report: ValidationReport,
    verdict: AcceptanceVerdict,
    task_detection: TaskDetection | None = None,
    thresholds: DiagnosisThresholds | None = None,
) -> FailureDiagnosis:
    """Turn a set of failed criteria into one actionable label.

    The order below is a triage priority: chance correlation invalidates
    everything else, so it is checked first, and a healthy-CV/poor-test pattern
    is distinguished from plain overfitting because they call for different fixes.
    """
    limits = thresholds or DiagnosisThresholds()

    if verdict.accepted:
        return FailureDiagnosis(label="accepted", explanation="All acceptance criteria passed.")

    failed = {c.name for c in verdict.failed_checks}
    cv = report.cv
    primary = report.primary_metric
    test_score = report.test_metrics.get(primary)
    evidence: dict[str, float] = {
        "mean_cv_score": cv.mean_cv_score,
        "mean_train_score": cv.mean_train_score,
        "std_cv_score": cv.std_cv_score,
        "train_cv_gap": cv.train_cv_gap,
        "n_train": float(report.n_train),
        "n_features": float(report.n_features),
    }
    if test_score is not None:
        evidence["test_score"] = float(test_score)

    scramble = report.y_scrambling
    if scramble is not None:
        evidence["mean_scrambled_score"] = scramble.mean_scrambled_score
        evidence["scramble_margin"] = scramble.margin

    samples_per_feature = (
        float(report.n_train) / float(report.n_features) if report.n_features else 0.0
    )
    evidence["samples_per_feature"] = samples_per_feature

    # 1. Chance correlation invalidates any other reading of the metrics.
    if scramble is not None and (
        {"max_scramble_score", "min_scramble_margin"} & failed
        or scramble.margin < limits.scramble_margin
    ):
        return FailureDiagnosis(
            label="chance_correlation",
            explanation=(
                f"Scrambled-activity models reach {scramble.mean_scrambled_score:.3f} "
                f"({primary}) against {scramble.real_cv_score:.3f} for the real labels, a margin "
                f"of only {scramble.margin:.3f}. With {report.n_features} features for "
                f"{report.n_train} training compounds the model can fit noise."
            ),
            evidence=evidence,
            suggested_actions=[
                "Cut the feature count sharply (univariate_top_k or a tighter correlation filter)",
                "Use a simpler, strongly regularised estimator",
                "Reconsider whether the endpoint is modellable from structure alone",
            ],
        )

    # 2. Too few compounds per feature to learn anything reliable.
    if samples_per_feature < limits.min_samples_per_feature and report.n_features > 0:
        return FailureDiagnosis(
            label="insufficient_data",
            explanation=(
                f"Only {samples_per_feature:.2f} training compounds per feature "
                f"({report.n_train} compounds, {report.n_features} features). Any metric here is "
                "unreliable."
            ),
            evidence=evidence,
            suggested_actions=[
                "Reduce dimensionality aggressively (univariate_top_k)",
                "Drop fingerprint blocks and keep physicochemical descriptors",
            ],
        )

    # 3. Class imbalance, when it is the plausible cause of poor class metrics.
    if (
        report.task == "classification"
        and task_detection is not None
        and task_detection.imbalance_ratio is not None
        and task_detection.imbalance_ratio >= limits.imbalance_ratio
        and {"min_test_mcc", "min_test_balanced_accuracy"} & failed
    ):
        evidence["imbalance_ratio"] = float(task_detection.imbalance_ratio)
        return FailureDiagnosis(
            label="class_imbalance",
            explanation=(
                f"Class sizes differ by a factor of {task_detection.imbalance_ratio:.1f} "
                f"({task_detection.class_counts}), and the imbalance-sensitive criteria are the "
                "ones failing."
            ),
            evidence=evidence,
            suggested_actions=[
                "Set class_weight='balanced' on the estimator",
                "Prefer balanced accuracy and MCC over raw accuracy when tuning",
            ],
        )

    # 4. Overfitting: fits training folds far better than held-out folds.
    if cv.train_cv_gap > limits.overfit_gap or "max_train_cv_gap" in failed:
        return FailureDiagnosis(
            label="overfit",
            explanation=(
                f"Training-fold {primary} of {cv.mean_train_score:.3f} against cross-validated "
                f"{cv.mean_cv_score:.3f} is a gap of {cv.train_cv_gap:.3f}: the model memorises "
                "its training folds."
            ),
            evidence=evidence,
            suggested_actions=[
                "Constrain model capacity (shallower trees, larger min_samples_leaf, higher alpha)",
                "Reduce the feature count",
                "Switch to a simpler estimator family",
            ],
        )

    # 5. Healthy CV but poor hold-out: a generalisation gap, not memorisation.
    if (
        test_score is not None
        and cv.mean_cv_score - float(test_score) > limits.generalisation_drop
    ):
        return FailureDiagnosis(
            label="poor_generalisation",
            explanation=(
                f"Cross-validation reaches {cv.mean_cv_score:.3f} but the hold-out test set only "
                f"{float(test_score):.3f}. The test compounds differ from the training set more "
                "than the CV folds do."
            ),
            evidence=evidence,
            suggested_actions=[
                "Add regularisation so the model relies on more general features",
                "Broaden the feature space to cover the test chemistry",
                "Check the split for a chemical-series effect",
            ],
        )

    # 6. Unstable CV: the mean is not trustworthy.
    if cv.std_cv_score > limits.unstable_cv_std:
        return FailureDiagnosis(
            label="unstable_cv",
            explanation=(
                f"Cross-validated {primary} varies by {cv.std_cv_score:.3f} across folds "
                f"(mean {cv.mean_cv_score:.3f}), so the model is sensitive to which compounds "
                "it sees."
            ),
            evidence=evidence,
            suggested_actions=[
                "Filter noisy features",
                "Average more estimators (larger n_estimators)",
                "Increase the number of CV folds",
            ],
        )

    # 7. Domain too narrow while performance itself is acceptable.
    if failed == {"min_ad_coverage_pct"}:
        coverage = (
            report.applicability_domain.test_coverage_pct
            if report.applicability_domain
            else float("nan")
        )
        return FailureDiagnosis(
            label="narrow_applicability_domain",
            explanation=(
                f"Performance criteria pass but only {coverage:.1f}% of test compounds fall inside "
                "the k-NN applicability domain, so the predictions are extrapolations."
            ),
            evidence=evidence,
            suggested_actions=[
                "Drop high-distance training outliers",
                "Reduce dimensionality so distances are more meaningful",
            ],
        )

    # 8. Nothing pathological — the model is simply not good enough yet.
    return FailureDiagnosis(
        label="underfit",
        explanation=(
            f"Cross-validated {primary} of {cv.mean_cv_score:.3f} and training "
            f"{cv.mean_train_score:.3f} are both below what the criteria require, with no "
            "overfitting signature: the current features and model lack capacity."
        ),
        evidence=evidence,
        suggested_actions=[
            "Add structural features (fingerprints, MACCS keys)",
            "Move to a higher-capacity estimator (ensembles, boosting)",
            "Loosen regularisation",
        ],
    )
