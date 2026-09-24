"""Tests for user-specified acceptance criteria and failure diagnosis."""

from __future__ import annotations

import pytest

from qsar_agent.schemas.agentic import (
    ADSummary,
    CVSummary,
    TaskDetection,
    ValidationReport,
    YScramblingResult,
)
from qsar_agent.tools.acceptance import (
    CRITERION_SPECS,
    OPTIONAL_BY_DEFAULT,
    AcceptanceCriteria,
    DiagnosisThresholds,
    diagnose_failure,
    evaluate_acceptance,
    spec_by_field,
    specs_for_task,
    suggested_criteria,
    suggested_value,
)


def make_report(
    task: str = "regression",
    test_score: float = 0.75,
    cv_score: float = 0.70,
    train_score: float = 0.80,
    cv_std: float = 0.05,
    scramble_mean: float | None = 0.02,
    ad_coverage: float | None = 95.0,
    n_train: int = 200,
    n_features: int = 40,
    extra_test_metrics: dict[str, float] | None = None,
) -> ValidationReport:
    primary = "r2" if task == "regression" else "roc_auc"
    test_metrics = {primary: test_score}
    if task == "regression":
        test_metrics.update({"rmse": 0.5, "mae": 0.4})
    else:
        test_metrics.update({"mcc": 0.5, "balanced_accuracy": 0.75})
    if extra_test_metrics:
        test_metrics.update(extra_test_metrics)

    scrambling = None
    if scramble_mean is not None:
        scrambling = YScramblingResult(
            n_repeats=10,
            primary_metric=primary,
            real_cv_score=cv_score,
            scrambled_scores=[scramble_mean] * 10,
            mean_scrambled_score=scramble_mean,
            std_scrambled_score=0.01,
            max_scrambled_score=scramble_mean,
            margin=cv_score - scramble_mean,
            fraction_scrambled_at_least_real=0.0,
        )

    domain = None
    if ad_coverage is not None:
        domain = ADSummary(
            threshold=1.0,
            train_mean_distance=0.5,
            test_in_domain_count=int(ad_coverage),
            test_total_count=100,
            test_coverage_pct=ad_coverage,
        )

    return ValidationReport(
        task=task,
        primary_metric=primary,
        n_train=n_train,
        n_test=50,
        n_features=n_features,
        train_metrics={primary: train_score},
        test_metrics=test_metrics,
        cv=CVSummary(
            folds=5,
            primary_metric=primary,
            mean_train_score=train_score,
            mean_cv_score=cv_score,
            std_cv_score=cv_std,
            train_cv_gap=train_score - cv_score,
        ),
        y_scrambling=scrambling,
        applicability_domain=domain,
    )


# -- criteria plumbing ---------------------------------------------------------


def test_no_criteria_are_enabled_by_default():
    empty = AcceptanceCriteria()
    assert empty.enabled_names("regression") == []
    assert empty.enabled_names("classification") == []


def test_suggested_criteria_enable_the_non_optional_ones():
    criteria = suggested_criteria("regression")
    enabled = set(criteria.enabled_names("regression"))
    assert "min_test_r2" in enabled
    assert enabled.isdisjoint(OPTIONAL_BY_DEFAULT)


def test_scramble_suggestions_differ_by_task():
    spec = spec_by_field("max_scramble_score")
    assert suggested_value(spec, "regression") == pytest.approx(0.20)
    assert suggested_value(spec, "classification") == pytest.approx(0.60)


def test_task_specific_criteria_are_separated():
    regression_fields = {spec.field for spec in specs_for_task("regression")}
    classification_fields = {spec.field for spec in specs_for_task("classification")}
    assert "min_test_r2" in regression_fields
    assert "min_test_r2" not in classification_fields
    assert "min_test_roc_auc" in classification_fields
    assert "min_ad_coverage_pct" in regression_fields & classification_fields


def test_every_spec_has_a_unique_field_and_flag():
    fields = [spec.field for spec in CRITERION_SPECS]
    flags = [spec.cli_flag for spec in CRITERION_SPECS]
    assert len(fields) == len(set(fields))
    assert len(flags) == len(set(flags))


def test_criteria_are_immutable_once_constructed():
    criteria = suggested_criteria("regression")
    with pytest.raises(Exception):
        criteria.min_test_r2 = 0.01


# -- evaluation ----------------------------------------------------------------


def test_a_good_model_passes_the_suggested_criteria():
    verdict = evaluate_acceptance(suggested_criteria("regression"), make_report())
    assert verdict.accepted
    assert verdict.failed_checks == []
    assert "passed" in verdict.summary()


def test_a_failing_criterion_is_named():
    criteria = AcceptanceCriteria(min_test_r2=0.90)
    verdict = evaluate_acceptance(criteria, make_report(test_score=0.75))
    assert not verdict.accepted
    assert [c.name for c in verdict.failed_checks] == ["min_test_r2"]
    assert verdict.checks[0].observed == pytest.approx(0.75)
    assert verdict.checks[0].threshold == pytest.approx(0.90)


def test_disabled_criteria_are_not_checked():
    criteria = AcceptanceCriteria(min_test_r2=0.90, min_cv_q2=None)
    verdict = evaluate_acceptance(criteria, make_report())
    assert [c.name for c in verdict.checks] == ["min_test_r2"]


def test_only_criteria_for_the_reports_task_are_evaluated():
    criteria = AcceptanceCriteria(min_test_r2=0.5, min_test_roc_auc=0.9)
    verdict = evaluate_acceptance(criteria, make_report(task="regression"))
    assert [c.name for c in verdict.checks] == ["min_test_r2"]


def test_max_direction_criteria_compare_the_other_way():
    criteria = AcceptanceCriteria(max_train_cv_gap=0.05)
    verdict = evaluate_acceptance(criteria, make_report(train_score=0.95, cv_score=0.60))
    assert not verdict.accepted
    assert verdict.checks[0].direction == "max"


def test_an_unmeasured_criterion_fails_rather_than_passing_silently():
    criteria = AcceptanceCriteria(min_ad_coverage_pct=80.0)
    verdict = evaluate_acceptance(criteria, make_report(ad_coverage=None))
    assert not verdict.accepted
    assert verdict.checks[0].observed is None
    assert not verdict.checks[0].passed


def test_a_nan_observation_fails():
    criteria = AcceptanceCriteria(min_test_roc_auc=0.7)
    report = make_report(task="classification", extra_test_metrics={"roc_auc": float("nan")})
    verdict = evaluate_acceptance(criteria, report)
    assert not verdict.accepted
    assert verdict.checks[0].observed is None


def test_no_enabled_criteria_means_not_accepted():
    verdict = evaluate_acceptance(AcceptanceCriteria(), make_report())
    assert not verdict.accepted
    assert verdict.checks == []


def test_check_renders_readably():
    criteria = AcceptanceCriteria(min_test_r2=0.60)
    check = evaluate_acceptance(criteria, make_report(test_score=0.75)).checks[0]
    rendered = check.render()
    assert "PASS" in rendered and ">=" in rendered and "0.7500" in rendered


def test_classification_criteria_are_all_applied():
    criteria = suggested_criteria("classification")
    report = make_report(task="classification", test_score=0.85, cv_score=0.82,
                         train_score=0.90, scramble_mean=0.52)
    verdict = evaluate_acceptance(criteria, report)
    names = {c.name for c in verdict.checks}
    assert {"min_test_roc_auc", "min_test_mcc", "min_test_balanced_accuracy",
            "min_cv_roc_auc", "max_scramble_score", "min_scramble_margin",
            "min_ad_coverage_pct"} == names
    assert verdict.accepted


# -- diagnosis -----------------------------------------------------------------


def test_a_passing_model_is_diagnosed_as_accepted():
    report = make_report()
    verdict = evaluate_acceptance(suggested_criteria("regression"), report)
    assert diagnose_failure(report, verdict).label == "accepted"


def test_chance_correlation_is_diagnosed_first():
    # Deliberately also overfit: chance correlation must still win the triage.
    report = make_report(cv_score=0.55, train_score=0.99, scramble_mean=0.50, n_features=500)
    criteria = AcceptanceCriteria(max_scramble_score=0.20, min_test_r2=0.6)
    diagnosis = diagnose_failure(report, evaluate_acceptance(criteria, report))
    assert diagnosis.label == "chance_correlation"
    assert "scrambl" in diagnosis.explanation.lower()


def test_insufficient_data_is_diagnosed_from_samples_per_feature():
    report = make_report(cv_score=0.2, train_score=0.3, n_train=30, n_features=200)
    criteria = AcceptanceCriteria(min_cv_q2=0.5)
    diagnosis = diagnose_failure(report, evaluate_acceptance(criteria, report))
    assert diagnosis.label == "insufficient_data"
    assert diagnosis.evidence["samples_per_feature"] == pytest.approx(0.15)


def test_overfitting_is_diagnosed_from_the_train_cv_gap():
    report = make_report(cv_score=0.40, train_score=0.95, test_score=0.35)
    criteria = AcceptanceCriteria(min_test_r2=0.6, max_train_cv_gap=0.30)
    diagnosis = diagnose_failure(report, evaluate_acceptance(criteria, report))
    assert diagnosis.label == "overfit"
    assert diagnosis.evidence["train_cv_gap"] == pytest.approx(0.55)


def test_underfitting_is_diagnosed_when_nothing_is_pathological():
    report = make_report(cv_score=0.30, train_score=0.34, test_score=0.28, cv_std=0.02)
    criteria = AcceptanceCriteria(min_test_r2=0.6, min_cv_q2=0.5)
    diagnosis = diagnose_failure(report, evaluate_acceptance(criteria, report))
    assert diagnosis.label == "underfit"
    assert any("capacity" in action or "features" in action
               for action in diagnosis.suggested_actions)


def test_poor_generalisation_separates_healthy_cv_from_a_bad_holdout():
    report = make_report(cv_score=0.80, train_score=0.88, test_score=0.30, cv_std=0.03)
    criteria = AcceptanceCriteria(min_test_r2=0.6)
    diagnosis = diagnose_failure(report, evaluate_acceptance(criteria, report))
    assert diagnosis.label == "poor_generalisation"


def test_unstable_cv_is_diagnosed_from_fold_spread():
    report = make_report(cv_score=0.45, train_score=0.50, test_score=0.44, cv_std=0.35)
    criteria = AcceptanceCriteria(min_cv_q2=0.5)
    diagnosis = diagnose_failure(report, evaluate_acceptance(criteria, report))
    assert diagnosis.label == "unstable_cv"


def test_narrow_domain_is_diagnosed_when_it_is_the_only_failure():
    report = make_report(test_score=0.80, cv_score=0.78, train_score=0.85, ad_coverage=40.0)
    criteria = AcceptanceCriteria(min_test_r2=0.6, min_ad_coverage_pct=80.0)
    diagnosis = diagnose_failure(report, evaluate_acceptance(criteria, report))
    assert diagnosis.label == "narrow_applicability_domain"


def test_class_imbalance_needs_both_skewed_classes_and_failing_class_metrics():
    report = make_report(
        task="classification",
        test_score=0.72,
        cv_score=0.71,
        train_score=0.78,
        scramble_mean=0.50,
        extra_test_metrics={"mcc": 0.05, "balanced_accuracy": 0.51},
    )
    detection = TaskDetection(
        task="classification",
        reason="binary",
        n_unique_values=2,
        n_classes=2,
        class_counts={"0": 190, "1": 10},
        imbalance_ratio=19.0,
    )
    criteria = AcceptanceCriteria(min_test_mcc=0.30, min_test_balanced_accuracy=0.65)
    diagnosis = diagnose_failure(report, evaluate_acceptance(criteria, report), detection)
    assert diagnosis.label == "class_imbalance"
    assert "class_weight" in " ".join(diagnosis.suggested_actions)


def test_balanced_classes_are_not_blamed_on_imbalance():
    report = make_report(
        task="classification",
        test_score=0.55,
        cv_score=0.54,
        train_score=0.58,
        scramble_mean=0.50,
        extra_test_metrics={"mcc": 0.05, "balanced_accuracy": 0.52},
    )
    detection = TaskDetection(
        task="classification",
        reason="binary",
        n_unique_values=2,
        n_classes=2,
        class_counts={"0": 100, "1": 100},
        imbalance_ratio=1.0,
    )
    criteria = AcceptanceCriteria(min_test_mcc=0.30)
    diagnosis = diagnose_failure(report, evaluate_acceptance(criteria, report), detection)
    assert diagnosis.label != "class_imbalance"


def test_diagnosis_sensitivity_is_configurable():
    report = make_report(cv_score=0.50, train_score=0.65, test_score=0.48, cv_std=0.02)
    criteria = AcceptanceCriteria(min_test_r2=0.6)
    lenient = diagnose_failure(report, evaluate_acceptance(criteria, report),
                               thresholds=DiagnosisThresholds(overfit_gap=0.50))
    strict = diagnose_failure(report, evaluate_acceptance(criteria, report),
                              thresholds=DiagnosisThresholds(overfit_gap=0.05))
    assert strict.label == "overfit"
    assert lenient.label != "overfit"


def test_every_diagnosis_offers_at_least_one_action():
    report = make_report(cv_score=0.30, train_score=0.34, test_score=0.28)
    criteria = AcceptanceCriteria(min_test_r2=0.6)
    diagnosis = diagnose_failure(report, evaluate_acceptance(criteria, report))
    assert diagnosis.suggested_actions
