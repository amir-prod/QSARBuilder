"""Tests for the modeling handoff package."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from qsar_agent.schemas.handoff import (
    AcceptanceCriteria,
    AgentConstraints,
    ArtifactIndex,
    DatasetAudit,
    DiagnosticFlags,
    DomainPerformance,
    DuplicateOverlap,
    ErrorAnalysis,
    ExperimentMetrics,
    ExperimentRecord,
    GitProvenance,
    HandoffPackage,
    LeakageSafeguards,
    PlotReference,
    ProblemDefinition,
    RandomSeeds,
    RepresentationPreprocessing,
    ResidualDiagnostics,
    RunMetadata,
    ValidationDesign,
    WinnerADResults,
    WorkflowConclusion,
)
from qsar_agent.schemas.hyperparameter_optimization import FoldMetrics
from qsar_agent.schemas.modeling import Metrics
from qsar_agent.services.handoff import (
    HandoffValidationError,
    drop_placeholder_cv_errors,
    fill_cv_error_metrics_from_folds,
    finalize_train_cv_gap_metrics,
    format_metric,
    render_modeling_handoff_md,
    validate_handoff_package,
    write_experiment_ledger_csv,
)
from qsar_agent.services.plotting import plot_residuals


def _plot(name: str, *, available: bool = False, path: str | None = None, reason: str = "not generated") -> PlotReference:
    if available:
        return PlotReference(name=name, status="available", relative_path=path)
    return PlotReference(name=name, status="unavailable", reason=reason)


def _experiment(
    run_id: str,
    *,
    train_r2: float = 0.81,
    cv_r2: float = 0.72,
    val_r2: float = 0.70,
    gap: float = 0.09,
    std: float = 0.04,
    is_winner: bool = True,
    plots: dict[str, PlotReference] | None = None,
    extra_paths: dict | None = None,
) -> ExperimentRecord:
    plots = plots or {
        "observed_vs_predicted": _plot("observed_vs_predicted"),
        "williams": _plot("williams"),
        "residuals": _plot("residuals"),
    }
    extra_paths = extra_paths or {}
    return ExperimentRecord(
        run_id=run_id,
        representation="RDKit, Mordred",
        feature_selection_method="ga",
        model="RandomForestRegressor",
        hyperparameters={"n_estimators": 100},
        feature_count=5,
        selected_feature_names=["a", "b", "c", "d", "e"],
        metrics=ExperimentMetrics(
            train_r2=train_r2,
            train_rmse=0.2,
            train_mae=0.15,
            cv_r2=cv_r2,
            cv_rmse=0.25,
            cv_mae=0.18,
            cv_r2_std=std,
            train_cv_r2_gap=gap,
            val_r2=val_r2,
            val_rmse=0.27,
            val_mae=0.19,
            refit_train_r2=train_r2,
            mean_cv_fold_train_r2=cv_r2 + gap,
            oof_cv_r2=cv_r2,
            refit_train_cv_gap=train_r2 - cv_r2,
            cv_fold_train_val_gap=gap,
        ),
        diagnostic_flags=DiagnosticFlags(status="good", is_acceptable=True),
        status="completed",
        artifacts=ArtifactIndex(
            observed_vs_predicted=plots["observed_vs_predicted"],
            williams=plots["williams"],
            residuals=plots["residuals"],
            cv_predictions=extra_paths.get("cv_predictions"),
            test_predictions=extra_paths.get("test_predictions"),
            config=extra_paths.get("config"),
            pipeline=extra_paths.get("pipeline"),
        ),
        is_winner=is_winner,
    )


def _package(experiments: list[ExperimentRecord] | None = None, **kwargs) -> HandoffPackage:
    experiments = experiments or [_experiment("abc123__random_forest__ga")]
    winner_id = next((e.run_id for e in experiments if e.is_winner), experiments[0].run_id)
    data = dict(
        run_metadata=RunMetadata(
            run_id="abc123",
            started_at="2026-01-01T00:00:00+00:00",
            completed_at="2026-01-01T01:00:00+00:00",
            git=GitProvenance(available=False, reason="not a git repository"),
            random_seeds=RandomSeeds(workflow=42, sfs=42, ga=42, hpo=42, model=42, clustering=60),
            configuration={"random_seed": 42},
            package_versions={"python": "3.11.0"},
            workflow_status="completed",
        ),
        problem_definition=ProblemDefinition(
            task="regression",
            target="pIC50",
            target_transformation="identity",
            units="unspecified",
            primary_metric="r2",
            acceptance_criteria=AcceptanceCriteria(
                primary_metric="r2",
                minimum_cv_r2=0.5,
                overfit_gap_threshold=0.15,
                severe_overfit_gap_threshold=0.25,
                cv_std_threshold=0.15,
                minimum_train_r2=0.4,
                min_cv_improvement=0.02,
            ),
        ),
        dataset_audit=DatasetAudit(
            invalid_structures=0,
            duplicates=0,
            missing_or_invalid_activity=0,
            descriptors_with_missing=0,
            train_size=20,
            validation_size=5,
            test_size=5,
            split_strategy="umap_cluster",
            dataset_hash="deadbeef",
            duplicate_overlap=DuplicateOverlap(),
        ),
        leakage_safeguards=LeakageSafeguards(
            test_lock_status="locked_from_selection",
            test_compound_id_hash="abc",
            preprocessing_scope="train_only_fit",
            feature_selection_scope="train_cv_and_holdout_validation",
            duplicate_overlap=DuplicateOverlap(),
            test_results_used_for_selection=False,
            selection_criterion="combined R2",
            confirmation="test unused",
            selection_records=[{"estimator": "RF", "mean_cv_r2": 0.7, "holdout_val_r2": 0.68}],
        ),
        representation_preprocessing=RepresentationPreprocessing(
            descriptor_backends=["RDKit"],
            fingerprint_settings={"enabled": False, "types": [], "note": "none"},
            scaling="StandardScaler",
            imputation="median",
            pipeline_order=["median_imputation", "standard_scaler"],
        ),
        validation_design=ValidationDesign(
            cv_method="KFold",
            folds=5,
            repeats=1,
            shuffle=True,
            seed=42,
            tuning_method="none",
            search_budget=0,
            optimization_metric="r2",
            combined_score_description="0.5 CV + 0.5 val",
        ),
        experiments=experiments,
        applicability_domain=WinnerADResults(
            winner_run_id=winner_id,
            method="williams_leverage",
            residual_threshold=3.0,
            structural_outlier_count=0,
            response_outlier_count=0,
            handling_decision="informational_only",
            handling_justification="diagnostic only",
        ),
        error_analysis=ErrorAnalysis(
            winner_run_id=winner_id,
            inside_domain=DomainPerformance(n=20, r2=0.7),
            outside_domain=DomainPerformance(n=0),
            residual_diagnostics=ResidualDiagnostics(mean=0.0, std=0.1),
        ),
        conclusion=WorkflowConclusion(
            best_run_id=winner_id,
            selection_criterion="combined R2",
            acceptance_status=True,
            winner_model="RandomForestRegressor",
            winner_cv_r2=0.72,
            winner_train_metrics=Metrics(r2=0.81, rmse=0.2, mae=0.15, n_samples=20),
        ),
        agent_constraints=AgentConstraints(
            permitted_actions=["explain feature count"],
            prohibited_actions=["use test set"],
            iteration_budget={"max_hpo_rounds": 3},
            compute_budget={"max_candidates_per_round": 120},
            stopping_conditions=["max rounds"],
        ),
    )
    data.update(kwargs)
    return HandoffPackage(**data)


def test_handoff_package_top_level_integrity_fields_default():
    package = _package()
    assert package.schema_version == "1.0"
    assert package.handoff_status == "COMPLETE"
    assert package.validation_passed is True
    assert package.validation_errors == []
    dumped = package.model_dump(mode="json")
    assert "dataset_hash" in dumped
    assert "development_split_hash" in dumped
    assert "sealed_test_hash" in dumped


def _write_views(tmp_path: Path, package: HandoffPackage) -> Path:
    report_dir = tmp_path / "final_report"
    report_dir.mkdir()
    (report_dir / "handoff_manifest.json").write_text(
        json.dumps(package.model_dump(mode="json"), indent=2), encoding="utf-8"
    )
    (report_dir / "modeling_handoff.md").write_text(
        render_modeling_handoff_md(package), encoding="utf-8"
    )
    write_experiment_ledger_csv(package, report_dir / "experiment_ledger.csv")
    return report_dir


def test_render_metrics_agree_across_md_json_csv(tmp_path):
    package = _package()
    report_dir = _write_views(tmp_path, package)
    validate_handoff_package(report_dir, package)
    exp = package.experiments[0]
    csv_text = (report_dir / "experiment_ledger.csv").read_text(encoding="utf-8")
    md_text = (report_dir / "modeling_handoff.md").read_text(encoding="utf-8")
    assert format_metric(exp.metrics.train_r2) in csv_text
    assert format_metric(exp.metrics.cv_r2) in csv_text
    assert f"run_id={exp.run_id}" in md_text
    assert format_metric(exp.metrics.train_r2) in md_text
    raw = json.loads((report_dir / "handoff_manifest.json").read_text(encoding="utf-8"))
    reloaded = HandoffPackage.model_validate(raw)
    assert format_metric(reloaded.experiments[0].metrics.train_r2) == format_metric(exp.metrics.train_r2)


def test_unique_run_ids(tmp_path):
    dup = _experiment("same_id", is_winner=True)
    other = _experiment("same_id", is_winner=False, train_r2=0.5)
    package = _package(experiments=[dup, other])
    report_dir = _write_views(tmp_path, package)
    with pytest.raises(HandoffValidationError, match="not unique"):
        validate_handoff_package(report_dir, package)


def test_missing_referenced_plot_fails(tmp_path):
    plots = {
        "observed_vs_predicted": _plot(
            "observed_vs_predicted", available=True, path="plots/missing.png"
        ),
        "williams": _plot("williams"),
        "residuals": _plot("residuals"),
    }
    package = _package(experiments=[_experiment("run1", plots=plots)])
    report_dir = _write_views(tmp_path, package)
    with pytest.raises(HandoffValidationError, match="Referenced file missing"):
        validate_handoff_package(report_dir, package)


def test_unavailable_plot_is_not_linked(tmp_path):
    reason = "Williams plot was not generated for this run."
    plots = {
        "observed_vs_predicted": _plot("observed_vs_predicted"),
        "williams": _plot("williams", reason=reason),
        "residuals": _plot("residuals"),
    }
    package = _package(experiments=[_experiment("run1", plots=plots)])
    md = render_modeling_handoff_md(package)
    assert "plots/run1_williams.png" not in md
    assert reason in md
    report_dir = _write_views(tmp_path, package)
    validate_handoff_package(report_dir, package)


def test_selection_records_reject_test_metrics(tmp_path):
    package = _package()
    package.leakage_safeguards.selection_records = [
        {"estimator": "RF", "mean_cv_r2": 0.7, "test_r2": 0.9}
    ]
    report_dir = _write_views(tmp_path, package)
    with pytest.raises(HandoffValidationError, match="External-test metric"):
        validate_handoff_package(report_dir, package)


def test_plot_residuals_writes_files(tmp_path):
    png = tmp_path / "residuals.png"
    svg = tmp_path / "residuals.svg"
    plot_residuals(
        [1.0, 2.0, 3.0],
        [0.1, -0.2, 0.0],
        ["train", "val", "test"],
        png,
        svg,
    )
    assert png.is_file()
    assert svg.is_file()
    assert png.stat().st_size > 0


def test_placeholder_cv_errors_are_filled_from_folds():
    metrics = ExperimentMetrics(cv_r2=0.316284, cv_rmse=0.0, cv_mae=0.0, cv_r2_std=0.139249)
    folds = [
        FoldMetrics(
            fold=1, train_r2=0.4, val_r2=0.09, train_rmse=0.78, val_rmse=0.70,
            train_mae=0.6, val_mae=0.50,
        ),
        FoldMetrics(
            fold=2, train_r2=0.35, val_r2=0.45, train_rmse=0.78, val_rmse=0.80,
            train_mae=0.6, val_mae=0.60,
        ),
    ]
    fill_cv_error_metrics_from_folds(metrics, folds)
    assert metrics.cv_rmse == pytest.approx(0.75)
    assert metrics.cv_mae == pytest.approx(0.55)
    filled = _experiment("run_filled")
    filled.metrics.cv_r2 = 0.316284
    filled.metrics.oof_cv_r2 = 0.316284
    filled.metrics.cv_rmse = 0.75
    filled.metrics.cv_mae = 0.55
    filled.metrics.cv_r2_std = 0.139249
    text = render_modeling_handoff_md(_package(experiments=[filled]))
    assert "OOF CV r2/rmse/mae (r2 std): 0.316284 / 0.750000 / 0.550000 (0.139249)" in text


def test_placeholder_cv_errors_become_none_without_folds():
    metrics = ExperimentMetrics(cv_r2=0.316284, cv_rmse=0.0, cv_mae=0.0)
    drop_placeholder_cv_errors(metrics)
    assert metrics.cv_rmse is None
    assert metrics.cv_mae is None
    exp = _experiment("run1")
    exp.metrics.cv_rmse = None
    exp.metrics.cv_mae = None
    exp.metrics.cv_r2 = 0.316284
    exp.metrics.oof_cv_r2 = 0.316284
    text = render_modeling_handoff_md(_package(experiments=[exp]))
    assert "OOF CV r2/rmse/mae (r2 std): 0.316284 / null / null" in text


def test_refit_and_fold_train_cv_gaps_are_distinct():
    metrics = ExperimentMetrics(
        train_r2=0.733866,
        cv_r2=0.534286,
        train_cv_r2_gap=0.343049,
        mean_cv_fold_train_r2=0.877335,
        cv_fold_train_val_gap=0.343049,
    )
    folds = [
        FoldMetrics(fold=1, train_r2=0.881815, val_r2=0.475401, train_rmse=0.3, val_rmse=0.5, train_mae=0.2, val_mae=0.4),
        FoldMetrics(fold=2, train_r2=0.879004, val_r2=0.592878, train_rmse=0.3, val_rmse=0.6, train_mae=0.2, val_mae=0.4),
        FoldMetrics(fold=3, train_r2=0.868414, val_r2=0.570995, train_rmse=0.3, val_rmse=0.6, train_mae=0.2, val_mae=0.4),
        FoldMetrics(fold=4, train_r2=0.874549, val_r2=0.584773, train_rmse=0.3, val_rmse=0.6, train_mae=0.2, val_mae=0.4),
        FoldMetrics(fold=5, train_r2=0.882895, val_r2=0.447383, train_rmse=0.3, val_rmse=0.7, train_mae=0.2, val_mae=0.4),
    ]
    finalize_train_cv_gap_metrics(metrics, folds)
    assert metrics.refit_train_r2 == pytest.approx(0.733866)
    assert metrics.mean_cv_fold_train_r2 == pytest.approx(0.877335, abs=1e-6)
    assert metrics.oof_cv_r2 == pytest.approx(0.534286)
    assert metrics.refit_train_cv_gap == pytest.approx(0.199580, abs=1e-6)
    assert metrics.cv_fold_train_val_gap == pytest.approx(0.343049, abs=1e-6)
    assert metrics.train_cv_r2_gap == metrics.cv_fold_train_val_gap
    exp = _experiment("rf_ga")
    exp.metrics = metrics
    text = render_modeling_handoff_md(_package(experiments=[exp]))
    assert "`cv_fold_train_val_gap`, used for acceptance" in text
    assert format_metric(metrics.refit_train_cv_gap) in text
    assert format_metric(metrics.cv_fold_train_val_gap) in text
    assert "overfit gap statistic: `cv_fold_train_val_gap`" in text


def test_git_provenance_handles_non_repo(tmp_path):
    from qsar_agent.tools.provenance import collect_package_versions, get_git_provenance

    git = get_git_provenance(tmp_path)
    assert git.available is False
    assert git.reason
    versions = collect_package_versions()
    assert "python" in versions
    assert "scikit-learn" in versions


def test_error_analysis_largest_errors_and_domain(tmp_path):
    import pandas as pd

    from qsar_agent.tools.error_analysis import analyze_errors

    pred = pd.DataFrame(
        {
            "compound_id": ["a", "b", "c", "d"],
            "activity": [1.0, 2.0, 3.0, 4.0],
            "predicted_activity": [1.1, 2.2, 2.0, 4.1],
            "split": ["train", "train", "test", "test"],
            "residual": [-0.1, -0.2, 1.0, -0.1],
        }
    )
    ad = pd.DataFrame(
        {
            "compound_id": ["a", "b", "c", "d"],
            "applicability_domain": ["in_domain", "in_domain", "high_leverage", "in_domain"],
            "in_domain": [True, True, False, True],
        }
    )
    pred_path = tmp_path / "pred.csv"
    ad_path = tmp_path / "ad.csv"
    pred.to_csv(pred_path, index=False)
    ad.to_csv(ad_path, index=False)
    result = analyze_errors(pred_path, ad_path, "run1", top_n=2)
    assert result.largest_error_compounds[0].compound_id == "c"
    assert result.inside_domain.n == 3
    assert result.outside_domain.n == 1

