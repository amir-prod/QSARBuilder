"""Assemble, render, and validate the modeling handoff package."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import shutil
from pathlib import Path
from typing import Any, Iterable

import joblib
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline

from qsar_agent.config import ModelConfig, WorkflowConfig
from qsar_agent.models.registry import estimator_slug, normalize_estimator_name
from qsar_agent.schemas.dataset import DatasetValidationResult
from qsar_agent.schemas.descriptors import DescriptorCalculationResult
from qsar_agent.schemas.feature_selection import FeatureCountSelection, GAResult
from qsar_agent.schemas.handoff import (
    AcceptanceCriteria,
    AgentConstraints,
    ArtifactIndex,
    CompletedSearch,
    CurationStep,
    DatasetAudit,
    DiagnosticFlags,
    DuplicateOverlap,
    ExperimentAD,
    ExperimentMetrics,
    ExperimentRecord,
    ExternalTestMetrics,
    HandoffPackage,
    LeakageSafeguards,
    PlotReference,
    ProblemDefinition,
    RandomSeeds,
    RepresentationPreprocessing,
    RunMetadata,
    StageStatusRecord,
    ValidationDesign,
    WinnerADResults,
    WorkflowConclusion,
    OVERFIT_GAP_DEFINITION,
    OVERFIT_GAP_STATISTIC,
)
from qsar_agent.schemas.hyperparameter_optimization import FoldMetrics
from qsar_agent.schemas.model_fallback import (
    BranchExternalArtifacts,
    CrossModelSelection,
    ModelBranchResult,
)
from qsar_agent.schemas.modeling import Metrics
from qsar_agent.schemas.preprocessing import PreprocessingResult
from qsar_agent.schemas.split import SplitResult
from qsar_agent.schemas.workflow import WorkflowState
from qsar_agent.services import build_estimator
from qsar_agent.services.artifact_manager import save_json
from qsar_agent.services.plotting import plot_residuals
from qsar_agent.tools.branch_external_evaluation import branch_display_label
from qsar_agent.tools.combined_score import CV_VAL_WEIGHT
from qsar_agent.tools.error_analysis import analyze_errors
from qsar_agent.tools.provenance import collect_package_versions, get_git_provenance

METRIC_DECIMALS = 6
AD_HANDLING_DECISION = "informational_only"
AD_HANDLING_JUSTIFICATION = (
    "Williams-plot applicability domain is a diagnostic report. "
    "Structural and response outliers were not excluded from training "
    "and were not used for model selection."
)
PREPROCESSING_PIPELINE_ORDER = [
    "drop_nonnumeric_descriptors",
    "drop_high_missing_fraction",
    "median_imputation",
    "drop_constant_descriptors",
    "drop_near_constant_descriptors",
    "drop_highly_correlated_descriptors",
    "standard_scaler",
]
TEST_METRIC_KEYS = {
    "test_r2",
    "test_rmse",
    "test_mae",
    "mean_test_r2",
    "external_test_r2",
    "holdout_test_r2",
}
SELECTION_CRITERION = (
    "Highest combined R² (equal-weight mean training CV R² and holdout validation R²) "
    "among acceptable models, with a one-standard-error rule and estimator-simplicity "
    "tie-break. External-test metrics were not used for model selection."
)
CANONICAL_RE = re.compile(
    r"<!-- canonical_metrics run_id=(?P<run_id>\S+)"
    r" train_r2=(?P<train_r2>\S+)"
    r" cv_r2=(?P<cv_r2>\S+)"
    r" val_r2=(?P<val_r2>\S+)"
    r" train_cv_r2_gap=(?P<train_cv_r2_gap>\S+)"
    r" cv_fold_train_val_gap=(?P<cv_fold_train_val_gap>\S+)"
    r" refit_train_cv_gap=(?P<refit_train_cv_gap>\S+)"
    r" mean_cv_fold_train_r2=(?P<mean_cv_fold_train_r2>\S+)"
    r" cv_r2_std=(?P<cv_r2_std>\S+) -->"
)


class HandoffValidationError(ValueError):
    """Raised when the handoff package fails completeness or consistency checks."""


class ColumnSelector(BaseEstimator, TransformerMixin):
    """Select named descriptor columns from a preprocessed DataFrame."""

    def __init__(self, columns: list[str] | None = None):
        self.columns = list(columns or [])

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            raise TypeError("ColumnSelector expects a pandas DataFrame of preprocessed descriptors.")
        return X.loc[:, self.columns]


def format_metric(value: float | None) -> str:
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return "null"
    return f"{float(value):.{METRIC_DECIMALS}f}"


def _is_placeholder_cv_error(error: float | None, r2: float | None) -> bool:
    """HPO search stores RMSE/MAE as 0.0 because GridSearchCV only scores R²."""
    if error is None or error != 0.0:
        return False
    return r2 is None or r2 < 1.0 - 1e-12


def fill_cv_error_metrics_from_folds(
    metrics: ExperimentMetrics,
    folds: list[FoldMetrics],
) -> ExperimentMetrics:
    """Replace placeholder CV RMSE/MAE with means of per-fold validation errors."""
    if not folds:
        return drop_placeholder_cv_errors(metrics)
    metrics.cv_rmse = float(sum(f.val_rmse for f in folds) / len(folds))
    metrics.cv_mae = float(sum(f.val_mae for f in folds) / len(folds))
    return metrics


def drop_placeholder_cv_errors(metrics: ExperimentMetrics) -> ExperimentMetrics:
    if _is_placeholder_cv_error(metrics.cv_rmse, metrics.cv_r2):
        metrics.cv_rmse = None
    if _is_placeholder_cv_error(metrics.cv_mae, metrics.cv_r2):
        metrics.cv_mae = None
    return metrics


def finalize_train_cv_gap_metrics(
    metrics: ExperimentMetrics,
    folds: list[FoldMetrics],
) -> ExperimentMetrics:
    """Populate explicit refit vs in-fold training gaps from stored scores."""
    metrics.refit_train_r2 = metrics.train_r2
    metrics.oof_cv_r2 = metrics.cv_r2
    if folds:
        metrics.mean_cv_fold_train_r2 = float(sum(f.train_r2 for f in folds) / len(folds))
        mean_oof = float(sum(f.val_r2 for f in folds) / len(folds))
        metrics.cv_fold_train_val_gap = metrics.mean_cv_fold_train_r2 - mean_oof
    elif metrics.cv_fold_train_val_gap is None:
        metrics.cv_fold_train_val_gap = metrics.train_cv_r2_gap
    if (
        metrics.mean_cv_fold_train_r2 is None
        and metrics.train_cv_r2_gap is not None
        and metrics.cv_r2 is not None
    ):
        metrics.mean_cv_fold_train_r2 = float(metrics.cv_r2) + float(metrics.train_cv_r2_gap)
    if metrics.train_r2 is not None and metrics.cv_r2 is not None:
        metrics.refit_train_cv_gap = float(metrics.train_r2) - float(metrics.cv_r2)
    if metrics.cv_fold_train_val_gap is not None:
        metrics.train_cv_r2_gap = metrics.cv_fold_train_val_gap
    return metrics


def write_and_validate_handoff(
    *,
    run_dir: Path,
    run_id: str,
    started_at: str,
    completed_at: str,
    config: WorkflowConfig,
    state: WorkflowState,
    validation: DatasetValidationResult,
    descriptors: DescriptorCalculationResult,
    split: SplitResult,
    preprocessing: PreprocessingResult,
    feature_count: FeatureCountSelection,
    ga: GAResult,
    rf_branch: ModelBranchResult,
    fallback_branches: list[ModelBranchResult],
    branch_external_artifacts: list[BranchExternalArtifacts],
    winning_estimator: str,
    winning_features: list[str],
    winner_is_expansion: bool,
    winner_expansion_label: str,
    model_comparison_summary: str,
    cross_model_selection: CrossModelSelection | None,
    dataset_hash: str,
    warnings: list[str],
) -> HandoffPackage:
    """Build the handoff object, write artifacts, render views, and validate."""
    report_dir = Path(run_dir) / "final_report"
    plots_dir = report_dir / "plots"
    preds_dir = report_dir / "predictions"
    configs_dir = report_dir / "configs"
    models_dir = report_dir / "models"
    for folder in (plots_dir, preds_dir, configs_dir, models_dir):
        folder.mkdir(parents=True, exist_ok=True)

    git = get_git_provenance()
    versions = collect_package_versions()
    seeds = RandomSeeds(
        workflow=config.random_seed,
        sfs=config.sfs.random_seed,
        ga=config.ga.random_seed,
        hpo=config.random_seed,
        model=config.model.random_state,
        clustering=config.clustering.random_state,
    )
    overlap = _duplicate_overlap(split)
    test_id_hash = _hash_compound_ids(split.test_path)
    preprocessor_rel = _copy_preprocessor(preprocessing.preprocessor_path, models_dir)

    selection_records = _collect_selection_records(cross_model_selection, rf_branch)
    package_experiments: list[ExperimentRecord] = []
    used_ids: set[str] = set()
    art_by_dir = {
        str(Path(a.branch_dir).resolve()): a
        for a in branch_external_artifacts
        if a.branch_dir
    }

    branches = list(_iter_branch_variants(rf_branch, *fallback_branches))
    winner_branch = _match_winner(
        branches,
        winning_estimator=winning_estimator,
        winning_features=winning_features,
        winner_is_expansion=winner_is_expansion,
        winner_expansion_label=winner_expansion_label,
    )

    backends = list(descriptors.backends)
    representation = ", ".join(backends) if backends else "molecular_descriptors"
    train_path = Path(preprocessing.preprocessed_train_path)

    for branch in branches:
        experiment = _build_experiment(
            branch=branch,
            artifacts=art_by_dir.get(str(Path(branch.branch_dir).resolve())) if branch.branch_dir else None,
            workflow_run_id=run_id,
            used_ids=used_ids,
            representation=representation,
            report_dir=report_dir,
            train_path=train_path,
            config=config,
            is_winner=branch is winner_branch,
        )
        package_experiments.append(experiment)

    if winner_branch is None and package_experiments:
        package_experiments[0].is_winner = True
    winner = next((e for e in package_experiments if e.is_winner), None)
    if winner is None and package_experiments:
        winner = package_experiments[0]
        winner.is_winner = True
    if winner is None:
        raise HandoffValidationError("No experiments available for the modeling handoff.")

    winner_ad = winner.applicability_domain or ExperimentAD(
        method="williams_leverage",
        handling_decision=AD_HANDLING_DECISION,
        handling_justification=AD_HANDLING_JUSTIFICATION,
    )
    ad_results = WinnerADResults(
        winner_run_id=winner.run_id,
        method=winner_ad.method,
        warning_leverage=winner_ad.warning_leverage,
        residual_threshold=winner_ad.residual_threshold or 3.0,
        structural_outlier_count=winner_ad.structural_outlier_count,
        response_outlier_count=winner_ad.response_outlier_count,
        structural_outlier_ids=list(winner_ad.structural_outlier_ids),
        response_outlier_ids=list(winner_ad.response_outlier_ids),
        outliers_by_partition=dict(winner_ad.outliers_by_partition),
        handling_decision=winner_ad.handling_decision,
        handling_justification=winner_ad.handling_justification,
    )

    winner_pred = ""
    winner_ad_csv = ""
    if winner.artifacts.source_predictions and Path(winner.artifacts.source_predictions).exists():
        winner_pred = winner.artifacts.source_predictions
    winner_art = None
    if winner_branch is not None and winner_branch.branch_dir:
        winner_art = art_by_dir.get(str(Path(winner_branch.branch_dir).resolve()))
    if winner_art and winner_art.ad_classifications_path:
        winner_ad_csv = winner_art.ad_classifications_path
    if winner_pred:
        error_analysis = analyze_errors(winner_pred, winner_ad_csv or None, winner.run_id)
    else:
        error_analysis = _empty_error_analysis(winner.run_id)

    if warnings:
        winner.warnings = list(winner.warnings) + list(warnings)

    save_json(configs_dir / "workflow_config.json", config.to_dict())

    winner_fs = None
    if winner_branch is not None and winner_branch.hpo_result.final_selection is not None:
        winner_fs = winner_branch.hpo_result.final_selection
    assessment = winner_fs.assessment if winner_fs is not None else None
    cv_summary = winner_fs.cv_summary if winner_fs is not None else None
    winner_train = None
    winner_val = None
    if winner_art and winner_art.metrics_path and Path(winner_art.metrics_path).exists():
        metrics_blob = json.loads(Path(winner_art.metrics_path).read_text(encoding="utf-8"))
        if "train" in metrics_blob:
            winner_train = Metrics(**metrics_blob["train"])
        if "val" in metrics_blob:
            winner_val = Metrics(**metrics_blob["val"])

    hpo = config.hpo
    fp_types = [b for b in backends if _looks_like_fingerprint(b)]
    package = HandoffPackage(
        run_metadata=RunMetadata(
            run_id=run_id,
            started_at=started_at,
            completed_at=completed_at,
            git=git,
            random_seeds=seeds,
            configuration=config.to_dict(),
            package_versions=versions,
            workflow_status="completed",
            stages=[
                StageStatusRecord(stage=s.stage, status=s.status.value, message=s.message)
                for s in state.stages
            ],
        ),
        problem_definition=ProblemDefinition(
            task="regression",
            target=config.activity_column or "activity",
            target_transformation="identity",
            units=(config.activity_units.strip() or "unspecified"),
            primary_metric="r2",
            acceptance_criteria=AcceptanceCriteria(
                primary_metric="r2",
                minimum_cv_r2=hpo.minimum_cv_r2,
                overfit_gap_threshold=hpo.overfit_gap_threshold,
                severe_overfit_gap_threshold=hpo.severe_overfit_gap_threshold,
                cv_std_threshold=hpo.cv_std_threshold,
                minimum_train_r2=hpo.minimum_train_r2,
                min_cv_improvement=hpo.min_cv_improvement,
                overfit_gap_statistic=OVERFIT_GAP_STATISTIC,
                overfit_gap_definition=OVERFIT_GAP_DEFINITION,
                minimum_train_r2_scope="mean_cv_fold_train_r2",
            ),
        ),
        dataset_audit=_dataset_audit(
            validation, descriptors, split, preprocessing, dataset_hash, overlap,
            selected_feature_count=feature_count.selected_feature_count,
            ga_feature_count=len(ga.selected_features),
        ),
        leakage_safeguards=LeakageSafeguards(
            test_lock_status="locked_from_selection",
            test_compound_id_hash=test_id_hash,
            preprocessing_scope="train_only_fit",
            feature_selection_scope="train_cv_and_holdout_validation",
            duplicate_overlap=overlap,
            test_results_used_for_selection=False,
            selection_criterion=SELECTION_CRITERION,
            confirmation=(
                "Model selection used training cross-validation and held-out validation only. "
                "External-test predictions were generated after selection and did not influence "
                "the winning run."
            ),
            selection_records=selection_records,
        ),
        representation_preprocessing=RepresentationPreprocessing(
            descriptor_backends=backends,
            fingerprint_settings={
                "enabled": bool(fp_types),
                "types": fp_types,
                "note": (
                    "Hashed fingerprints were not used; representation is 2D molecular descriptors."
                    if not fp_types
                    else "Fingerprint backends listed in types."
                ),
            },
            geometry_optimization=descriptors.run_geometry_optimization,
            three_d_descriptors_included=descriptors.three_d_descriptors_included,
            filters={
                "missing_value_threshold": config.preprocessing.missing_value_threshold,
                "near_constant_std_threshold": config.preprocessing.near_constant_std_threshold,
                "correlation_threshold": config.preprocessing.correlation_threshold,
            },
            scaling="StandardScaler",
            imputation="median",
            pipeline_order=list(PREPROCESSING_PIPELINE_ORDER),
            preprocessor_relative_path=preprocessor_rel,
        ),
        validation_design=ValidationDesign(
            cv_method="KFold",
            folds=int(hpo.cv_folds or config.sfs.cv_folds),
            repeats=1,
            shuffle=True,
            seed=config.random_seed,
            tuning_method="GridSearchCV" if hpo.enabled else "none",
            search_budget=int(hpo.max_candidates_per_round) if hpo.enabled else 0,
            optimization_metric="r2",
            combined_score_description=(
                f"{CV_VAL_WEIGHT}·mean training CV R² + {1.0 - CV_VAL_WEIGHT}·holdout validation R²"
            ),
        ),
        experiments=package_experiments,
        applicability_domain=ad_results,
        error_analysis=error_analysis,
        conclusion=WorkflowConclusion(
            best_run_id=winner.run_id,
            selection_criterion=SELECTION_CRITERION,
            acceptance_status=bool(assessment.is_acceptable) if assessment is not None else False,
            failed_criteria=_failed_criteria(assessment),
            completed_searches=[
                CompletedSearch(name=s.stage, completed=s.status.value == "Completed", detail=s.message)
                for s in state.stages
            ],
            winner_model=winner.model,
            winner_train_metrics=winner_train,
            winner_cv_r2=None if cv_summary is None else cv_summary.mean_cv_r2,
            winner_val_metrics=winner_val,
        ),
        agent_constraints=_agent_constraints(config),
    )

    manifest_path = report_dir / "handoff_manifest.json"
    md_path = report_dir / "modeling_handoff.md"
    csv_path = report_dir / "experiment_ledger.csv"
    save_json(manifest_path, package.model_dump(mode="json"))
    md_path.write_text(render_modeling_handoff_md(package), encoding="utf-8")
    write_experiment_ledger_csv(package, csv_path)
    validate_handoff_package(report_dir, package)
    return package


def render_modeling_handoff_md(package: HandoffPackage) -> str:
    """Render Markdown using only fields from the structured package."""
    meta = package.run_metadata
    problem = package.problem_definition
    acc = problem.acceptance_criteria
    audit = package.dataset_audit
    leak = package.leakage_safeguards
    prep = package.representation_preprocessing
    val = package.validation_design
    ad = package.applicability_domain
    err = package.error_analysis
    conc = package.conclusion
    agent = package.agent_constraints
    git_line = meta.git.commit if meta.git.available else f"unavailable ({meta.git.reason})"
    lines: list[str] = [
        "# Modeling Handoff",
        "",
        "## Run metadata",
        "",
        f"- Run ID: `{meta.run_id}`",
        f"- Started: {meta.started_at}",
        f"- Completed: {meta.completed_at}",
        f"- Git commit: `{git_line}`",
        f"- Git dirty: {meta.git.dirty}",
        f"- Workflow status: {meta.workflow_status}",
        f"- Seeds: workflow={meta.random_seeds.workflow}, sfs={meta.random_seeds.sfs}, "
        f"ga={meta.random_seeds.ga}, hpo={meta.random_seeds.hpo}, "
        f"model={meta.random_seeds.model}, clustering={meta.random_seeds.clustering}",
        "- Package versions:",
    ]
    for name, version in meta.package_versions.items():
        lines.append(f"  - {name}: `{version}`")
    lines.extend(["", "### Workflow stages", ""])
    for stage in meta.stages:
        extra = f" — {stage.message}" if stage.message else ""
        lines.append(f"- `{stage.stage}`: {stage.status}{extra}")
    lines.extend(
        [
            "",
            "## Problem definition",
            "",
            f"- Task: {problem.task}",
            f"- Target: `{problem.target}`",
            f"- Target transformation: `{problem.target_transformation}`",
            f"- Units: {problem.units}",
            f"- Primary metric: `{problem.primary_metric}`",
            "- Acceptance criteria:",
            f"  - minimum CV {acc.primary_metric}: {format_metric(acc.minimum_cv_r2)}",
            f"  - overfit gap statistic: `{acc.overfit_gap_statistic}`",
            f"  - overfit gap definition: {acc.overfit_gap_definition}",
            f"  - overfit gap threshold: {format_metric(acc.overfit_gap_threshold)}",
            f"  - severe overfit gap threshold: {format_metric(acc.severe_overfit_gap_threshold)}",
            f"  - CV std threshold: {format_metric(acc.cv_std_threshold)}",
            f"  - minimum train {acc.primary_metric} (`{acc.minimum_train_r2_scope}`): "
            f"{format_metric(acc.minimum_train_r2)}",
            f"  - min CV improvement: {format_metric(acc.min_cv_improvement)}",
            "",
            "## Dataset audit",
            "",
        ]
    )
    for step in audit.curation_steps:
        bits = [f"- **{step.step}**"]
        if step.n_compounds is not None:
            bits.append(f"compounds={step.n_compounds}")
        if step.n_features is not None:
            bits.append(f"features={step.n_features}")
        if step.n_removed is not None:
            bits.append(f"removed={step.n_removed}")
        if step.notes:
            bits.append(step.notes)
        lines.append(" ".join(bits))
    lines.extend(
        [
            "",
            f"- Invalid structures: {audit.invalid_structures}",
            f"- Duplicates: {audit.duplicates}",
            f"- Missing or invalid activity: {audit.missing_or_invalid_activity}",
            f"- Descriptors with missing values: {audit.descriptors_with_missing}",
            f"- Train / validation / test sizes: {audit.train_size} / {audit.validation_size} / {audit.test_size}",
            f"- Split strategy: `{audit.split_strategy}`",
            f"- Dataset hash: `{audit.dataset_hash}`",
            "- Target statistics:",
        ]
    )
    for key, value in audit.target_statistics.items():
        lines.append(f"  - {key}: {format_metric(value) if value is not None else 'null'}")
    lines.append("- Feature counts:")
    for key, value in audit.feature_counts.items():
        lines.append(f"  - {key}: {value}")
    lines.extend(
        [
            f"- Duplicate overlap train-val: {audit.duplicate_overlap.train_val or 'none'}",
            f"- Duplicate overlap train-test: {audit.duplicate_overlap.train_test or 'none'}",
            f"- Duplicate overlap val-test: {audit.duplicate_overlap.val_test or 'none'}",
            "",
            "## Leakage safeguards",
            "",
            f"- Test-lock status: `{leak.test_lock_status}`",
            f"- Test compound-ID hash: `{leak.test_compound_id_hash}`",
            f"- Preprocessing scope: {leak.preprocessing_scope}",
            f"- Feature-selection scope: {leak.feature_selection_scope}",
            f"- Duplicate overlap present: {leak.duplicate_overlap.any_overlap}",
            f"- Test results used for model selection: {leak.test_results_used_for_selection}",
            f"- Selection criterion: {leak.selection_criterion}",
            f"- Confirmation: {leak.confirmation}",
            "",
            "## Representation and preprocessing",
            "",
            f"- Descriptor backends: {', '.join(prep.descriptor_backends) or 'none'}",
            f"- Fingerprints enabled: {prep.fingerprint_settings.get('enabled')}",
            f"- Fingerprint types: {prep.fingerprint_settings.get('types')}",
            f"- Fingerprint note: {prep.fingerprint_settings.get('note')}",
            f"- Geometry optimization: {prep.geometry_optimization}",
            f"- 3D descriptors included: {prep.three_d_descriptors_included}",
            f"- Scaling: `{prep.scaling}`",
            f"- Imputation: `{prep.imputation}`",
            f"- Preprocessor: `{prep.preprocessor_relative_path or 'unavailable'}`",
            "- Filters:",
        ]
    )
    for key, value in prep.filters.items():
        lines.append(f"  - {key}: {value}")
    lines.append("- Pipeline order:")
    for i, step in enumerate(prep.pipeline_order, start=1):
        lines.append(f"  {i}. `{step}`")
    lines.extend(
        [
            "",
            "## Validation design",
            "",
            f"- CV method: `{val.cv_method}`",
            f"- Folds: {val.folds}",
            f"- Repeats: {val.repeats}",
            f"- Shuffle: {val.shuffle}",
            f"- Seed: {val.seed}",
            f"- Tuning method: `{val.tuning_method}`",
            f"- Search budget (max candidates per round): {val.search_budget}",
            f"- Optimization metric: `{val.optimization_metric}`",
            f"- Combined score: {val.combined_score_description}",
            "",
            "## Experiment ledger",
            "",
            "| run_id | representation | feature_selection | model | n_features | refit_train_r2 | oof_cv_r2 | cv_r2_std | fold_train_val_gap | refit_train_cv_gap | val_r2 | runtime_s | status |",
            "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for exp in package.experiments:
        m = exp.metrics
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{exp.run_id}`",
                    exp.representation,
                    exp.feature_selection_method,
                    exp.model,
                    str(exp.feature_count),
                    format_metric(m.refit_train_r2 if m.refit_train_r2 is not None else m.train_r2),
                    format_metric(m.oof_cv_r2 if m.oof_cv_r2 is not None else m.cv_r2),
                    format_metric(m.cv_r2_std),
                    format_metric(m.cv_fold_train_val_gap if m.cv_fold_train_val_gap is not None else m.train_cv_r2_gap),
                    format_metric(m.refit_train_cv_gap),
                    format_metric(m.val_r2),
                    format_metric(exp.runtime_seconds),
                    exp.status,
                ]
            )
            + " |"
        )
    lines.extend(["", "## Experiment details", ""])
    for exp in package.experiments:
        lines.extend(_render_experiment_section(exp))
    lines.extend(
        [
            "## Applicability domain",
            "",
            f"- Winner run: `{ad.winner_run_id}`",
            f"- Method: `{ad.method}`",
            f"- Warning leverage h*: {format_metric(ad.warning_leverage)}",
            f"- Residual threshold: {format_metric(ad.residual_threshold)}",
            f"- Structural outliers (n={ad.structural_outlier_count}): {ad.structural_outlier_ids or 'none'}",
            f"- Response outliers (n={ad.response_outlier_count}): {ad.response_outlier_ids or 'none'}",
            f"- Handling decision: `{ad.handling_decision}`",
            f"- Justification: {ad.handling_justification}",
            "- Outliers by partition:",
        ]
    )
    for split_name, groups in ad.outliers_by_partition.items():
        lines.append(
            f"  - {split_name}: structural={groups.get('structural') or 'none'}; "
            f"response={groups.get('response') or 'none'}"
        )
    lines.extend(
        [
            "",
            "## Error analysis",
            "",
            f"- Winner run: `{err.winner_run_id}`",
            "- Largest-error compounds:",
        ]
    )
    if not err.largest_error_compounds:
        lines.append("  - none")
    for row in err.largest_error_compounds:
        lines.append(
            f"  - `{row.compound_id}` ({row.split}): activity={format_metric(row.activity)}, "
            f"predicted={format_metric(row.predicted_activity)}, "
            f"|residual|={format_metric(row.abs_residual)}, AD={row.applicability_domain or 'n/a'}"
        )
    lines.append("- Target-range performance:")
    for row in err.target_range_performance:
        lines.append(
            f"  - {row.range_label}: n={row.n}, r2={format_metric(row.r2)}, "
            f"rmse={format_metric(row.rmse)}, mae={format_metric(row.mae)}"
        )
    lines.extend(
        [
            f"- Inside domain: n={err.inside_domain.n}, r2={format_metric(err.inside_domain.r2)}, "
            f"rmse={format_metric(err.inside_domain.rmse)}, mae={format_metric(err.inside_domain.mae)}",
            f"- Outside domain: n={err.outside_domain.n}, r2={format_metric(err.outside_domain.r2)}, "
            f"rmse={format_metric(err.outside_domain.rmse)}, mae={format_metric(err.outside_domain.mae)}",
            f"- Residual mean: {format_metric(err.residual_diagnostics.mean)}",
            f"- Residual std: {format_metric(err.residual_diagnostics.std)}",
            f"- Residual vs predicted correlation: "
            f"{format_metric(err.residual_diagnostics.residual_vs_predicted_correlation)}",
            "",
            "## Deterministic workflow conclusion",
            "",
            f"- Best run: `{conc.best_run_id}`",
            f"- Winner model: `{conc.winner_model}`",
            f"- Selection criterion: {conc.selection_criterion}",
            f"- Acceptance status: {conc.acceptance_status}",
            f"- Failed criteria: {conc.failed_criteria or 'none'}",
            f"- Winner CV r2: {format_metric(conc.winner_cv_r2)}",
        ]
    )
    if conc.winner_train_metrics is not None:
        tm = conc.winner_train_metrics
        lines.append(
            f"- Winner train: r2={format_metric(tm.r2)}, rmse={format_metric(tm.rmse)}, "
            f"mae={format_metric(tm.mae)}, n={tm.n_samples}"
        )
    if conc.winner_val_metrics is not None:
        vm = conc.winner_val_metrics
        lines.append(
            f"- Winner validation: r2={format_metric(vm.r2)}, rmse={format_metric(vm.rmse)}, "
            f"mae={format_metric(vm.mae)}, n={vm.n_samples}"
        )
    lines.append("- Completed searches:")
    for item in conc.completed_searches:
        mark = "completed" if item.completed else "not completed"
        extra = f" ({item.detail})" if item.detail else ""
        lines.append(f"  - `{item.name}`: {mark}{extra}")
    lines.extend(["", "## Agent constraints", "", "### Permitted actions"])
    for item in agent.permitted_actions:
        lines.append(f"- {item}")
    lines.append("### Prohibited actions")
    for item in agent.prohibited_actions:
        lines.append(f"- {item}")
    lines.append("### Iteration budget")
    for key, value in agent.iteration_budget.items():
        lines.append(f"- {key}: {value}")
    lines.append("### Compute budget")
    for key, value in agent.compute_budget.items():
        lines.append(f"- {key}: {value}")
    lines.append("### Approval-required actions")
    if agent.approval_required_actions:
        for item in agent.approval_required_actions:
            lines.append(f"- {item}")
    else:
        lines.append("- none")
    lines.append("### Stopping conditions")
    for item in agent.stopping_conditions:
        lines.append(f"- {item}")
    lines.append("")
    return "\n".join(lines)


def write_experiment_ledger_csv(package: HandoffPackage, path: Path) -> None:
    fieldnames = [
        "run_id",
        "representation",
        "feature_selection_method",
        "model",
        "hyperparameters",
        "feature_count",
        "train_r2",
        "train_rmse",
        "train_mae",
        "cv_r2",
        "cv_rmse",
        "cv_mae",
        "cv_r2_std",
        "train_cv_r2_gap",
        "mean_cv_fold_train_r2",
        "refit_train_cv_gap",
        "cv_fold_train_val_gap",
        "val_r2",
        "val_rmse",
        "val_mae",
        "runtime_seconds",
        "status",
        "observed_vs_predicted",
        "williams",
        "residuals",
        "cv_predictions",
        "test_predictions",
        "config",
        "pipeline",
        "is_winner",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for exp in package.experiments:
            m = exp.metrics
            writer.writerow(
                {
                    "run_id": exp.run_id,
                    "representation": exp.representation,
                    "feature_selection_method": exp.feature_selection_method,
                    "model": exp.model,
                    "hyperparameters": json.dumps(exp.hyperparameters, default=str, sort_keys=True),
                    "feature_count": exp.feature_count,
                    "train_r2": format_metric(m.train_r2),
                    "train_rmse": format_metric(m.train_rmse),
                    "train_mae": format_metric(m.train_mae),
                    "cv_r2": format_metric(m.cv_r2),
                    "cv_rmse": format_metric(m.cv_rmse),
                    "cv_mae": format_metric(m.cv_mae),
                    "cv_r2_std": format_metric(m.cv_r2_std),
                    "train_cv_r2_gap": format_metric(m.train_cv_r2_gap),
                    "mean_cv_fold_train_r2": format_metric(m.mean_cv_fold_train_r2),
                    "refit_train_cv_gap": format_metric(m.refit_train_cv_gap),
                    "cv_fold_train_val_gap": format_metric(m.cv_fold_train_val_gap),
                    "val_r2": format_metric(m.val_r2),
                    "val_rmse": format_metric(m.val_rmse),
                    "val_mae": format_metric(m.val_mae),
                    "runtime_seconds": format_metric(exp.runtime_seconds),
                    "status": exp.status,
                    "observed_vs_predicted": exp.artifacts.observed_vs_predicted.relative_path or "",
                    "williams": exp.artifacts.williams.relative_path or "",
                    "residuals": exp.artifacts.residuals.relative_path or "",
                    "cv_predictions": exp.artifacts.cv_predictions or "",
                    "test_predictions": exp.artifacts.test_predictions or "",
                    "config": exp.artifacts.config or "",
                    "pipeline": exp.artifacts.pipeline or "",
                    "is_winner": exp.is_winner,
                }
            )


def validate_handoff_package(report_dir: Path, package: HandoffPackage) -> None:
    report_dir = Path(report_dir)
    errors: list[str] = []
    ids = [e.run_id for e in package.experiments]
    if len(ids) != len(set(ids)):
        errors.append("Experiment run IDs are not unique.")
    if not ids:
        errors.append("Handoff contains no experiments.")

    referenced = _referenced_relative_paths(package)
    for rel in referenced:
        if not (report_dir / rel).is_file():
            errors.append(f"Referenced file missing: {rel}")

    manifest_path = report_dir / "handoff_manifest.json"
    md_path = report_dir / "modeling_handoff.md"
    csv_path = report_dir / "experiment_ledger.csv"
    for required in (manifest_path, md_path, csv_path):
        if not required.is_file():
            errors.append(f"Required handoff file missing: {required.name}")
    if errors:
        raise HandoffValidationError("; ".join(errors))

    try:
        raw = json.loads(manifest_path.read_text(encoding="utf-8"))
        reloaded = HandoffPackage.model_validate(raw)
    except Exception as exc:
        raise HandoffValidationError(f"handoff_manifest.json is not schema-valid: {exc}") from exc

    _audit_selection_records(package.leakage_safeguards.selection_records)
    _audit_selection_records(reloaded.leakage_safeguards.selection_records)
    if package.leakage_safeguards.test_results_used_for_selection:
        raise HandoffValidationError("External-test results were marked as used for model selection.")

    md_text = md_path.read_text(encoding="utf-8")
    md_metrics = {m.group("run_id"): m.groupdict() for m in CANONICAL_RE.finditer(md_text)}
    csv_rows = list(csv.DictReader(csv_path.open(encoding="utf-8")))
    csv_by_id = {row["run_id"]: row for row in csv_rows}
    if len(csv_rows) != len(package.experiments):
        errors.append("experiment_ledger.csv row count does not match experiments.")

    metric_fields = (
        "train_r2",
        "cv_r2",
        "val_r2",
        "train_cv_r2_gap",
        "cv_fold_train_val_gap",
        "refit_train_cv_gap",
        "mean_cv_fold_train_r2",
        "cv_r2_std",
    )
    orig_by_id = {e.run_id: e for e in package.experiments}
    reload_by_id = {e.run_id: e for e in reloaded.experiments}
    for exp in package.experiments:
        orig = orig_by_id[exp.run_id]
        reloaded_exp = reload_by_id.get(exp.run_id)
        if reloaded_exp is None:
            errors.append(f"JSON missing experiment {exp.run_id}.")
            continue
        csv_row = csv_by_id.get(exp.run_id)
        if csv_row is None:
            errors.append(f"CSV missing experiment {exp.run_id}.")
            continue
        md_row = md_metrics.get(exp.run_id)
        if md_row is None:
            errors.append(f"Markdown missing canonical metrics for {exp.run_id}.")
            continue
        for field in metric_fields:
            orig_val = format_metric(getattr(orig.metrics, field))
            json_val = format_metric(getattr(reloaded_exp.metrics, field))
            csv_val = csv_row.get(field, "")
            md_val = md_row.get(field, "")
            if orig_val != json_val or orig_val != csv_val or orig_val != md_val:
                errors.append(
                    f"Metric mismatch for {exp.run_id}.{field}: "
                    f"package={orig_val} json={json_val} csv={csv_val} md={md_val}"
                )

    if errors:
        raise HandoffValidationError("; ".join(errors))


def _render_experiment_section(exp: ExperimentRecord) -> list[str]:
    m = exp.metrics
    lines = [
        f"### `{exp.run_id}`",
        "",
        _canonical_comment(exp),
        "",
        f"- Representation: {exp.representation}",
        f"- Feature-selection method: `{exp.feature_selection_method}`",
        f"- Model: `{exp.model}`",
        f"- Hyperparameters: `{json.dumps(exp.hyperparameters, default=str, sort_keys=True)}`",
        f"- Feature count: {exp.feature_count}",
        f"- Selected features: {', '.join(f'`{n}`' for n in exp.selected_feature_names) or 'none'}",
        f"- Refit train r2/rmse/mae: {format_metric(m.refit_train_r2 if m.refit_train_r2 is not None else m.train_r2)} / "
        f"{format_metric(m.train_rmse)} / {format_metric(m.train_mae)}",
        f"- Mean CV fold-train r2: {format_metric(m.mean_cv_fold_train_r2)}",
        f"- OOF CV r2/rmse/mae (r2 std): {format_metric(m.oof_cv_r2 if m.oof_cv_r2 is not None else m.cv_r2)} / "
        f"{format_metric(m.cv_rmse)} / {format_metric(m.cv_mae)} ({format_metric(m.cv_r2_std)})",
        f"- Refit-train vs OOF gap (`refit_train_cv_gap`): {format_metric(m.refit_train_cv_gap)}",
        f"- CV fold-train vs fold-val gap (`cv_fold_train_val_gap`, used for acceptance): "
        f"{format_metric(m.cv_fold_train_val_gap if m.cv_fold_train_val_gap is not None else m.train_cv_r2_gap)}",
        f"- Validation r2/rmse/mae: {format_metric(m.val_r2)} / {format_metric(m.val_rmse)} / {format_metric(m.val_mae)}",
        f"- Runtime (s): {format_metric(exp.runtime_seconds)}",
        f"- Status: {exp.status}",
        f"- Winner: {exp.is_winner}",
    ]
    if exp.failure_reason:
        lines.append(f"- Failure reason: {exp.failure_reason}")
    flags = exp.diagnostic_flags
    lines.append(
        f"- Diagnostic flags: status={flags.status}, acceptable={flags.is_acceptable}, "
        f"overfit={flags.is_overfit}, underfit={flags.is_underfit}, "
        f"unstable={flags.is_unstable}, severe_overfit={flags.is_severe_overfit}"
    )
    if exp.warnings:
        lines.append("- Warnings:")
        for warning in exp.warnings:
            lines.append(f"  - {warning}")
    else:
        lines.append("- Warnings: none")
    if exp.errors:
        lines.append("- Errors:")
        for error in exp.errors:
            lines.append(f"  - {error}")
    else:
        lines.append("- Errors: none")
    lines.append("- Per-fold scores:")
    if not exp.per_fold_scores:
        lines.append("  - unavailable")
    for fold in exp.per_fold_scores:
        lines.append(
            f"  - fold {fold.fold}: train_r2={format_metric(fold.train_r2)}, "
            f"val_r2={format_metric(fold.val_r2)}, train_rmse={format_metric(fold.train_rmse)}, "
            f"val_rmse={format_metric(fold.val_rmse)}"
        )
    lines.extend(
        [
            "- Plots:",
            f"  - Observed vs predicted: {_render_plot(exp.artifacts.observed_vs_predicted)}",
            f"  - Williams: {_render_plot(exp.artifacts.williams)}",
            f"  - Residuals: {_render_plot(exp.artifacts.residuals)}",
            f"- CV predictions: `{exp.artifacts.cv_predictions or 'unavailable'}`",
            f"- Test predictions: `{exp.artifacts.test_predictions or 'unavailable'}`",
            f"- Config: `{exp.artifacts.config or 'unavailable'}`",
            f"- Pipeline: `{exp.artifacts.pipeline or 'unavailable'}`",
        ]
    )
    if exp.applicability_domain is not None:
        ad = exp.applicability_domain
        lines.extend(
            [
                f"- AD method: `{ad.method}`",
                f"- Warning leverage: {format_metric(ad.warning_leverage)}",
                f"- Residual threshold: {format_metric(ad.residual_threshold)}",
                f"- Structural outlier IDs: {ad.structural_outlier_ids or 'none'}",
                f"- Response outlier IDs: {ad.response_outlier_ids or 'none'}",
                f"- AD handling: `{ad.handling_decision}` — {ad.handling_justification}",
            ]
        )
    lines.append("")
    return lines


def _canonical_comment(exp: ExperimentRecord) -> str:
    m = exp.metrics
    return (
        f"<!-- canonical_metrics run_id={exp.run_id}"
        f" train_r2={format_metric(m.train_r2)}"
        f" cv_r2={format_metric(m.cv_r2)}"
        f" val_r2={format_metric(m.val_r2)}"
        f" train_cv_r2_gap={format_metric(m.train_cv_r2_gap)}"
        f" cv_fold_train_val_gap={format_metric(m.cv_fold_train_val_gap)}"
        f" refit_train_cv_gap={format_metric(m.refit_train_cv_gap)}"
        f" mean_cv_fold_train_r2={format_metric(m.mean_cv_fold_train_r2)}"
        f" cv_r2_std={format_metric(m.cv_r2_std)} -->"
    )


def _render_plot(ref: PlotReference) -> str:
    if ref.status == "available" and ref.relative_path:
        return f"![{ref.name}]({ref.relative_path})"
    return f"unavailable ({ref.reason or 'not generated'})"


def _build_experiment(
    *,
    branch: ModelBranchResult,
    artifacts: BranchExternalArtifacts | None,
    workflow_run_id: str,
    used_ids: set[str],
    representation: str,
    report_dir: Path,
    train_path: Path,
    config: WorkflowConfig,
    is_winner: bool,
) -> ExperimentRecord:
    fs_tag = _feature_selection_tag(branch)
    run_id = _make_experiment_id(workflow_run_id, branch.estimator, fs_tag, used_ids)
    features = list(branch.ga.selected_features) if branch.ga is not None else []
    model_name = normalize_estimator_name(branch.estimator) or branch.estimator
    params = {}
    if branch.hpo_result.final_selection is not None:
        params = dict(branch.hpo_result.final_selection.params)
    elif branch.model_config_snapshot:
        params = dict(branch.model_config_snapshot.get("params") or {})

    status: str = "completed"
    failure_reason = ""
    errors: list[str] = []
    warnings: list[str] = []
    if branch.hpo_result.final_selection is None:
        status = "failed"
        failure_reason = "Branch has no final model selection."
        errors.append(failure_reason)
    if artifacts is None and status == "completed":
        status = "failed"
        failure_reason = "External-evaluation artifacts were not produced."
        errors.append(failure_reason)

    flags = DiagnosticFlags()
    metrics = ExperimentMetrics()
    external = ExternalTestMetrics()
    folds: list[FoldMetrics] = []
    fs = branch.hpo_result.final_selection
    if fs is not None:
        cv = fs.cv_summary
        metrics.cv_r2 = cv.mean_cv_r2
        metrics.cv_rmse = cv.mean_cv_rmse
        metrics.cv_mae = cv.mean_cv_mae
        metrics.cv_r2_std = cv.std_cv_r2
        metrics.train_cv_r2_gap = cv.train_cv_r2_gap
        metrics.mean_cv_fold_train_r2 = cv.mean_train_r2
        metrics.cv_fold_train_val_gap = cv.train_cv_r2_gap
        metrics.oof_cv_r2 = cv.mean_cv_r2
        metrics.val_r2 = cv.holdout_val_r2
        flags = DiagnosticFlags(
            status=fs.assessment.status,
            is_acceptable=fs.assessment.is_acceptable,
            is_overfit=fs.assessment.is_overfit,
            is_underfit=fs.assessment.is_underfit,
            is_unstable=fs.assessment.is_unstable,
            is_severe_overfit=fs.assessment.is_severe_overfit,
        )
        warnings.extend(fs.assessment.warnings)
        if fs.warning:
            warnings.append(fs.warning)
        if branch.hpo_result.baseline_cv is not None:
            folds = list(branch.hpo_result.baseline_cv.fold_metrics)

    if artifacts is not None:
        metrics.train_r2 = artifacts.train_r2
        if artifacts.val_r2 is not None:
            metrics.val_r2 = artifacts.val_r2
        external = ExternalTestMetrics(
            reported_after_selection=True,
            r2=artifacts.test_r2,
        )
        if artifacts.metrics_path and Path(artifacts.metrics_path).exists():
            blob = json.loads(Path(artifacts.metrics_path).read_text(encoding="utf-8"))
            if "train" in blob:
                metrics.train_r2 = blob["train"]["r2"]
                metrics.train_rmse = blob["train"]["rmse"]
                metrics.train_mae = blob["train"]["mae"]
                metrics.train_n = blob["train"]["n_samples"]
            if "val" in blob:
                metrics.val_r2 = blob["val"]["r2"]
                metrics.val_rmse = blob["val"]["rmse"]
                metrics.val_mae = blob["val"]["mae"]
                metrics.val_n = blob["val"]["n_samples"]
            if "test" in blob:
                external.r2 = blob["test"]["r2"]
                external.rmse = blob["test"]["rmse"]
                external.mae = blob["test"]["mae"]
                external.n = blob["test"]["n_samples"]

    runtime = None
    parts = [v for v in (branch.runtime_seconds, None if artifacts is None else artifacts.runtime_seconds) if v is not None]
    if parts:
        runtime = float(sum(parts))

    model_config = _model_config_for_branch(branch)
    ovp = _copy_named_plot(
        None if artifacts is None else artifacts.scatter_png_path,
        report_dir / "plots" / f"{run_id}_observed_vs_predicted.png",
        "observed_vs_predicted",
        missing_reason="Observed-versus-predicted plot was not generated for this run.",
    )
    williams = _copy_named_plot(
        None if artifacts is None else artifacts.williams_png_path,
        report_dir / "plots" / f"{run_id}_williams.png",
        "williams",
        missing_reason="Williams plot was not generated for this run.",
    )
    residual_src = "" if artifacts is None else artifacts.residual_png_path
    pred_path = "" if artifacts is None else artifacts.predictions_path
    residuals = _residual_plot_reference(run_id, report_dir, residual_src, pred_path)

    cv_rel = None
    test_rel = None
    if features and train_path.exists() and model_config is not None and status == "completed":
        cv_dest = report_dir / "predictions" / f"{run_id}_cv_predictions.csv"
        fold_scores, cv_ok = _write_cv_predictions(
            train_path,
            features,
            model_config,
            int(config.hpo.cv_folds or config.sfs.cv_folds),
            config.random_seed,
            cv_dest,
        )
        if cv_ok:
            cv_rel = f"predictions/{cv_dest.name}"
            if fold_scores:
                folds = fold_scores
                fill_cv_error_metrics_from_folds(metrics, folds)
        else:
            warnings.append("CV predictions could not be written for this run.")
    elif status == "completed":
        warnings.append("CV predictions skipped: missing training data or features.")

    if _is_placeholder_cv_error(metrics.cv_rmse, metrics.cv_r2) or _is_placeholder_cv_error(
        metrics.cv_mae, metrics.cv_r2
    ):
        if folds and fs is not None and fs.source == "baseline":
            fill_cv_error_metrics_from_folds(metrics, folds)
        else:
            drop_placeholder_cv_errors(metrics)

    if pred_path and Path(pred_path).exists():
        test_dest = report_dir / "predictions" / f"{run_id}_test_predictions.csv"
        pred_df = pd.read_csv(pred_path)
        test_df = pred_df[pred_df["split"] == "test"] if "split" in pred_df.columns else pred_df
        test_df.to_csv(test_dest, index=False)
        test_rel = f"predictions/{test_dest.name}"

    cfg_rel = f"configs/{run_id}_config.json"
    save_json(
        report_dir / cfg_rel,
        {
            "run_id": run_id,
            "estimator": model_name,
            "feature_selection_method": fs_tag,
            "hyperparameters": params,
            "selected_features": features,
            "cv_folds": int(config.hpo.cv_folds or config.sfs.cv_folds),
            "random_seed": config.random_seed,
        },
    )

    pipeline_rel = None
    model_src = "" if artifacts is None else artifacts.model_path
    if model_src and Path(model_src).exists() and features:
        pipe_dest = report_dir / "models" / f"{run_id}_pipeline.joblib"
        try:
            estimator = joblib.load(model_src)
            pipeline = Pipeline(
                [
                    ("select_features", ColumnSelector(features)),
                    ("estimator", estimator),
                ]
            )
            joblib.dump(pipeline, pipe_dest)
            pipeline_rel = f"models/{pipe_dest.name}"
        except Exception as exc:
            errors.append(f"Fitted pipeline could not be saved: {exc}")

    finalize_train_cv_gap_metrics(metrics, folds)
    ad = _experiment_ad(artifacts)
    return ExperimentRecord(
        run_id=run_id,
        representation=representation,
        feature_selection_method=fs_tag,
        model=model_name,
        hyperparameters=params,
        feature_count=len(features),
        selected_feature_names=features,
        metrics=metrics,
        external_test=external,
        per_fold_scores=folds,
        diagnostic_flags=flags,
        warnings=warnings,
        errors=errors,
        failure_reason=failure_reason,
        runtime_seconds=runtime,
        status=status,  # type: ignore[arg-type]
        artifacts=ArtifactIndex(
            observed_vs_predicted=ovp,
            williams=williams,
            residuals=residuals,
            cv_predictions=cv_rel,
            test_predictions=test_rel,
            config=cfg_rel,
            pipeline=pipeline_rel,
            source_predictions=pred_path,
            source_model=model_src,
        ),
        applicability_domain=ad,
        is_winner=is_winner,
    )


def _write_cv_predictions(
    train_path: Path,
    selected_features: list[str],
    model_config: ModelConfig,
    cv_folds: int,
    random_seed: int,
    dest: Path,
) -> tuple[list[FoldMetrics], bool]:
    try:
        df = pd.read_csv(train_path)
        missing = [f for f in selected_features if f not in df.columns]
        if missing or "activity" not in df.columns or "compound_id" not in df.columns:
            return [], False
        n = len(df)
        folds = min(int(cv_folds), n)
        if folds < 2:
            return [], False
        X = df[selected_features]
        y = df["activity"]
        ids = df["compound_id"]
        kf = KFold(n_splits=folds, shuffle=True, random_state=random_seed)
        fold_metrics: list[FoldMetrics] = []
        rows: list[dict[str, Any]] = []
        for fold_idx, (tr_idx, va_idx) in enumerate(kf.split(X), start=1):
            model = build_estimator(model_config)
            X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
            y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]
            model.fit(X_tr, y_tr)
            y_tr_pred = model.predict(X_tr)
            y_va_pred = model.predict(X_va)
            fold_metrics.append(
                FoldMetrics(
                    fold=fold_idx,
                    train_r2=float(r2_score(y_tr, y_tr_pred)),
                    val_r2=float(r2_score(y_va, y_va_pred)),
                    train_rmse=float((mean_squared_error(y_tr, y_tr_pred) ** 0.5)),
                    val_rmse=float((mean_squared_error(y_va, y_va_pred) ** 0.5)),
                    train_mae=float(mean_absolute_error(y_tr, y_tr_pred)),
                    val_mae=float(mean_absolute_error(y_va, y_va_pred)),
                )
            )
            for i, idx in enumerate(va_idx):
                activity = float(y.iloc[idx])
                predicted = float(y_va_pred[i])
                rows.append(
                    {
                        "compound_id": ids.iloc[idx],
                        "fold": fold_idx,
                        "activity": activity,
                        "predicted_activity": predicted,
                        "residual": activity - predicted,
                    }
                )
        dest.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(dest, index=False)
        return fold_metrics, True
    except Exception:
        return [], False


def _residual_plot_reference(
    run_id: str,
    report_dir: Path,
    source_png: str,
    predictions_path: str,
) -> PlotReference:
    dest = report_dir / "plots" / f"{run_id}_residuals.png"
    copied = _copy_named_plot(
        source_png,
        dest,
        "residuals",
        missing_reason="Residual plot was not generated for this run.",
    )
    if copied.status == "available":
        return copied
    if predictions_path and Path(predictions_path).exists():
        try:
            pred_df = pd.read_csv(predictions_path)
            svg_path = dest.with_suffix(".svg")
            plot_residuals(
                pred_df["predicted_activity"].values,
                pred_df["residual"].values,
                pred_df["split"].values,
                dest,
                svg_path,
            )
            return PlotReference(
                name="residuals",
                status="available",
                relative_path=f"plots/{dest.name}",
            )
        except Exception as exc:
            return PlotReference(
                name="residuals",
                status="unavailable",
                reason=f"Residual plot could not be generated: {exc}",
            )
    return copied


def _copy_named_plot(
    source: str | None,
    dest: Path,
    name: str,
    *,
    missing_reason: str,
) -> PlotReference:
    if source and Path(source).is_file():
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, dest)
        return PlotReference(name=name, status="available", relative_path=f"plots/{dest.name}")
    return PlotReference(name=name, status="unavailable", reason=missing_reason)


def _copy_preprocessor(source: str, models_dir: Path) -> str | None:
    if source and Path(source).is_file():
        dest = models_dir / "descriptor_preprocessor.joblib"
        shutil.copy2(source, dest)
        return f"models/{dest.name}"
    return None


def _experiment_ad(artifacts: BranchExternalArtifacts | None) -> ExperimentAD | None:
    if artifacts is None:
        return None
    warning_leverage = None
    residual_threshold = 3.0
    structural: list[str] = []
    response: list[str] = []
    by_part: dict[str, dict[str, list[str]]] = {}
    if artifacts.ad_report_path and Path(artifacts.ad_report_path).exists():
        report = json.loads(Path(artifacts.ad_report_path).read_text(encoding="utf-8"))
        warning_leverage = report.get("warning_leverage")
        residual_threshold = float(report.get("residual_threshold", 3.0))
        structural = [str(x) for x in report.get("high_leverage_ids", [])]
        response = [str(x) for x in report.get("response_outlier_ids", [])]
    if artifacts.ad_classifications_path and Path(artifacts.ad_classifications_path).exists():
        df = pd.read_csv(artifacts.ad_classifications_path)
        for split_name in ("train", "val", "test"):
            sub = df[df["split"] == split_name] if "split" in df.columns else df.iloc[0:0]
            struct_ids = (
                sub[sub["applicability_domain"].isin(["high_leverage", "outside_both"])]["compound_id"]
                .astype(str)
                .tolist()
                if "applicability_domain" in sub.columns
                else []
            )
            resp_ids = (
                sub[sub["applicability_domain"].isin(["response_outlier", "outside_both"])]["compound_id"]
                .astype(str)
                .tolist()
                if "applicability_domain" in sub.columns
                else []
            )
            by_part[split_name] = {"structural": struct_ids, "response": resp_ids}
    return ExperimentAD(
        method="williams_leverage",
        warning_leverage=None if warning_leverage is None else float(warning_leverage),
        residual_threshold=residual_threshold,
        structural_outlier_ids=structural,
        response_outlier_ids=response,
        structural_outlier_count=len(structural),
        response_outlier_count=len(response),
        outliers_by_partition=by_part,
        handling_decision=AD_HANDLING_DECISION,
        handling_justification=AD_HANDLING_JUSTIFICATION,
    )


def _dataset_audit(
    validation: DatasetValidationResult,
    descriptors: DescriptorCalculationResult,
    split: SplitResult,
    preprocessing: PreprocessingResult,
    dataset_hash: str,
    overlap: DuplicateOverlap,
    selected_feature_count: int = 0,
    ga_feature_count: int = 0,
) -> DatasetAudit:
    n0 = preprocessing.initial_descriptor_count
    after_nonnumeric = n0 - preprocessing.removed_nonnumeric
    after_missing = after_nonnumeric - preprocessing.removed_missing
    after_constant = after_missing - preprocessing.removed_constant
    after_near = after_constant - preprocessing.removed_near_constant
    steps = [
        CurationStep(step="input", n_compounds=validation.original_row_count),
        CurationStep(
            step="after_structure_and_activity_validation",
            n_compounds=validation.valid_compound_count,
            n_removed=validation.invalid_smiles_count + validation.missing_or_invalid_activity_count,
        ),
        CurationStep(
            step="after_descriptor_calculation",
            n_compounds=descriptors.compound_count,
            n_features=descriptors.descriptor_count,
        ),
        CurationStep(
            step="after_nonnumeric_filter",
            n_features=after_nonnumeric,
            n_removed=preprocessing.removed_nonnumeric,
        ),
        CurationStep(
            step="after_missing_filter",
            n_features=after_missing,
            n_removed=preprocessing.removed_missing,
        ),
        CurationStep(
            step="after_constant_filter",
            n_features=after_constant,
            n_removed=preprocessing.removed_constant,
        ),
        CurationStep(
            step="after_near_constant_filter",
            n_features=after_near,
            n_removed=preprocessing.removed_near_constant,
        ),
        CurationStep(
            step="after_correlation_filter",
            n_features=preprocessing.final_descriptor_count,
            n_removed=preprocessing.removed_correlated,
        ),
        CurationStep(
            step="after_split",
            n_compounds=split.train_count + split.val_count + split.test_count,
            notes=f"train={split.train_count}, val={split.val_count}, test={split.test_count}",
        ),
    ]
    stats = validation.activity_stats
    return DatasetAudit(
        curation_steps=steps,
        invalid_structures=validation.invalid_smiles_count,
        duplicates=validation.duplicate_compound_count,
        missing_or_invalid_activity=validation.missing_or_invalid_activity_count,
        descriptors_with_missing=descriptors.descriptors_with_missing,
        target_statistics={
            "min": stats.min,
            "max": stats.max,
            "mean": stats.mean,
            "median": stats.median,
            "std": stats.std,
        },
        feature_counts={
            "raw_descriptors": descriptors.descriptor_count,
            "generated_descriptors": descriptors.generated_descriptor_count,
            "external_descriptors": descriptors.external_descriptor_count,
            "after_preprocessing": preprocessing.final_descriptor_count,
            "one_se_selected_feature_count": selected_feature_count,
            "ga_selected_feature_count": ga_feature_count,
        },
        train_size=split.train_count,
        validation_size=split.val_count,
        test_size=split.test_count,
        split_strategy=split.split_method,
        dataset_hash=dataset_hash,
        duplicate_overlap=overlap,
    )


def _duplicate_overlap(split: SplitResult) -> DuplicateOverlap:
    def smiles(path: str) -> set[str]:
        if not path or not Path(path).exists():
            return set()
        df = pd.read_csv(path)
        col = "canonical_smiles" if "canonical_smiles" in df.columns else None
        if col is None:
            return set()
        return set(df[col].astype(str))

    train = smiles(split.train_path)
    val = smiles(split.val_path)
    test = smiles(split.test_path)
    tv = sorted(train & val)
    tt = sorted(train & test)
    vt = sorted(val & test)
    return DuplicateOverlap(
        train_val=tv,
        train_test=tt,
        val_test=vt,
        any_overlap=bool(tv or tt or vt),
    )


def _hash_compound_ids(path: str) -> str:
    if not path or not Path(path).exists():
        return ""
    df = pd.read_csv(path)
    if "compound_id" not in df.columns:
        return ""
    payload = "\n".join(sorted(df["compound_id"].astype(str)))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _iter_branch_variants(*roots: ModelBranchResult | None) -> Iterable[ModelBranchResult]:
    seen: set[str] = set()
    for root in roots:
        if root is None:
            continue
        for branch in (root, root.sfs_subset, root.sfs_subset_hpo, root.expansion):
            if branch is None:
                continue
            key = str(Path(branch.branch_dir).resolve()) if branch.branch_dir else str(id(branch))
            if key in seen:
                continue
            seen.add(key)
            yield branch


def _match_winner(
    branches: list[ModelBranchResult],
    *,
    winning_estimator: str,
    winning_features: list[str],
    winner_is_expansion: bool,
    winner_expansion_label: str,
) -> ModelBranchResult | None:
    feat = list(winning_features)
    for branch in branches:
        label = branch_display_label(branch)
        if bool(branch.is_expansion) != bool(winner_is_expansion):
            continue
        if winner_is_expansion and branch.expansion_label != winner_expansion_label:
            continue
        if label != winning_estimator and branch.estimator != winning_estimator:
            base = winning_estimator.split("(", 1)[0].strip()
            if branch.estimator != base and label != base:
                continue
        if list(branch.ga.selected_features) != feat:
            continue
        return branch
    for branch in branches:
        if list(branch.ga.selected_features) == feat:
            if bool(branch.is_expansion) == bool(winner_is_expansion):
                return branch
    return branches[0] if branches else None


def _feature_selection_tag(branch: ModelBranchResult) -> str:
    if branch.is_expansion and branch.expansion_label:
        return _safe_token(branch.expansion_label)
    return "ga"


def _make_experiment_id(run_id: str, estimator: str, fs_tag: str, used: set[str]) -> str:
    slug = estimator_slug(normalize_estimator_name(estimator) or estimator)
    base = f"{run_id}__{slug}__{_safe_token(fs_tag)}"
    candidate = base
    index = 2
    while candidate in used:
        candidate = f"{base}_{index}"
        index += 1
    used.add(candidate)
    return candidate


def _safe_token(text: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_\-]+", "_", str(text)).strip("_").lower()
    return cleaned or "unknown"


def _looks_like_fingerprint(name: str) -> bool:
    lowered = name.lower()
    return any(token in lowered for token in ("fingerprint", "morgan", "maccs", "ecfp", "fcfp"))


def _model_config_for_branch(branch: ModelBranchResult) -> ModelConfig | None:
    if branch.model_config_snapshot:
        try:
            return ModelConfig(**branch.model_config_snapshot)
        except Exception:
            pass
    try:
        return ModelConfig(estimator=normalize_estimator_name(branch.estimator) or branch.estimator)
    except Exception:
        return None


def _collect_selection_records(
    cross: CrossModelSelection | None,
    rf_branch: ModelBranchResult,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    if cross is not None:
        records.extend(dict(row) for row in cross.compared_models)
        if cross.final_selection is not None:
            records.extend(dict(row) for row in cross.final_selection.compared_candidates)
    fs = rf_branch.hpo_result.final_selection
    if fs is not None:
        records.extend(dict(row) for row in fs.compared_candidates)
    return records


def _audit_selection_records(records: list[dict[str, Any]]) -> None:
    for row in records:
        for key in row:
            lowered = str(key).lower()
            if lowered in TEST_METRIC_KEYS:
                raise HandoffValidationError(
                    f"External-test metric '{key}' appeared in model-selection records."
                )


def _failed_criteria(assessment: Any) -> list[str]:
    if assessment is None:
        return ["no_overfitting_assessment"]
    if assessment.is_acceptable:
        return []
    failed: list[str] = []
    if assessment.is_unstable:
        failed.append("cv_variability")
    if assessment.is_underfit:
        failed.append("underfitting")
    if assessment.is_overfit:
        failed.append("train_cv_gap")
    if assessment.is_severe_overfit:
        failed.append("severe_train_cv_gap")
    if assessment.status == "poor_performance":
        failed.append("minimum_cv_r2")
    if not failed:
        failed.append(str(assessment.status))
    return failed


def _agent_constraints(config: WorkflowConfig) -> AgentConstraints:
    return AgentConstraints(
        permitted_actions=[
            "Explain the deterministic one-SE feature-count selection without changing it",
            "Propose hyperparameter grids within the allowed estimator parameter space",
        ],
        prohibited_actions=[
            "Override the selected feature count",
            "Invent metrics or training results",
            "Train models or execute the scientific pipeline",
            "Use the external test set for tuning, feature selection, or model selection",
        ],
        iteration_budget={
            "max_hpo_rounds": int(config.hpo.max_hpo_rounds),
        },
        compute_budget={
            "max_candidates_per_round": int(config.hpo.max_candidates_per_round),
            "hpo_n_jobs": int(config.hpo.n_jobs),
            "sfs_n_jobs": int(config.sfs.n_jobs),
            "ga_n_jobs": int(config.ga.n_jobs),
        },
        approval_required_actions=[],
        stopping_conditions=[
            "Model meets acceptance criteria (overfitting status good)",
            "CV improvement below min_cv_improvement",
            "Maximum HPO rounds reached",
        ],
    )


def _referenced_relative_paths(package: HandoffPackage) -> list[str]:
    paths: list[str] = []
    if package.representation_preprocessing.preprocessor_relative_path:
        paths.append(package.representation_preprocessing.preprocessor_relative_path)
    for exp in package.experiments:
        for plot in (
            exp.artifacts.observed_vs_predicted,
            exp.artifacts.williams,
            exp.artifacts.residuals,
        ):
            if plot.status == "available" and plot.relative_path:
                paths.append(plot.relative_path)
        for rel in (
            exp.artifacts.cv_predictions,
            exp.artifacts.test_predictions,
            exp.artifacts.config,
            exp.artifacts.pipeline,
        ):
            if rel:
                paths.append(rel)
    return paths


def _empty_error_analysis(winner_run_id: str):
    from qsar_agent.schemas.handoff import DomainPerformance, ErrorAnalysis, ResidualDiagnostics

    return ErrorAnalysis(
        winner_run_id=winner_run_id,
        inside_domain=DomainPerformance(n=0),
        outside_domain=DomainPerformance(n=0),
        residual_diagnostics=ResidualDiagnostics(),
    )
