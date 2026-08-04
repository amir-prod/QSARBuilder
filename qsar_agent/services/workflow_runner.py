"""QSAR workflow runner — orchestrates deterministic pipeline stages."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from qsar_agent.agents.qsar_agent import (
    build_final_report,
    propose_hyperparameter_grid,
    run_agent_feature_count_selection,
)
from qsar_agent.config import ModelConfig, WorkflowConfig
from qsar_agent.logging_utils import append_log, append_warning, get_logger
from qsar_agent.schemas.hyperparameter_optimization import HPOConfig, OverfittingThresholds
from qsar_agent.schemas.workflow import StageStatus, WorkflowState
from qsar_agent.services.artifact_manager import (
    copy_input_dataset,
    create_zip_archive,
    file_hash,
    generate_run_id,
    get_run_dir,
    save_json,
)
from qsar_agent.services.plotting import plot_sfs_r2
from qsar_agent.tools.branch_external_evaluation import (
    evaluate_branch_on_external_test,
    find_winning_branch,
    flatten_branches,
    load_modeling_and_ad_from_artifacts,
    modeling_result_with_run_dir_paths,
    promote_branch_artifacts_to_run_dir,
)
from qsar_agent.tools.dataset_validation import validate_dataset
from qsar_agent.tools.descriptor_preprocessing import fit_descriptor_preprocessor
from qsar_agent.tools.genetic_algorithm import run_genetic_algorithm
from qsar_agent.tools.hyperparameter_optimization import run_iterative_hyperparameter_optimization
from qsar_agent.tools.descriptor_calculation import calculate_descriptors
from qsar_agent.tools.sequential_feature_selection import run_sequential_feature_selection
from qsar_agent.tools.umap_split import create_umap_cluster_split
from qsar_agent.schemas.model_fallback import BranchExternalArtifacts, ModelBranchResult
from qsar_agent.tools.model_fallback import run_model_fallback_if_needed

logger = get_logger()


def _hpo_config_from_workflow(config: WorkflowConfig) -> HPOConfig:
    h = config.hpo
    return HPOConfig(
        enabled=h.enabled,
        max_hpo_rounds=min(h.max_hpo_rounds, 3),
        cv_folds=h.cv_folds or config.sfs.cv_folds,
        max_candidates_per_round=h.max_candidates_per_round,
        min_cv_improvement=h.min_cv_improvement,
        random_seed=config.random_seed,
        n_jobs=h.n_jobs,
        openai_model=h.openai_model or "",
        thresholds=OverfittingThresholds(
            overfit_gap_threshold=h.overfit_gap_threshold,
            severe_overfit_gap_threshold=h.severe_overfit_gap_threshold,
            minimum_cv_r2=h.minimum_cv_r2,
            cv_std_threshold=h.cv_std_threshold,
            minimum_train_r2=h.minimum_train_r2,
        ),
    )


class WorkflowRunner:
    """Runs the full QSAR workflow with stage tracking."""

    def __init__(
        self,
        config: WorkflowConfig,
        dataset_path: str | Path,
        progress_callback: Callable[[WorkflowState], None] | None = None,
        run_id: str | None = None,
    ):
        self.config = config
        self.dataset_path = Path(dataset_path)
        self.run_id = run_id or generate_run_id()
        self.run_dir = get_run_dir(config.output_dir, self.run_id)
        self.state = WorkflowState.create(
            self.run_id, config_snapshot=config.to_dict()
        )
        self.progress_callback = progress_callback
        self.warnings: list[str] = []
        self.artifact_paths: dict[str, str] = {}

    def _notify(self) -> None:
        if self.progress_callback:
            self.progress_callback(self.state)

    def _start_stage(self, stage: str) -> None:
        self.state.set_stage_status(stage, StageStatus.RUNNING)
        append_log(self.state.logs, f"Starting stage: {stage}")
        self._notify()

    def _complete_stage(self, stage: str, message: str = "") -> None:
        self.state.set_stage_status(stage, StageStatus.COMPLETED, message)
        append_log(self.state.logs, f"Completed stage: {stage}")
        self._notify()

    def _fail_stage(self, stage: str, error: str) -> None:
        self.state.set_stage_status(stage, StageStatus.FAILED, error)
        append_log(self.state.logs, f"Failed stage {stage}: {error}", "ERROR")
        self._notify()

    def _skip_stage(self, stage: str, message: str = "Skipped") -> None:
        self.state.set_stage_status(stage, StageStatus.CANCELLED, message)
        self._notify()

    def run(self) -> WorkflowState:
        copy_input_dataset(self.dataset_path, self.run_dir)
        dataset_hash = file_hash(self.run_dir / "input_dataset.csv")

        try:
            # 1. Dataset validation
            self._start_stage("dataset_validation")
            validation = validate_dataset(
                self.dataset_path,
                self.config.smiles_column,
                self.config.activity_column,
                self.config.id_column,
                self.run_dir,
                self.config.min_valid_compounds,
            )
            self.warnings.extend(validation.warnings)
            self.artifact_paths.update(
                {
                    "cleaned_dataset": validation.cleaned_dataset_path,
                    "dataset_validation": validation.validation_report_path,
                }
            )
            self._complete_stage("dataset_validation")

            # 2. Descriptor calculation (DescJocky + optional external merge)
            self._start_stage("descriptor_calculation")
            descriptors = calculate_descriptors(
                validation.cleaned_dataset_path,
                self.run_dir,
                self.config.descriptors,
            )
            self.warnings.extend(descriptors.warnings)
            self.artifact_paths["descriptors_raw"] = descriptors.raw_descriptors_path
            if descriptors.generated_descriptors_path:
                self.artifact_paths["generated_descriptors"] = (
                    descriptors.generated_descriptors_path
                )
            if descriptors.calculation_report_path:
                self.artifact_paths["descriptor_calculation_report"] = (
                    descriptors.calculation_report_path
                )
            if descriptors.calculation_report_md_path:
                self.artifact_paths["descriptor_calculation_report_md"] = (
                    descriptors.calculation_report_md_path
                )
            if descriptors.external_descriptors_path:
                self.artifact_paths["external_descriptors"] = (
                    descriptors.external_descriptors_path
                )
            self._complete_stage(
                "descriptor_calculation",
                f"backends={','.join(descriptors.backends)}; "
                f"3D_descriptors={descriptors.three_d_descriptors_included}",
            )

            # 3. UMAP split
            self._start_stage("umap_split")
            split = create_umap_cluster_split(
                descriptors.raw_descriptors_path,
                self.run_dir,
                self.config.test_fraction,
                self.config.random_seed,
                self.config.umap,
                self.config.clustering,
            )
            self.warnings.extend(split.warnings)
            self.artifact_paths.update(
                {
                    "train_raw": split.train_path,
                    "test_raw": split.test_path,
                    "umap_plot": split.umap_plot_png,
                }
            )
            self._complete_stage("umap_split")

            # 4. Descriptor preprocessing
            self._start_stage("descriptor_preprocessing")
            preprocessing = fit_descriptor_preprocessor(
                split.train_path,
                split.test_path,
                self.run_dir,
                self.config.preprocessing,
            )
            self.warnings.extend(preprocessing.warnings)
            self.artifact_paths.update(
                {
                    "preprocessed_train": preprocessing.preprocessed_train_path,
                    "preprocessed_test": preprocessing.preprocessed_test_path,
                    "preprocessor": preprocessing.preprocessor_path,
                }
            )
            self._complete_stage("descriptor_preprocessing")

            # 5. Sequential feature selection
            self._start_stage("sequential_feature_selection")
            append_log(
                self.state.logs,
                "Sequential feature selection may take several minutes with many descriptors.",
            )
            sfs = run_sequential_feature_selection(
                preprocessing.preprocessed_train_path,
                self.run_dir,
                self.config.sfs.max_features,
                self.config.sfs.cv_folds,
                self.config.model,
                self.config.sfs.random_seed,
                self.config.sfs.n_jobs,
            )
            self.artifact_paths["sfs_results"] = sfs.results_csv_path
            self._complete_stage("sequential_feature_selection")

            # 6. Feature count selection (agent-assisted explanation)
            self._start_stage("feature_count_selection")
            feature_count = run_agent_feature_count_selection(sfs, self.run_dir)
            plot_sfs_r2(
                __import__("pandas").read_csv(sfs.results_csv_path),
                Path(sfs.plot_png_path),
                Path(sfs.plot_svg_path),
                feature_count.selected_feature_count,
            )
            self.artifact_paths["feature_count_selection"] = feature_count.selection_json_path
            self._complete_stage("feature_count_selection")

            # 7. Genetic algorithm
            self._start_stage("genetic_algorithm")
            ga = run_genetic_algorithm(
                preprocessing.preprocessed_train_path,
                self.run_dir,
                feature_count.selected_feature_count,
                self.config.ga,
                self.config.model,
            )
            self.artifact_paths["ga_selected_features"] = ga.selected_features_path
            self._complete_stage("genetic_algorithm")

            hpo_cfg = _hpo_config_from_workflow(self.config)
            train_df = __import__("pandas").read_csv(preprocessing.preprocessed_train_path)
            n_train = len(train_df)
            n_features = len(ga.selected_features)

            def hpo_log(msg: str) -> None:
                append_log(self.state.logs, msg)

            if not hpo_cfg.enabled:
                self._skip_stage("baseline_cv_diagnostics", "HPO disabled")
                self._skip_stage("overfitting_assessment", "HPO disabled")
                for r in (1, 2, 3):
                    self._skip_stage(f"hpo_round_{r}", "HPO disabled")
                self._skip_stage("final_model_selection", "HPO disabled")
                hpo_result = run_iterative_hyperparameter_optimization(
                    preprocessing.preprocessed_train_path,
                    ga.selected_features,
                    self.config.model,
                    hpo_cfg,
                    self.run_dir,
                    log_callback=hpo_log,
                )
                final_model_config = ModelConfig(**hpo_result.final_model_config)
                hpo_metadata = {
                    "enabled": False,
                    "max_rounds": hpo_cfg.max_hpo_rounds,
                    "rounds_completed": 0,
                    "final_model_source": "baseline",
                    "final_params": self.config.model.model_dump(),
                }
            else:
                self._start_stage("baseline_cv_diagnostics")

                def grid_proposer(**kwargs):
                    return propose_hyperparameter_grid(
                        openai_model=hpo_cfg.openai_model or None,
                        **kwargs,
                    )

                hpo_result = run_iterative_hyperparameter_optimization(
                    preprocessing.preprocessed_train_path,
                    ga.selected_features,
                    self.config.model,
                    hpo_cfg,
                    self.run_dir,
                    grid_proposer=grid_proposer,
                    log_callback=hpo_log,
                    n_features=n_features,
                    n_train_samples=n_train,
                )
                self._complete_stage("baseline_cv_diagnostics")
                self._start_stage("overfitting_assessment")
                append_log(
                    self.state.logs,
                    f"Overfitting assessment: {hpo_result.baseline_assessment.status}. "
                    f"Train-CV R² gap = {hpo_result.baseline_assessment.train_cv_r2_gap:.3f}.",
                )
                self._complete_stage("overfitting_assessment")

                completed_rounds = hpo_result.rounds_completed
                for r in (1, 2, 3):
                    stage = f"hpo_round_{r}"
                    if r <= completed_rounds:
                        self.state.set_stage_status(
                            stage,
                            StageStatus.COMPLETED,
                            f"Best CV R²={hpo_result.rounds[r - 1].best_cv_summary.mean_cv_r2:.3f}",
                        )
                    elif hpo_result.baseline_assessment and hpo_result.baseline_assessment.is_acceptable:
                        self._skip_stage(stage, "Baseline acceptable")
                    elif not hpo_result.hpo_triggered:
                        self._skip_stage(stage, "HPO not required")
                    else:
                        self._skip_stage(stage, "Stopped early or max rounds not reached")
                self._notify()

                self._start_stage("final_model_selection")
                append_log(
                    self.state.logs,
                    f"Final selected model: {hpo_result.final_selection.source}.",
                )
                self._complete_stage("final_model_selection")

                final_model_config = ModelConfig(**hpo_result.final_model_config)
                hpo_metadata = {
                    "enabled": True,
                    "max_rounds": hpo_cfg.max_hpo_rounds,
                    "rounds_completed": hpo_result.rounds_completed,
                    "final_model_source": hpo_result.final_selection.source,
                    "final_params": hpo_result.final_selection.params,
                    "baseline_assessment": (
                        hpo_result.baseline_assessment.model_dump()
                        if hpo_result.baseline_assessment
                        else {}
                    ),
                    "final_assessment": (
                        hpo_result.final_assessment.model_dump()
                        if hpo_result.final_assessment
                        else {}
                    ),
                }
                self.artifact_paths.update(
                    {
                        "baseline_cv_metrics": hpo_result.baseline_cv.fold_metrics_path
                        if hpo_result.baseline_cv
                        else "",
                        "baseline_overfitting_assessment": hpo_result.baseline_assessment_path,
                        "hpo_iteration_log": hpo_result.iteration_log_md_path,
                        "hpo_final_selection": hpo_result.final_selection_json_path,
                        "hpo_summary_plot": hpo_result.summary_plot_png_path,
                    }
                )
                for rr in hpo_result.rounds:
                    self.artifact_paths[f"hpo_round_{rr.round_index}_results"] = (
                        rr.search_results_path
                    )

            # Build RF branch result and optionally try fallback / SFS-fixed GA expansion
            rf_branch = ModelBranchResult(
                estimator=self.config.model.estimator,
                model_config_snapshot=hpo_result.final_model_config,
                branch_dir=str(self.run_dir),
                sfs=sfs,
                feature_count=feature_count,
                ga=ga,
                hpo_result=hpo_result,
            )

            winning_estimator = self.config.model.estimator
            winning_features = ga.selected_features
            model_comparison_summary = ""
            winner_is_expansion = False
            winner_expansion_label = ""
            branch_external_artifacts: list[BranchExternalArtifacts] = []
            fallback_branches: list[ModelBranchResult] = []
            hpo_metadata.setdefault("winning_estimator", winning_estimator)
            hpo_metadata.setdefault("model_fallback_triggered", False)
            hpo_metadata.setdefault("fallback_models_tried", [])
            hpo_metadata.setdefault("winner_is_expansion", False)
            hpo_metadata.setdefault("winner_expansion_label", "")

            rf_acceptable = (
                hpo_result.final_selection is not None
                and hpo_result.final_selection.assessment.is_acceptable
            )

            if not hpo_result.final_selection:
                self._skip_stage("model_fallback", "No HPO selection available")
            elif rf_acceptable:
                self._skip_stage("model_fallback", "RF model acceptable")
            else:
                # RF not acceptable: run expansion (+ fallbacks if enabled) and compete.
                self._start_stage("model_fallback")

                def grid_proposer_fallback(**kwargs):
                    return propose_hyperparameter_grid(
                        openai_model=hpo_cfg.openai_model or None,
                        **kwargs,
                    )

                fallback_result = run_model_fallback_if_needed(
                    rf_branch,
                    train_path=preprocessing.preprocessed_train_path,
                    test_path=preprocessing.preprocessed_test_path,
                    run_dir=self.run_dir,
                    workflow_config=self.config,
                    hpo_config=hpo_cfg,
                    grid_proposer=grid_proposer_fallback if hpo_cfg.enabled else None,
                    log_callback=hpo_log,
                    activity_label=self.config.activity_column or "activity",
                    dataset_hash=dataset_hash,
                    config_snapshot=self.config.to_dict(),
                )
                rf_branch = fallback_result.rf_branch
                fallback_branches = list(fallback_result.fallback_branches)
                branch_external_artifacts = list(fallback_result.branch_external_artifacts)
                cross = fallback_result.cross_model_selection
                if cross:
                    final_model_config = ModelConfig(**cross.final_model_config)
                    winning_estimator = cross.winning_estimator
                    winning_features = cross.selected_features
                    winner_is_expansion = cross.winner_is_expansion
                    winner_expansion_label = cross.winner_expansion_label
                    model_comparison_summary = cross.selection_rationale
                    if cross.warning:
                        append_warning(self.warnings, cross.warning)
                        model_comparison_summary += f" Warning: {cross.warning}"
                    hpo_metadata["winning_estimator"] = winning_estimator
                    hpo_metadata["fallback_models_tried"] = fallback_result.fallback_models_tried
                    hpo_metadata["model_fallback_triggered"] = fallback_result.triggered
                    hpo_metadata["winner_is_expansion"] = cross.winner_is_expansion
                    hpo_metadata["winner_expansion_label"] = cross.winner_expansion_label
                    if fallback_result.comparison_json_path:
                        self.artifact_paths["model_comparison"] = (
                            fallback_result.comparison_json_path
                        )
                        self.artifact_paths["model_comparison_csv"] = (
                            fallback_result.comparison_csv_path
                        )
                msg = (
                    f"Tried {len(fallback_result.fallback_models_tried)} fallback model(s); "
                    f"winner: {winning_estimator}"
                    if fallback_result.triggered
                    else f"Winner: {winning_estimator}"
                )
                self._complete_stage("model_fallback", msg)

            # External evaluation: every completed branch gets scatter + Williams in its dir.
            # Winner artifacts are promoted to run root (no second train for that branch).
            self._start_stage("final_model")
            activity_label = self.config.activity_column or "activity"
            if not branch_external_artifacts:
                art, modeling, ad = evaluate_branch_on_external_test(
                    rf_branch,
                    train_path=preprocessing.preprocessed_train_path,
                    test_path=preprocessing.preprocessed_test_path,
                    activity_label=activity_label,
                    dataset_hash=dataset_hash,
                    config_snapshot=self.config.to_dict(),
                    hpo_metadata=hpo_metadata,
                )
                branch_external_artifacts = [art]
            else:
                branches = flatten_branches(rf_branch, *fallback_branches)
                winner_branch = find_winning_branch(
                    branches,
                    winning_estimator=winning_estimator,
                    selected_features=winning_features,
                    winner_is_expansion=winner_is_expansion,
                    winner_expansion_label=winner_expansion_label,
                ) or rf_branch
                winner_art = next(
                    (
                        a
                        for a in branch_external_artifacts
                        if Path(a.branch_dir).resolve()
                        == Path(winner_branch.branch_dir).resolve()
                    ),
                    None,
                )
                if winner_art is None:
                    _, modeling, ad = evaluate_branch_on_external_test(
                        winner_branch,
                        train_path=preprocessing.preprocessed_train_path,
                        test_path=preprocessing.preprocessed_test_path,
                        activity_label=activity_label,
                        dataset_hash=dataset_hash,
                        config_snapshot=self.config.to_dict(),
                        hpo_metadata=hpo_metadata,
                    )
                else:
                    modeling, ad = load_modeling_and_ad_from_artifacts(
                        winner_art, hpo_metadata=hpo_metadata
                    )
                promote_branch_artifacts_to_run_dir(winner_branch.branch_dir, self.run_dir)
                modeling, ad = modeling_result_with_run_dir_paths(modeling, ad, self.run_dir)

            save_json(
                self.run_dir / "branch_external_artifacts.json",
                [a.model_dump() for a in branch_external_artifacts],
            )
            self.artifact_paths["branch_external_artifacts"] = str(
                self.run_dir / "branch_external_artifacts.json"
            )
            self.artifact_paths.update(
                {
                    "predictions": modeling.predictions_path,
                    "final_model": modeling.model_path,
                    "prediction_scatter": modeling.scatter_png_path,
                    "run_manifest": modeling.manifest_path,
                }
            )
            self._complete_stage("final_model")

            self._start_stage("applicability_domain")
            self.artifact_paths["williams_plot"] = ad.williams_png_path
            self._complete_stage("applicability_domain")

            self.state.warnings = self.warnings
            self.state.artifact_paths = self.artifact_paths
            self.state.final_report = build_final_report(
                self.run_id,
                validation,
                descriptors,
                preprocessing,
                split,
                feature_count,
                ga,
                modeling,
                ad,
                self.artifact_paths,
                self.warnings,
                estimator=winning_estimator,
                model_comparison_summary=model_comparison_summary,
            )

            zip_path = create_zip_archive(self.run_dir, self.run_id)
            self.state.zip_path = str(zip_path)
            self.artifact_paths["zip"] = str(zip_path)
            append_log(self.state.logs, "Workflow completed successfully.")
            self._notify()

        except Exception as exc:
            logger.exception("Workflow failed")
            for stage_info in self.state.stages:
                if stage_info.status == StageStatus.RUNNING:
                    self._fail_stage(stage_info.stage, str(exc))
                    break
            raise

        return self.state
