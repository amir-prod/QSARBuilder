"""QSAR workflow runner — orchestrates deterministic pipeline stages."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from qsar_agent.agents.qsar_agent import build_final_report, run_agent_feature_count_selection
from qsar_agent.config import WorkflowConfig
from qsar_agent.logging_utils import append_log, append_warning, get_logger
from qsar_agent.schemas.workflow import StageStatus, WorkflowState
from qsar_agent.services.artifact_manager import (
    copy_input_dataset,
    create_zip_archive,
    file_hash,
    generate_run_id,
    get_run_dir,
)
from qsar_agent.services.plotting import plot_sfs_r2
from qsar_agent.tools.applicability_domain import calculate_applicability_domain
from qsar_agent.tools.dataset_validation import validate_dataset
from qsar_agent.tools.descriptor_preprocessing import fit_descriptor_preprocessor
from qsar_agent.tools.final_model import train_and_evaluate_final_model
from qsar_agent.tools.genetic_algorithm import run_genetic_algorithm
from qsar_agent.tools.mordred_descriptors import calculate_mordred_descriptors
from qsar_agent.tools.sequential_feature_selection import run_sequential_feature_selection
from qsar_agent.tools.umap_split import create_umap_cluster_split

logger = get_logger()


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

            # 2. Mordred descriptors
            self._start_stage("mordred_descriptors")
            mordred = calculate_mordred_descriptors(
                validation.cleaned_dataset_path,
                self.run_dir,
                self.config.descriptors,
            )
            self.warnings.extend(mordred.warnings)
            self.artifact_paths["mordred_descriptors"] = mordred.raw_descriptors_path
            self._complete_stage("mordred_descriptors")

            # 3. UMAP split
            self._start_stage("umap_split")
            split = create_umap_cluster_split(
                mordred.raw_descriptors_path,
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

            # 8. Final model
            self._start_stage("final_model")
            modeling = train_and_evaluate_final_model(
                preprocessing.preprocessed_train_path,
                preprocessing.preprocessed_test_path,
                self.run_dir,
                ga.selected_features,
                self.config.model,
                self.config.activity_column or "activity",
                dataset_hash,
                self.config.to_dict(),
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

            # 9. Applicability domain
            self._start_stage("applicability_domain")
            ad = calculate_applicability_domain(
                preprocessing.preprocessed_train_path,
                preprocessing.preprocessed_test_path,
                modeling.predictions_path,
                self.run_dir,
                ga.selected_features,
            )
            self.artifact_paths["williams_plot"] = ad.williams_png_path
            self._complete_stage("applicability_domain")

            self.state.warnings = self.warnings
            self.state.artifact_paths = self.artifact_paths
            self.state.final_report = build_final_report(
                self.run_id,
                validation,
                mordred,
                preprocessing,
                split,
                feature_count,
                ga,
                modeling,
                ad,
                self.artifact_paths,
                self.warnings,
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
