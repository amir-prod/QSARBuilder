"""Tests for final model training."""

from qsar_agent.tools.dataset_validation import validate_dataset
from qsar_agent.tools.descriptor_preprocessing import fit_descriptor_preprocessor
from qsar_agent.tools.final_model import train_and_evaluate_final_model
from qsar_agent.tools.genetic_algorithm import run_genetic_algorithm
from qsar_agent.tools.descriptor_calculation import calculate_descriptors
from qsar_agent.tools.umap_split import create_umap_cluster_split
from qsar_agent.config import GAConfig
from tests.descriptor_test_utils import fake_descjocky_pipeline


def test_metrics_and_predictions(synthetic_dataset, tmp_run_dir):
    cleaned = validate_dataset(
        synthetic_dataset, "smiles", "pIC50", "compound_id", tmp_run_dir, min_valid_compounds=15
    )
    descriptors = calculate_descriptors(
        cleaned.cleaned_dataset_path,
        tmp_run_dir,
        pipeline_runner=fake_descjocky_pipeline,
    )
    split = create_umap_cluster_split(descriptors.raw_descriptors_path, tmp_run_dir)
    prep = fit_descriptor_preprocessor(
        split.train_path, split.test_path, tmp_run_dir, val_path=split.val_path
    )
    ga = run_genetic_algorithm(
        prep.preprocessed_train_path, tmp_run_dir, 3,
        ga_config=GAConfig(population_size=15, n_generations=3, cv_folds=3),
        val_path=prep.preprocessed_val_path,
    )
    result = train_and_evaluate_final_model(
        prep.preprocessed_train_path,
        prep.preprocessed_test_path,
        tmp_run_dir,
        ga.selected_features,
        val_path=prep.preprocessed_val_path,
    )
    import pandas as pd
    preds = pd.read_csv(result.predictions_path)
    assert "predicted_activity" in preds.columns
    assert "split" in preds.columns
    assert set(preds["split"]) == {"train", "val", "test"}
    assert result.train_metrics.n_samples > 0
    assert result.val_metrics is not None
    assert result.val_metrics.n_samples > 0
    assert result.test_metrics.n_samples > 0
