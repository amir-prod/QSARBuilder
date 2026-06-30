"""Tests for final model training."""

from qsar_agent.tools.dataset_validation import validate_dataset
from qsar_agent.tools.descriptor_preprocessing import fit_descriptor_preprocessor
from qsar_agent.tools.final_model import train_and_evaluate_final_model
from qsar_agent.tools.genetic_algorithm import run_genetic_algorithm
from qsar_agent.tools.mordred_descriptors import calculate_mordred_descriptors
from qsar_agent.tools.umap_split import create_umap_cluster_split
from qsar_agent.config import GAConfig


def test_metrics_and_predictions(synthetic_dataset, tmp_run_dir):
    cleaned = validate_dataset(
        synthetic_dataset, "smiles", "pIC50", "compound_id", tmp_run_dir, min_valid_compounds=15
    )
    mordred = calculate_mordred_descriptors(cleaned.cleaned_dataset_path, tmp_run_dir)
    split = create_umap_cluster_split(mordred.raw_descriptors_path, tmp_run_dir)
    prep = fit_descriptor_preprocessor(split.train_path, split.test_path, tmp_run_dir)
    ga = run_genetic_algorithm(
        prep.preprocessed_train_path, tmp_run_dir, 3,
        ga_config=GAConfig(population_size=15, n_generations=3, cv_folds=3),
    )
    result = train_and_evaluate_final_model(
        prep.preprocessed_train_path,
        prep.preprocessed_test_path,
        tmp_run_dir,
        ga.selected_features,
    )
    import pandas as pd
    preds = pd.read_csv(result.predictions_path)
    assert "predicted_activity" in preds.columns
    assert "split" in preds.columns
    assert result.train_metrics.n_samples > 0
    assert result.test_metrics.n_samples > 0
