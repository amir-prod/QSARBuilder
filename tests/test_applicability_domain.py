"""Tests for applicability domain / Williams plot."""

import numpy as np

from qsar_agent.tools.applicability_domain import calculate_applicability_domain
from qsar_agent.tools.dataset_validation import validate_dataset
from qsar_agent.tools.descriptor_preprocessing import fit_descriptor_preprocessor
from qsar_agent.tools.final_model import train_and_evaluate_final_model
from qsar_agent.tools.genetic_algorithm import run_genetic_algorithm
from qsar_agent.tools.mordred_descriptors import calculate_mordred_descriptors
from qsar_agent.tools.umap_split import create_umap_cluster_split
from qsar_agent.config import GAConfig


def test_williams_leverage_and_classification(synthetic_dataset, tmp_run_dir):
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
    modeling = train_and_evaluate_final_model(
        prep.preprocessed_train_path,
        prep.preprocessed_test_path,
        tmp_run_dir,
        ga.selected_features,
    )
    ad = calculate_applicability_domain(
        prep.preprocessed_train_path,
        prep.preprocessed_test_path,
        modeling.predictions_path,
        tmp_run_dir,
        ga.selected_features,
    )
    p = len(ga.selected_features)
    n_train = split.train_count
    expected_h = 3 * (p + 1) / n_train
    assert abs(ad.summary.warning_leverage - expected_h) < 0.01

    import pandas as pd
    ad_df = pd.read_csv(ad.classifications_path)
    assert "applicability_domain" in ad_df.columns
    assert "in_domain" in ad_df.columns
