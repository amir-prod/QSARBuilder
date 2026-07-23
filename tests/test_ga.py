"""Tests for genetic algorithm feature selection."""

import json

from qsar_agent.config import GAConfig
from qsar_agent.tools.dataset_validation import validate_dataset
from qsar_agent.tools.descriptor_preprocessing import fit_descriptor_preprocessor
from qsar_agent.tools.genetic_algorithm import run_genetic_algorithm
from qsar_agent.tools.mordred_descriptors import META_COLUMNS, calculate_mordred_descriptors
from qsar_agent.tools.umap_split import create_umap_cluster_split


def _preprocessed_train(synthetic_dataset, tmp_run_dir):
    cleaned = validate_dataset(
        synthetic_dataset, "smiles", "pIC50", "compound_id", tmp_run_dir, min_valid_compounds=15
    )
    mordred = calculate_mordred_descriptors(cleaned.cleaned_dataset_path, tmp_run_dir)
    split = create_umap_cluster_split(mordred.raw_descriptors_path, tmp_run_dir)
    prep = fit_descriptor_preprocessor(split.train_path, split.test_path, tmp_run_dir)
    return prep.preprocessed_train_path


def test_ga_exact_feature_count(synthetic_dataset, tmp_run_dir):
    train_path = _preprocessed_train(synthetic_dataset, tmp_run_dir)
    ga = run_genetic_algorithm(
        train_path,
        tmp_run_dir,
        number_of_features=3,
        ga_config=GAConfig(population_size=20, n_generations=5, cv_folds=3),
    )
    assert len(ga.selected_features) == 3
    assert len(set(ga.selected_features)) == 3


def test_ga_fixed_features_adds_exactly_n_extras(synthetic_dataset, tmp_run_dir):
    import pandas as pd

    train_path = _preprocessed_train(synthetic_dataset, tmp_run_dir)
    df = pd.read_csv(train_path)
    feature_cols = [c for c in df.columns if c not in META_COLUMNS]
    fixed = feature_cols[:2]
    assert len(feature_cols) >= 4

    out = tmp_run_dir / "ga_fixed"
    out.mkdir()
    ga = run_genetic_algorithm(
        train_path,
        out,
        number_of_features=2,
        ga_config=GAConfig(population_size=20, n_generations=5, cv_folds=3),
        fixed_features=fixed,
    )

    assert ga.selected_features[:2] == fixed
    extras = ga.selected_features[2:]
    assert len(extras) == 2
    assert len(set(ga.selected_features)) == len(ga.selected_features)
    assert not set(extras) & set(fixed)

    with open(out / "ga_selected_features.json", encoding="utf-8") as f:
        payload = json.load(f)
    assert payload["fixed_features"] == fixed
    assert payload["extra_features"] == extras

    with open(out / "ga_configuration.json", encoding="utf-8") as f:
        cfg = json.load(f)
    assert cfg["fixed_features"] == fixed
    assert cfg["extra_features"] == 2
