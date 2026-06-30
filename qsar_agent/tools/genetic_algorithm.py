"""Genetic algorithm feature selection using DEAP (adapted from examples)."""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
from deap import base, creator, tools
from sklearn.model_selection import cross_val_score

from qsar_agent.config import GAConfig, ModelConfig
from qsar_agent.schemas.feature_selection import GAResult
from qsar_agent.services import build_estimator
from qsar_agent.services.artifact_manager import save_json
from qsar_agent.services.plotting import plot_ga_convergence
from qsar_agent.tools.mordred_descriptors import META_COLUMNS

# DEAP creator can only be registered once per process
if not hasattr(creator, "FitnessMax"):
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
if not hasattr(creator, "Individual"):
    creator.create("Individual", list, fitness=creator.FitnessMax)


def _get_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    feature_cols = [c for c in df.columns if c not in META_COLUMNS]
    return df[feature_cols], df["activity"].values.ravel()


def run_genetic_algorithm(
    train_path: str | Path,
    run_dir: Path,
    number_of_features: int,
    ga_config: GAConfig | None = None,
    model_config: ModelConfig | None = None,
) -> GAResult:
    """
    GA feature selection optimizing CV R² on training data only.

    Corrected from examples/ga_feature_selection_regression.py which used the
    external test set for fitness (data leakage).
    """
    cfg = ga_config or GAConfig()
    random.seed(cfg.random_seed)
    np.random.seed(cfg.random_seed)

    df = pd.read_csv(train_path)
    X, y = _get_xy(df)
    feature_names = X.columns.tolist()
    n_features = len(feature_names)
    n_select = number_of_features

    if n_select > n_features:
        raise ValueError(
            f"Cannot select {n_select} features from {n_features} available descriptors."
        )

    estimator_template = model_config or ModelConfig()

    def evaluate_individual(individual):
        selected = [i for i, bit in enumerate(individual) if bit == 1]
        if len(selected) != n_select or len(set(selected)) != n_select:
            return (-1000.0,)
        X_sel = X.iloc[:, selected]
        model = build_estimator(estimator_template)
        scores = cross_val_score(
            model,
            X_sel.values,
            y,
            cv=cfg.cv_folds,
            scoring="r2",
            n_jobs=cfg.n_jobs,
        )
        return (float(scores.mean()),)

    toolbox = base.Toolbox()

    def create_individual():
        individual = [0] * n_features
        selected_indices = random.sample(range(n_features), n_select)
        for idx in selected_indices:
            individual[idx] = 1
        return creator.Individual(individual)

    def fix_individual(individual):
        selected_count = sum(individual)
        if selected_count < n_select:
            available = [i for i, bit in enumerate(individual) if bit == 0]
            to_add = random.sample(available, n_select - selected_count)
            for idx in to_add:
                individual[idx] = 1
        elif selected_count > n_select:
            selected = [i for i, bit in enumerate(individual) if bit == 1]
            to_remove = random.sample(selected, selected_count - n_select)
            for idx in to_remove:
                individual[idx] = 0
        return individual

    def custom_mate(ind1, ind2):
        tools.cxTwoPoint(ind1, ind2)
        fix_individual(ind1)
        fix_individual(ind2)
        return ind1, ind2

    def custom_mutate(individual):
        selected = [i for i, bit in enumerate(individual) if bit == 1]
        unselected = [i for i, bit in enumerate(individual) if bit == 0]
        if selected and unselected and random.random() < cfg.mutation_prob:
            swap_out = random.choice(selected)
            swap_in = random.choice(unselected)
            individual[swap_out] = 0
            individual[swap_in] = 1
        return individual,

    toolbox.register("individual", create_individual)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", evaluate_individual)
    toolbox.register("mate", custom_mate)
    toolbox.register("mutate", custom_mutate)
    toolbox.register("select", tools.selTournament, tournsize=cfg.tournament_size)

    population = toolbox.population(n=cfg.population_size)
    history = []

    for ind in population:
        ind.fitness.values = toolbox.evaluate(ind)

    for gen in range(cfg.n_generations):
        offspring = toolbox.select(population, len(population))
        offspring = list(map(toolbox.clone, offspring))

        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < cfg.crossover_prob:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            if random.random() < cfg.mutation_prob:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        invalid = [ind for ind in offspring if not ind.fitness.valid]
        for ind in invalid:
            ind.fitness.values = toolbox.evaluate(ind)

        population[:] = offspring
        fits = [ind.fitness.values[0] for ind in population]
        history.append(
            {
                "generation": gen + 1,
                "best_fitness": max(fits),
                "avg_fitness": float(np.mean(fits)),
            }
        )

    best_ind = tools.selBest(population, 1)[0]
    selected_indices = [i for i, bit in enumerate(best_ind) if bit == 1]
    if len(selected_indices) != n_select:
        raise RuntimeError(
            f"GA final chromosome has {len(selected_indices)} features, expected {n_select}."
        )

    selected_names = [feature_names[i] for i in selected_indices]

    history_df = pd.DataFrame(history)
    history_path = run_dir / "ga_history.csv"
    history_df.to_csv(history_path, index=False)

    selected_path = run_dir / "ga_selected_features.json"
    with open(selected_path, "w", encoding="utf-8") as f:
        json.dump({"selected_features": selected_names}, f, indent=2)

    config_path = run_dir / "ga_configuration.json"
    save_json(config_path, {**cfg.model_dump(), "number_of_features": n_select})

    png_path = run_dir / "ga_convergence.png"
    svg_path = run_dir / "ga_convergence.svg"
    plot_ga_convergence(history_df, png_path, svg_path)

    return GAResult(
        selected_features=selected_names,
        best_fitness=float(best_ind.fitness.values[0]),
        history_csv_path=str(history_path),
        selected_features_path=str(selected_path),
        configuration_path=str(config_path),
        convergence_png_path=str(png_path),
        convergence_svg_path=str(svg_path),
    )
