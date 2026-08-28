"""Feature-selection methods, nested protocol, and multi-seed stability."""

from __future__ import annotations

import pytest

from qsar_agent.agentic.tools import _pareto_pick, execute_tool, run_feature_selection_search
from qsar_agent.config import GAConfig, ModelConfig
from qsar_agent.schemas.agentic import PipelinePhase
from qsar_agent.tools.feature_selection_methods import select_features
from qsar_agent.tools.feature_stability import consensus_subset, pairwise_jaccard, summarize_stability
from qsar_agent.tools.genetic_algorithm import run_genetic_algorithm
from qsar_agent.tools.sequential_feature_selection import run_sequential_feature_selection
from tests.agentic_harness import default_agent_config, write_agent_run, write_development_tables


def test_select_features_methods_use_train_only(tmp_path):
    run_dir = tmp_path / "fs"
    run_dir.mkdir()
    names = write_development_tables(run_dir)
    train = run_dir / "preprocessed_train_descriptors.csv"
    for method in (
        "rfe",
        "elastic_net",
        "mutual_information",
        "model_embedded",
        "stability_selection",
    ):
        out = run_dir / method
        selected = select_features(
            train,
            method=method,  # type: ignore[arg-type]
            n_features=3,
            out_dir=out,
            model_config=ModelConfig(),
            random_seed=0,
            n_stability_draws=4,
        )
        assert 1 <= len(selected["selected_features"]) <= 3
        assert all(name in names for name in selected["selected_features"])
        assert (out / "selected_features.json").is_file()


def test_pca_pls_require_allow_latent(tmp_path):
    run_dir = tmp_path / "latent"
    run_dir.mkdir()
    write_development_tables(run_dir)
    train = run_dir / "preprocessed_train_descriptors.csv"
    with pytest.raises(ValueError, match="Latent-component"):
        select_features(train, "pca", 2, run_dir / "pca_no", allow_latent=False)
    allowed = select_features(train, "pca", 2, run_dir / "pca_yes", allow_latent=True)
    assert allowed["selected_features"][0].startswith("pca_component_")
    pls = select_features(train, "pls", 2, run_dir / "pls_yes", allow_latent=True)
    assert pls["selected_features"][0].startswith("pls_component_")


def test_forward_and_backward_sfs(tmp_path):
    run_dir = tmp_path / "sfs"
    run_dir.mkdir()
    write_development_tables(run_dir)
    train = run_dir / "preprocessed_train_descriptors.csv"
    val = run_dir / "preprocessed_val_descriptors.csv"
    fwd = run_sequential_feature_selection(
        train, run_dir / "fwd", max_features=3, cv_folds=3, n_jobs=1, val_path=None, forward=True
    )
    assert [row.n_features for row in fwd.results] == [1, 2, 3]
    assert all(row.val_r2 is None for row in fwd.results)
    bwd = run_sequential_feature_selection(
        train, run_dir / "bwd", max_features=3, cv_folds=3, n_jobs=1, val_path=val, forward=False
    )
    assert any(row.n_features == 3 for row in bwd.results)
    assert all(row.val_r2 is not None for row in bwd.results)


def test_ga_nested_search_does_not_take_val_path(tmp_path, monkeypatch):
    run_dir = tmp_path / "ga"
    run_dir.mkdir()
    write_development_tables(run_dir)
    train = run_dir / "preprocessed_train_descriptors.csv"
    seen = []
    real = run_genetic_algorithm

    def wrapped(*args, **kwargs):
        seen.append(kwargs.get("val_path", "missing"))
        return real(*args, **kwargs)

    monkeypatch.setattr("qsar_agent.agentic.tools.run_genetic_algorithm", wrapped)
    ctx = {
        "config": default_agent_config(),
        "run_dir": run_dir,
        "parent_id": None,
        "selected_features": None,
    }
    result = run_feature_selection_search(
        ctx,
        {
            "method": "genetic_algorithm",
            "n_features": 3,
            "seeds": [1, 2],
            "population_size": 6,
            "generations": 2,
            "cv_folds": 3,
        },
        "ga-exp",
    )
    assert seen
    assert all(path is None for path in seen)
    assert result.selected_features
    extra = result.extra
    assert "stability" in extra
    assert "per_seed" in extra
    assert extra["stability"]["stability_status"] in {"stable", "mixed", "unstable"}


def test_ga_objective_without_validation(tmp_path):
    run_dir = tmp_path / "obj"
    run_dir.mkdir()
    write_development_tables(run_dir)
    ga = run_genetic_algorithm(
        run_dir / "preprocessed_train_descriptors.csv",
        run_dir,
        number_of_features=3,
        ga_config=GAConfig(population_size=6, n_generations=2, cv_folds=3, n_jobs=1),
        val_path=None,
        objective={"name": "cv_r2_minus_complexity", "feature_count_penalty": 0.05},
    )
    assert len(ga.selected_features) == 3


def test_stability_summary_and_consensus():
    subsets = [
        ["a", "b", "c"],
        ["a", "b", "d"],
        ["a", "b", "c"],
    ]
    report = summarize_stability(subsets)
    assert report.selection_frequency["a"] == pytest.approx(1.0)
    assert report.mean_pairwise_jaccard == pytest.approx(pairwise_jaccard([set(s) for s in subsets]))
    consensus = consensus_subset(subsets, min_frequency=0.5, max_size=3)
    assert consensus[0] in {"a", "b"}


def test_one_se_pareto_picks_smallest_eligible_subset():
    candidates = [
        {"selected_features": ["a", "b", "c", "d"], "metrics": {"cv_r2": 0.80, "cv_r2_std": 0.04, "feature_count": 4}},
        {"selected_features": ["a", "b"], "metrics": {"cv_r2": 0.78, "cv_r2_std": 0.04, "feature_count": 2}},
        {"selected_features": ["a"], "metrics": {"cv_r2": 0.50, "cv_r2_std": 0.04, "feature_count": 1}},
    ]
    picked = _pareto_pick(candidates)
    assert picked["selected_features"] == ["a", "b"]


def test_execute_tool_pca_requests_capability_path(tmp_path):
    run_dir = write_agent_run(tmp_path, passing=False)
    write_development_tables(run_dir)
    with pytest.raises(ValueError, match="Latent-component|PCA/PLS"):
        execute_tool(
            "run_feature_selection_search",
            {"method": "pca", "n_features": 2},
            run_dir=run_dir,
            state_phase=PipelinePhase.DEVELOPMENT,
            dataset_hash="ds",
            development_split_hash="dev",
        )
