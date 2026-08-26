"""End-to-end workflow smoke test."""

from pathlib import Path
from unittest.mock import patch

import pytest

from qsar_agent.config import GAConfig, HPOSettings, ModelFallbackSettings, SFSConfig, WorkflowConfig
from qsar_agent.schemas.handoff import HandoffPackage
from qsar_agent.services.handoff import format_metric
from qsar_agent.services.workflow_runner import WorkflowRunner
from qsar_agent.schemas.workflow import StageStatus
from qsar_agent.tools.descriptor_calculation import calculate_descriptors
from tests.descriptor_test_utils import fake_descjocky_pipeline


def _calc_with_fake(*args, **kwargs):
    kwargs["pipeline_runner"] = fake_descjocky_pipeline
    return calculate_descriptors(*args, **kwargs)


@pytest.mark.slow
def test_workflow_smoke(synthetic_dataset, tmp_run_dir):
    config = WorkflowConfig(
        output_dir=str(tmp_run_dir),
        smiles_column="smiles",
        activity_column="pIC50",
        id_column="compound_id",
        min_valid_compounds=15,
        ga=GAConfig(population_size=15, n_generations=3, cv_folds=3),
        sfs=SFSConfig(max_features=5, cv_folds=3),
        hpo=HPOSettings(enabled=False),
        model_fallback=ModelFallbackSettings(enabled=False),
    )

    with patch("qsar_agent.agents.qsar_agent.get_openai_api_key", return_value=None):
        with patch(
            "qsar_agent.services.workflow_runner.calculate_descriptors",
            side_effect=_calc_with_fake,
        ):
            runner = WorkflowRunner(config, synthetic_dataset)
            state = runner.run()

    assert state.final_report is not None
    completed = [s for s in state.stages if s.status == StageStatus.COMPLETED]
    assert len(completed) == 9
    assert state.zip_path is not None
    assert state.final_report.initial_descriptor_count > 0

    report_dir = Path(runner.run_dir) / "final_report"
    assert report_dir.is_dir()
    manifest_path = report_dir / "handoff_manifest.json"
    ledger_path = report_dir / "experiment_ledger.csv"
    md_path = report_dir / "modeling_handoff.md"
    assert manifest_path.is_file()
    assert ledger_path.is_file()
    assert md_path.is_file()
    package = HandoffPackage.model_validate_json(manifest_path.read_text(encoding="utf-8"))
    assert len(package.experiments) >= 1
    assert len({e.run_id for e in package.experiments}) == len(package.experiments)
    winner = next(e for e in package.experiments if e.is_winner)
    csv_header, *csv_rows = ledger_path.read_text(encoding="utf-8").splitlines()
    assert "run_id" in csv_header
    assert len(csv_rows) == len(package.experiments)
    winner_row = next(r for r in csv_rows if r.startswith(winner.run_id) or f"{winner.run_id}," in r)
    assert format_metric(winner.metrics.cv_r2) in winner_row
    assert format_metric(winner.metrics.train_r2) in winner_row
    for exp in package.experiments:
        for plot in (
            exp.artifacts.observed_vs_predicted,
            exp.artifacts.williams,
            exp.artifacts.residuals,
        ):
            if plot.status == "available" and plot.relative_path:
                assert (report_dir / plot.relative_path).is_file()
        if exp.artifacts.config:
            assert (report_dir / exp.artifacts.config).is_file()
    assert package.leakage_safeguards.test_results_used_for_selection is False
