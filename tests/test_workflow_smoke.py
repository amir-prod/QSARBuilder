"""End-to-end workflow smoke test."""

from unittest.mock import patch

import pytest

from qsar_agent.config import GAConfig, HPOSettings, SFSConfig, WorkflowConfig
from qsar_agent.services.workflow_runner import WorkflowRunner
from qsar_agent.schemas.workflow import StageStatus


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
    )

    with patch("qsar_agent.agents.qsar_agent.get_openai_api_key", return_value=None):
        runner = WorkflowRunner(config, synthetic_dataset)
        state = runner.run()

    assert state.final_report is not None
    completed = [s for s in state.stages if s.status == StageStatus.COMPLETED]
    assert len(completed) == 9
    assert state.zip_path is not None
