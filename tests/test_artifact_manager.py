"""Tests for artifact manager."""

from pathlib import Path

from qsar_agent.services.artifact_manager import copy_input_dataset


def test_copy_input_dataset_skips_when_source_is_dest(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    source = run_dir / "input_dataset.csv"
    source.write_text("compound_id,smiles,activity\n1,CCO,1.0\n")
    result = copy_input_dataset(source, run_dir)
    assert result.resolve() == source.resolve()
    assert source.read_text().startswith("compound_id")
