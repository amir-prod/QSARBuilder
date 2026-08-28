"""Tests for artifact manager."""

from qsar_agent.services.artifact_manager import (
    atomic_write_text,
    copy_input_dataset,
    hash_sorted_ids,
    save_json,
)


def test_copy_input_dataset_skips_when_source_is_dest(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    source = run_dir / "input_dataset.csv"
    source.write_text("compound_id,smiles,activity\n1,CCO,1.0\n")
    result = copy_input_dataset(source, run_dir)
    assert result.resolve() == source.resolve()
    assert source.read_text().startswith("compound_id")


def test_atomic_write_text_replaces_file(tmp_path):
    path = tmp_path / "nested" / "state.json"
    atomic_write_text(path, '{"ok": true}')
    assert path.read_text(encoding="utf-8") == '{"ok": true}'
    save_json(path, {"a": 1})
    assert '"a"' in path.read_text(encoding="utf-8")


def test_hash_sorted_ids_is_order_invariant():
    assert hash_sorted_ids(["b", "a"]) == hash_sorted_ids(["a", "b", "a"])
