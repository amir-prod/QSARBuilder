"""Structured load-time validation of the modeling handoff package."""

from __future__ import annotations

from qsar_agent.agentic.handoff_validator import validate_handoff_dir
from qsar_agent.services.artifact_manager import file_hash, hash_sorted_ids
from tests.test_handoff import _experiment, _package, _plot, _write_views


def test_valid_handoff_passes(tmp_path):
    package = _package()
    report_dir = _write_views(tmp_path, package)
    result = validate_handoff_dir(report_dir, tmp_path)
    assert result.passed is True
    assert result.errors == []
    assert result.package is not None


def test_malformed_manifest_fails(tmp_path):
    report_dir = tmp_path / "final_report"
    report_dir.mkdir()
    (report_dir / "handoff_manifest.json").write_text("{not-json", encoding="utf-8")
    result = validate_handoff_dir(report_dir, tmp_path)
    assert result.passed is False
    assert any("not schema-valid" in err or "is not schema-valid" in err for err in result.errors)


def test_missing_manifest_fails(tmp_path):
    report_dir = tmp_path / "final_report"
    report_dir.mkdir()
    result = validate_handoff_dir(report_dir, tmp_path)
    assert result.passed is False
    assert any("handoff_manifest.json is missing" in err for err in result.errors)


def test_json_csv_disagreement_fails(tmp_path):
    package = _package()
    report_dir = _write_views(tmp_path, package)
    csv_path = report_dir / "experiment_ledger.csv"
    text = csv_path.read_text(encoding="utf-8")
    csv_path.write_text(text.replace("0.720000", "0.111111", 1), encoding="utf-8")
    result = validate_handoff_dir(report_dir, tmp_path)
    assert result.passed is False
    assert any("disagreement" in err or "mismatch" in err.lower() for err in result.errors)


def test_missing_referenced_artifact_fails(tmp_path):
    plots = {
        "observed_vs_predicted": _plot(
            "observed_vs_predicted", available=True, path="plots/missing.png"
        ),
        "williams": _plot("williams"),
        "residuals": _plot("residuals"),
    }
    package = _package(experiments=[_experiment("run1", plots=plots)])
    report_dir = _write_views(tmp_path, package)
    result = validate_handoff_dir(report_dir, tmp_path)
    assert result.passed is False
    assert any("Referenced file missing" in err for err in result.errors)


def test_invalid_train_cv_gap_fails(tmp_path):
    exp = _experiment("gap_bad", cv_r2=0.50, gap=0.10)
    exp.metrics.mean_cv_fold_train_r2 = 0.90
    exp.metrics.oof_cv_r2 = 0.50
    exp.metrics.cv_fold_train_val_gap = 0.10
    exp.metrics.train_cv_r2_gap = 0.10
    package = _package(experiments=[exp])
    report_dir = _write_views(tmp_path, package)
    result = validate_handoff_dir(report_dir, tmp_path)
    assert result.passed is False
    assert any("train–CV gap is inconsistent" in err or "gap is inconsistent" in err for err in result.errors)


def test_placeholder_zero_cv_errors_fail(tmp_path):
    exp = _experiment("placeholder", cv_r2=0.50, gap=0.10)
    exp.metrics.cv_rmse = 0.0
    exp.metrics.cv_mae = 0.0
    package = _package(experiments=[exp])
    report_dir = _write_views(tmp_path, package)
    result = validate_handoff_dir(report_dir, tmp_path)
    assert result.passed is False
    assert any("placeholder" in err for err in result.errors)


def test_hash_mismatch_against_input_dataset(tmp_path):
    package = _package()
    package.dataset_hash = "abc123"
    package.dataset_audit.dataset_hash = "abc123"
    report_dir = _write_views(tmp_path, package)
    input_csv = tmp_path / "input_dataset.csv"
    input_csv.write_text("compound_id,smiles,activity\nC001,CCO,1.0\n", encoding="utf-8")
    assert file_hash(input_csv) != "abc123"
    result = validate_handoff_dir(report_dir, tmp_path)
    assert result.passed is False
    assert any("dataset_hash does not match" in err for err in result.errors)


def test_split_hash_mismatch(tmp_path):
    package = _package()
    package.development_split_hash = "deadbeef"
    package.sealed_test_hash = "cafebabe"
    report_dir = _write_views(tmp_path, package)
    (tmp_path / "split_assignments.csv").write_text(
        "compound_id,split\nC001,train\nC002,val\nC003,test\n",
        encoding="utf-8",
    )
    result = validate_handoff_dir(report_dir, tmp_path)
    assert result.passed is False
    assert any("development_split_hash" in err for err in result.errors)
    actual_dev = hash_sorted_ids(["C001", "C002"])
    assert actual_dev != "deadbeef"
