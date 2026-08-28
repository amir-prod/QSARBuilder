"""Structured load-time validation of ``final_report/`` for the modeling agent."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

from qsar_agent.schemas.agentic import HandoffValidationResult
from qsar_agent.schemas.handoff import HandoffPackage
from qsar_agent.services.artifact_manager import file_hash, hash_sorted_ids
from qsar_agent.services.handoff import (
    HandoffValidationError,
    TEST_METRIC_KEYS,
    _is_placeholder_cv_error,
    format_metric,
    validate_handoff_package,
)

GAP_TOLERANCE = 1e-4
IMPOSSIBLE_R2 = 1.5


def _finite_or_none(value: Any) -> bool:
    if value is None:
        return True
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(number)


def _check_metrics(package: HandoffPackage) -> list[str]:
    errors: list[str] = []
    ids = [exp.run_id for exp in package.experiments]
    if len(ids) != len(set(ids)):
        errors.append("Experiment IDs are not unique.")
    for exp in package.experiments:
        m = exp.metrics
        for name, value in m.model_dump().items():
            if value is None:
                continue
            if isinstance(value, (int, float)) and not _finite_or_none(value):
                errors.append(f"{exp.run_id}.{name} is NaN or infinite.")
            if name.endswith("_r2") and isinstance(value, (int, float)) and abs(float(value)) > IMPOSSIBLE_R2:
                errors.append(f"{exp.run_id}.{name}={value} is an impossible R².")
            if name.endswith(("_rmse", "_mae")) and isinstance(value, (int, float)) and float(value) < 0:
                errors.append(f"{exp.run_id}.{name} is negative.")
        if _is_placeholder_cv_error(m.cv_rmse, m.cv_r2):
            errors.append(
                f"{exp.run_id}.cv_rmse is a suspicious zero placeholder while cv_r2 is not perfect."
            )
        if _is_placeholder_cv_error(m.cv_mae, m.cv_r2):
            errors.append(
                f"{exp.run_id}.cv_mae is a suspicious zero placeholder while cv_r2 is not perfect."
            )
        oof = m.oof_cv_r2 if m.oof_cv_r2 is not None else m.cv_r2
        fold_train = m.mean_cv_fold_train_r2
        gap = m.cv_fold_train_val_gap if m.cv_fold_train_val_gap is not None else m.train_cv_r2_gap
        if fold_train is not None and oof is not None and gap is not None:
            expected = float(fold_train) - float(oof)
            if abs(expected - float(gap)) > GAP_TOLERANCE:
                errors.append(
                    f"{exp.run_id} train–CV gap is inconsistent: "
                    f"cv_fold_train_val_gap={gap} vs mean_cv_fold_train_r2 - oof_cv_r2={expected}."
                )
        if (
            m.train_cv_r2_gap is not None
            and m.cv_fold_train_val_gap is not None
            and abs(float(m.train_cv_r2_gap) - float(m.cv_fold_train_val_gap)) > GAP_TOLERANCE
        ):
            errors.append(
                f"{exp.run_id}.train_cv_r2_gap does not match cv_fold_train_val_gap "
                "(the acceptance statistic)."
            )
        if m.refit_train_r2 is not None and m.oof_cv_r2 is not None and m.refit_train_cv_gap is not None:
            expected_refit = float(m.refit_train_r2) - float(m.oof_cv_r2)
            if abs(expected_refit - float(m.refit_train_cv_gap)) > GAP_TOLERANCE:
                errors.append(
                    f"{exp.run_id}.refit_train_cv_gap is inconsistent with refit_train_r2 - oof_cv_r2."
                )
    return errors


def _check_leakage(package: HandoffPackage) -> list[str]:
    errors: list[str] = []
    leak = package.leakage_safeguards
    if leak.preprocessing_scope != "train_only_fit":
        errors.append(
            f"Preprocessing was not fitted on train only (scope={leak.preprocessing_scope!r})."
        )
    if leak.test_results_used_for_selection:
        errors.append("External-test results were used for selection.")
    if "test" in (leak.feature_selection_scope or "").lower() and "not" not in leak.feature_selection_scope.lower():
        if "train" not in leak.feature_selection_scope.lower():
            errors.append("Feature selection scope includes the external test set.")
    for row in leak.selection_records:
        for key in row:
            if str(key).lower() in TEST_METRIC_KEYS:
                errors.append(f"Selection record contains external-test metric '{key}'.")
    return errors


def _check_hashes(package: HandoffPackage, run_dir: Path) -> list[str]:
    errors: list[str] = []
    input_csv = run_dir / "input_dataset.csv"
    if input_csv.is_file() and package.dataset_hash:
        actual = file_hash(input_csv)
        top = package.dataset_hash or package.dataset_audit.dataset_hash
        if top and actual != top:
            errors.append("dataset_hash does not match input_dataset.csv.")
        if package.dataset_audit.dataset_hash and package.dataset_hash:
            if package.dataset_audit.dataset_hash != package.dataset_hash:
                errors.append("Top-level dataset_hash disagrees with dataset_audit.dataset_hash.")
    assignments = run_dir / "split_assignments.csv"
    if assignments.is_file() and package.development_split_hash:
        import pandas as pd

        df = pd.read_csv(assignments)
        if "compound_id" in df.columns and "split" in df.columns:
            dev = df[df["split"].astype(str).isin(["train", "val"])]
            actual_dev = hash_sorted_ids(dev["compound_id"].astype(str).tolist())
            if actual_dev != package.development_split_hash:
                errors.append("development_split_hash does not match split_assignments.csv.")
            test = df[df["split"].astype(str) == "test"]
            actual_test = hash_sorted_ids(test["compound_id"].astype(str).tolist())
            if package.sealed_test_hash and actual_test != package.sealed_test_hash:
                errors.append("sealed_test_hash does not match test compound IDs.")
    return errors


def _check_ledger_agreement(report_dir: Path, package: HandoffPackage) -> list[str]:
    errors: list[str] = []
    csv_path = report_dir / "experiment_ledger.csv"
    if not csv_path.is_file():
        return ["experiment_ledger.csv is missing."]
    rows = list(csv.DictReader(csv_path.open(encoding="utf-8")))
    by_id = {row.get("run_id", ""): row for row in rows}
    if len(rows) != len(package.experiments):
        errors.append("experiment_ledger.csv row count does not match experiments.")
    fields = (
        "train_r2",
        "cv_r2",
        "val_r2",
        "train_cv_r2_gap",
        "cv_fold_train_val_gap",
        "refit_train_cv_gap",
        "mean_cv_fold_train_r2",
        "cv_r2_std",
    )
    for exp in package.experiments:
        row = by_id.get(exp.run_id)
        if row is None:
            errors.append(f"CSV missing experiment {exp.run_id}.")
            continue
        for field in fields:
            json_val = format_metric(getattr(exp.metrics, field))
            csv_val = row.get(field, "")
            if json_val != csv_val:
                errors.append(
                    f"Metric disagreement for {exp.run_id}.{field}: json={json_val} csv={csv_val}"
                )
    return errors


def validate_handoff_dir(report_dir: str | Path, run_dir: str | Path | None = None) -> HandoffValidationResult:
    """Load and validate a handoff package. Never silently repairs values."""
    report_dir = Path(report_dir)
    errors: list[str] = []
    warnings: list[str] = []
    manifest = report_dir / "handoff_manifest.json"
    if not manifest.is_file():
        return HandoffValidationResult(passed=False, errors=["handoff_manifest.json is missing."])
    try:
        raw = json.loads(manifest.read_text(encoding="utf-8"))
        package = HandoffPackage.model_validate(raw)
    except Exception as exc:
        return HandoffValidationResult(
            passed=False,
            errors=[f"handoff_manifest.json is not schema-valid: {exc}"],
        )
    if package.schema_version != "1.0":
        errors.append(f"Unsupported schema_version {package.schema_version!r}.")

    try:
        validate_handoff_package(report_dir, package)
    except HandoffValidationError as exc:
        errors.extend([part.strip() for part in str(exc).split(";") if part.strip()])

    errors.extend(_check_metrics(package))
    errors.extend(_check_leakage(package))
    errors.extend(_check_ledger_agreement(report_dir, package))
    parent = Path(run_dir) if run_dir is not None else report_dir.parent
    errors.extend(_check_hashes(package, parent))

    # Deduplicate while preserving order.
    unique: list[str] = []
    seen: set[str] = set()
    for item in errors:
        if item not in seen:
            seen.add(item)
            unique.append(item)

    return HandoffValidationResult(
        passed=not unique,
        errors=unique,
        warnings=warnings,
        package=package.model_dump(mode="json"),
        dataset_hash=package.dataset_hash or package.dataset_audit.dataset_hash,
        development_split_hash=package.development_split_hash,
        sealed_test_hash=package.sealed_test_hash or package.leakage_safeguards.test_compound_id_hash,
    )
