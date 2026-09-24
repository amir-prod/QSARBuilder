"""Tests for classification-safe dataset ingestion."""

from __future__ import annotations

import pandas as pd
import pytest

from qsar_agent.tools.dataset_ingest import ingest_dataset

VALID_SMILES = [
    "CCO", "CC(C)O", "c1ccccc1", "CC(=O)O", "CCN", "CCC", "CCCC", "CC(C)C",
    "c1ccc(O)cc1", "c1ccc(N)cc1", "C1CCCCC1", "CCOC", "CC(C)OC", "CCCOC",
    "CC(=O)OC", "CNC", "CCNC", "c1ccncc1", "c1ccc(Cl)cc1", "c1ccc(Br)cc1",
    "CCCl", "CCBr", "CCI", "CCCCCC", "CCCCCCC",
]


def write_csv(tmp_path, frame, name="data.csv"):
    path = tmp_path / name
    frame.to_csv(path, index=False)
    return path


def regression_frame():
    return pd.DataFrame(
        {
            "compound_id": [f"C{i:03d}" for i in range(len(VALID_SMILES))],
            "smiles": VALID_SMILES,
            "pIC50": [4.0 + 0.1 * i for i in range(len(VALID_SMILES))],
        }
    )


def classification_frame():
    return pd.DataFrame(
        {
            "smiles": VALID_SMILES,
            "active": [i % 2 for i in range(len(VALID_SMILES))],
        }
    )


def test_regression_dataset_ingests(tmp_path):
    result = ingest_dataset(
        write_csv(tmp_path, regression_frame()), "smiles", "pIC50", tmp_path, "compound_id"
    )
    assert result.task_detection.task == "regression"
    assert result.n_compounds == len(VALID_SMILES)
    assert result.n_invalid_smiles == 0
    frame = pd.read_csv(result.cleaned_dataset_path)
    assert {"canonical_smiles", "compound_id", "activity"} <= set(frame.columns)


def test_classification_labels_survive_ingestion(tmp_path):
    result = ingest_dataset(
        write_csv(tmp_path, classification_frame()), "smiles", "active", tmp_path
    )
    assert result.task_detection.task == "classification"
    assert result.task_detection.n_classes == 2
    assert result.n_compounds == len(VALID_SMILES)


def test_string_class_labels_are_preserved(tmp_path):
    frame = classification_frame()
    frame["active"] = ["active" if v else "inactive" for v in frame["active"]]
    result = ingest_dataset(write_csv(tmp_path, frame), "smiles", "active", tmp_path)
    activities = set(pd.read_csv(result.cleaned_dataset_path)["activity"])
    assert activities == {"active", "inactive"}


def test_smiles_are_canonicalised(tmp_path):
    frame = regression_frame()
    frame.loc[2, "smiles"] = "C1=CC=CC=C1"  # non-canonical benzene
    result = ingest_dataset(
        write_csv(tmp_path, frame), "smiles", "pIC50", tmp_path, "compound_id"
    )
    assert "c1ccccc1" in set(pd.read_csv(result.cleaned_dataset_path)["canonical_smiles"])


def test_invalid_smiles_are_excluded_and_recorded(tmp_path):
    frame = regression_frame()
    frame.loc[len(frame)] = ["C999", "not_a_molecule", 9.9]
    result = ingest_dataset(
        write_csv(tmp_path, frame), "smiles", "pIC50", tmp_path, "compound_id"
    )
    assert result.n_invalid_smiles == 1
    assert result.invalid_rows_path is not None
    assert "not_a_molecule" in pd.read_csv(result.invalid_rows_path)["smiles"].tolist()


def test_missing_activities_are_excluded(tmp_path):
    frame = regression_frame()
    frame.loc[len(frame)] = ["C999", "CCCCCCCC", None]
    result = ingest_dataset(
        write_csv(tmp_path, frame), "smiles", "pIC50", tmp_path, "compound_id"
    )
    assert result.n_missing_activity == 1
    assert result.n_compounds == len(VALID_SMILES)


def test_duplicate_structures_have_their_activities_averaged(tmp_path):
    frame = regression_frame()
    frame.loc[len(frame)] = ["DUP", "CCO", 10.0]
    result = ingest_dataset(
        write_csv(tmp_path, frame), "smiles", "pIC50", tmp_path, "compound_id"
    )
    assert result.n_duplicate_structures == 1
    assert "averaged" in result.duplicate_resolution
    cleaned = pd.read_csv(result.cleaned_dataset_path)
    ethanol = cleaned.loc[cleaned["canonical_smiles"] == "CCO", "activity"].iat[0]
    assert ethanol == pytest.approx((4.0 + 10.0) / 2)


def test_duplicate_structures_get_the_majority_class(tmp_path):
    frame = classification_frame()
    frame.loc[len(frame)] = ["CCO", 0]
    frame.loc[len(frame)] = ["CCO", 0]
    result = ingest_dataset(write_csv(tmp_path, frame), "smiles", "active", tmp_path)
    assert "majority" in result.duplicate_resolution
    cleaned = pd.read_csv(result.cleaned_dataset_path)
    assert str(cleaned.loc[cleaned["canonical_smiles"] == "CCO", "activity"].iat[0]) == "0"


def test_ids_are_generated_when_no_column_is_given(tmp_path):
    result = ingest_dataset(
        write_csv(tmp_path, classification_frame()), "smiles", "active", tmp_path
    )
    ids = pd.read_csv(result.cleaned_dataset_path)["compound_id"].tolist()
    assert all(str(value).startswith("cmp_") for value in ids)
    assert len(set(ids)) == len(ids)


def test_a_missing_column_names_the_available_ones(tmp_path):
    with pytest.raises(ValueError, match="Available columns"):
        ingest_dataset(
            write_csv(tmp_path, regression_frame()), "smiles", "no_such_column", tmp_path
        )


def test_a_missing_id_column_is_reported(tmp_path):
    with pytest.raises(ValueError, match="ID column"):
        ingest_dataset(
            write_csv(tmp_path, regression_frame()), "smiles", "pIC50", tmp_path, "nope"
        )


def test_too_few_compounds_is_refused(tmp_path):
    frame = regression_frame().head(5)
    with pytest.raises(ValueError, match="at least 20"):
        ingest_dataset(write_csv(tmp_path, frame), "smiles", "pIC50", tmp_path)


def test_the_minimum_compound_count_is_configurable(tmp_path):
    frame = regression_frame().head(5)
    result = ingest_dataset(
        write_csv(tmp_path, frame), "smiles", "pIC50", tmp_path, min_compounds=5
    )
    assert result.n_compounds == 5


def test_the_task_can_be_forced(tmp_path):
    result = ingest_dataset(
        write_csv(tmp_path, classification_frame()),
        "smiles",
        "active",
        tmp_path,
        forced_task="regression",
    )
    assert result.task_detection.task == "regression"


def test_a_report_is_written(tmp_path):
    result = ingest_dataset(
        write_csv(tmp_path, regression_frame()), "smiles", "pIC50", tmp_path, "compound_id"
    )
    import json

    report = json.loads((tmp_path / "agentic_ingest_report.json").read_text())
    assert report["n_compounds"] == result.n_compounds
    assert report["task_detection"]["task"] == "regression"


def test_nothing_usable_is_an_error(tmp_path):
    frame = pd.DataFrame({"smiles": ["???", "!!!"], "activity": [1.0, 2.0]})
    with pytest.raises(ValueError, match="No usable rows"):
        ingest_dataset(write_csv(tmp_path, frame), "smiles", "activity", tmp_path)
