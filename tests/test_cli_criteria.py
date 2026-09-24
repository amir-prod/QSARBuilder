"""Tests for how the CLI establishes the user's acceptance criteria."""

from __future__ import annotations

import io
import json

import pytest

from qsar_agent.cli import (
    CriteriaResolutionError,
    build_parser,
    command_show_criteria,
    load_criteria_file,
    main,
    prompt_for_criteria,
    resolve_criteria,
)
from qsar_agent.schemas.agentic import TaskDetection

REGRESSION_DETECTION = TaskDetection(
    task="regression", reason="continuous values", n_unique_values=120
)
CLASSIFICATION_DETECTION = TaskDetection(
    task="classification",
    reason="binary labels",
    n_unique_values=2,
    n_classes=2,
    class_counts={"0": 60, "1": 40},
    imbalance_ratio=1.5,
)


def parse_run(*extra: str):
    argv = [
        "run",
        "--csv", "data.csv",
        "--smiles-col", "smiles",
        "--activity-col", "activity",
        *extra,
    ]
    return build_parser().parse_args(argv)


class FakeTTY(io.StringIO):
    """A stream that reports itself as interactive."""

    def __init__(self, text: str = "", interactive: bool = True):
        super().__init__(text)
        self._interactive = interactive

    def isatty(self) -> bool:
        return self._interactive


# -- interactive prompting -----------------------------------------------------


def test_pressing_enter_accepts_every_suggestion():
    out = io.StringIO()
    values = prompt_for_criteria(
        "regression", REGRESSION_DETECTION, {}, FakeTTY("\n" * 10), out
    )
    assert values["min_test_r2"] == pytest.approx(0.60)
    assert values["min_cv_q2"] == pytest.approx(0.50)
    assert values["min_ad_coverage_pct"] == pytest.approx(80.0)
    # The dataset-specific error criteria stay off unless asked for.
    assert values["max_test_rmse"] is None


def test_the_prompt_shows_the_suggestion_and_its_basis():
    out = io.StringIO()
    prompt_for_criteria("regression", REGRESSION_DETECTION, {}, FakeTTY("\n" * 10), out)
    text = out.getvalue()
    assert "Suggested: 0.6" in text
    assert "Tropsha" in text
    assert "OECD" in text
    assert "What are you looking for" in text


def test_optional_criteria_explain_why_they_are_off():
    out = io.StringIO()
    prompt_for_criteria("regression", REGRESSION_DETECTION, {}, FakeTTY("\n" * 10), out)
    text = out.getvalue()
    assert "Disabled unless you give a value" in text
    assert "depends on your activity scale" in text
    assert "No general value exists (no general value exists" not in text


def test_the_prompt_reports_the_detected_task():
    out = io.StringIO()
    prompt_for_criteria(
        "classification", CLASSIFICATION_DETECTION, {}, FakeTTY("\n" * 10), out
    )
    assert "classification" in out.getvalue()
    assert "binary labels" in out.getvalue()


def test_a_typed_number_overrides_the_suggestion():
    out = io.StringIO()
    values = prompt_for_criteria(
        "regression", REGRESSION_DETECTION, {}, FakeTTY("0.85\n" + "\n" * 10), out
    )
    assert values["min_test_r2"] == pytest.approx(0.85)


def test_off_disables_a_criterion():
    out = io.StringIO()
    values = prompt_for_criteria(
        "regression", REGRESSION_DETECTION, {}, FakeTTY("off\n" + "\n" * 10), out
    )
    assert values["min_test_r2"] is None


@pytest.mark.parametrize("word", ["off", "skip", "none", "disable"])
def test_several_words_disable_a_criterion(word):
    out = io.StringIO()
    values = prompt_for_criteria(
        "regression", REGRESSION_DETECTION, {}, FakeTTY(f"{word}\n" + "\n" * 10), out
    )
    assert values["min_test_r2"] is None


def test_unparseable_input_is_rejected_with_guidance():
    out = io.StringIO()
    with pytest.raises(CriteriaResolutionError, match="or 'off' to disable"):
        prompt_for_criteria(
            "regression", REGRESSION_DETECTION, {}, FakeTTY("very good\n"), out
        )


def test_command_line_values_are_not_re_prompted():
    out = io.StringIO()
    values = prompt_for_criteria(
        "regression", REGRESSION_DETECTION, {"min_test_r2": 0.77}, FakeTTY("\n" * 10), out
    )
    assert values["min_test_r2"] == pytest.approx(0.77)
    assert "from the command line" in out.getvalue()


def test_classification_prompts_for_class_metrics():
    out = io.StringIO()
    values = prompt_for_criteria(
        "classification", CLASSIFICATION_DETECTION, {}, FakeTTY("\n" * 10), out
    )
    assert values["min_test_roc_auc"] == pytest.approx(0.70)
    assert values["min_test_mcc"] == pytest.approx(0.30)
    assert values["min_test_balanced_accuracy"] == pytest.approx(0.65)
    assert "min_test_r2" not in values


def test_the_classification_scramble_suggestion_is_auc_scaled():
    out = io.StringIO()
    values = prompt_for_criteria(
        "classification", CLASSIFICATION_DETECTION, {}, FakeTTY("\n" * 10), out
    )
    assert values["max_scramble_score"] == pytest.approx(0.60)


# -- flags and files -----------------------------------------------------------


def test_flags_supply_criteria_without_prompting():
    args = parse_run("--min-test-r2", "0.8", "--min-cv-q2", "0.7", "--non-interactive",
                     "--max-train-cv-gap", "0.2", "--max-scramble-score", "0.1",
                     "--min-scramble-margin", "0.3", "--min-ad-coverage", "75")
    criteria = resolve_criteria(
        args, "regression", REGRESSION_DETECTION, FakeTTY(interactive=False), io.StringIO()
    )
    assert criteria.min_test_r2 == pytest.approx(0.8)
    assert criteria.min_ad_coverage_pct == pytest.approx(75.0)


def test_a_flag_can_disable_a_criterion():
    args = parse_run("--min-test-r2", "0.8", "--min-cv-q2", "off",
                     "--max-train-cv-gap", "off", "--max-scramble-score", "off",
                     "--min-scramble-margin", "off", "--min-ad-coverage", "off",
                     "--non-interactive")
    criteria = resolve_criteria(
        args, "regression", REGRESSION_DETECTION, FakeTTY(interactive=False), io.StringIO()
    )
    assert criteria.enabled_names("regression") == ["min_test_r2"]


def test_accept_suggested_skips_the_prompt():
    args = parse_run("--accept-suggested")
    criteria = resolve_criteria(
        args, "regression", REGRESSION_DETECTION, FakeTTY(interactive=False), io.StringIO()
    )
    assert criteria.min_test_r2 == pytest.approx(0.60)
    assert criteria.max_test_rmse is None


def test_flags_override_accept_suggested():
    args = parse_run("--accept-suggested", "--min-test-r2", "0.95")
    criteria = resolve_criteria(
        args, "regression", REGRESSION_DETECTION, FakeTTY(interactive=False), io.StringIO()
    )
    assert criteria.min_test_r2 == pytest.approx(0.95)
    assert criteria.min_cv_q2 == pytest.approx(0.50)


def test_a_complete_criteria_file_needs_no_other_input(tmp_path):
    path = tmp_path / "criteria.json"
    path.write_text(
        json.dumps(
            {
                "min_test_r2": 0.72,
                "min_cv_q2": None,
                "max_train_cv_gap": 0.25,
                "max_scramble_score": 0.15,
                "min_scramble_margin": 0.30,
                "min_ad_coverage_pct": 85.0,
            }
        )
    )
    args = parse_run("--criteria-file", str(path), "--non-interactive")
    criteria = resolve_criteria(
        args, "regression", REGRESSION_DETECTION, FakeTTY(interactive=False), io.StringIO()
    )
    assert criteria.min_test_r2 == pytest.approx(0.72)
    assert criteria.min_cv_q2 is None
    assert criteria.min_ad_coverage_pct == pytest.approx(85.0)


def test_a_partial_criteria_file_still_needs_the_rest(tmp_path):
    path = tmp_path / "criteria.json"
    path.write_text(json.dumps({"min_test_r2": 0.72}))
    args = parse_run("--criteria-file", str(path), "--non-interactive")
    with pytest.raises(CriteriaResolutionError, match="max_train_cv_gap"):
        resolve_criteria(
            args, "regression", REGRESSION_DETECTION, FakeTTY(interactive=False), io.StringIO()
        )


def test_a_partial_file_combines_with_accept_suggested(tmp_path):
    path = tmp_path / "criteria.json"
    path.write_text(json.dumps({"min_test_r2": 0.72}))
    args = parse_run("--criteria-file", str(path), "--accept-suggested")
    criteria = resolve_criteria(
        args, "regression", REGRESSION_DETECTION, FakeTTY(interactive=False), io.StringIO()
    )
    assert criteria.min_test_r2 == pytest.approx(0.72)
    assert criteria.min_cv_q2 == pytest.approx(0.50)


def test_flags_take_precedence_over_the_file(tmp_path):
    path = tmp_path / "criteria.json"
    path.write_text(json.dumps({"min_test_r2": 0.50}))
    args = parse_run("--criteria-file", str(path), "--min-test-r2", "0.90",
                     "--accept-suggested")
    criteria = resolve_criteria(
        args, "regression", REGRESSION_DETECTION, FakeTTY(interactive=False), io.StringIO()
    )
    assert criteria.min_test_r2 == pytest.approx(0.90)


def test_a_criteria_file_accepts_flag_style_keys(tmp_path):
    path = tmp_path / "criteria.json"
    path.write_text(json.dumps({"--min-test-r2": 0.65}))
    assert load_criteria_file(path) == {"min_test_r2": 0.65}


def test_an_unknown_criterion_in_a_file_is_rejected(tmp_path):
    path = tmp_path / "criteria.json"
    path.write_text(json.dumps({"min_vibes": 0.9}))
    with pytest.raises(CriteriaResolutionError, match="Unknown acceptance criterion"):
        load_criteria_file(path)


def test_a_non_numeric_file_value_is_rejected(tmp_path):
    path = tmp_path / "criteria.json"
    path.write_text(json.dumps({"min_test_r2": "high"}))
    with pytest.raises(CriteriaResolutionError, match="must be a number or null"):
        load_criteria_file(path)


def test_a_criteria_file_must_be_an_object(tmp_path):
    path = tmp_path / "criteria.json"
    path.write_text(json.dumps([0.6]))
    with pytest.raises(CriteriaResolutionError, match="JSON object"):
        load_criteria_file(path)


# -- refusing to guess ---------------------------------------------------------


def test_incomplete_criteria_without_a_terminal_is_an_error():
    args = parse_run("--min-test-r2", "0.8")
    with pytest.raises(CriteriaResolutionError, match="no terminal to ask on"):
        resolve_criteria(
            args, "regression", REGRESSION_DETECTION, FakeTTY(interactive=False), io.StringIO()
        )


def test_the_error_names_the_missing_criteria():
    args = parse_run("--non-interactive", "--min-test-r2", "0.8")
    with pytest.raises(CriteriaResolutionError, match="min_cv_q2"):
        resolve_criteria(
            args, "regression", REGRESSION_DETECTION, FakeTTY(interactive=False), io.StringIO()
        )


def test_the_error_suggests_accept_suggested():
    args = parse_run("--non-interactive")
    with pytest.raises(CriteriaResolutionError, match="--accept-suggested"):
        resolve_criteria(
            args, "regression", REGRESSION_DETECTION, FakeTTY(interactive=False), io.StringIO()
        )


def test_disabling_everything_is_refused():
    args = parse_run(
        "--min-test-r2", "off", "--min-cv-q2", "off", "--max-train-cv-gap", "off",
        "--max-scramble-score", "off", "--min-scramble-margin", "off",
        "--min-ad-coverage", "off", "--non-interactive",
    )
    with pytest.raises(CriteriaResolutionError, match="nothing would define success"):
        resolve_criteria(
            args, "regression", REGRESSION_DETECTION, FakeTTY(interactive=False), io.StringIO()
        )


def test_criteria_for_the_other_task_are_ignored_with_a_note():
    args = parse_run("--min-roc-auc", "0.8", "--accept-suggested")
    out = io.StringIO()
    criteria = resolve_criteria(
        args, "regression", REGRESSION_DETECTION, FakeTTY(interactive=False), out
    )
    assert "do not apply to a regression task" in out.getvalue()
    assert criteria.min_test_roc_auc is None


# -- confirmation echo ---------------------------------------------------------


def test_the_chosen_criteria_are_echoed_as_frozen():
    args = parse_run("--accept-suggested")
    out = io.StringIO()
    resolve_criteria(args, "regression", REGRESSION_DETECTION, FakeTTY(interactive=False), out)
    text = out.getvalue()
    assert "frozen from here on" in text
    assert "Hold-out test R2 at least >= 0.6" in text


# -- show-criteria -------------------------------------------------------------


def test_show_criteria_lists_both_tasks():
    out = io.StringIO()
    args = build_parser().parse_args(["show-criteria"])
    assert command_show_criteria(args, out) == 0
    text = out.getvalue()
    assert "regression" in text and "classification" in text
    assert "--min-test-r2" in text and "--min-roc-auc" in text
    assert "suggestions, not defaults" in text


def test_show_criteria_can_be_limited_to_one_task():
    out = io.StringIO()
    args = build_parser().parse_args(["show-criteria", "--task", "classification"])
    command_show_criteria(args, out)
    assert "--min-test-r2 " not in out.getvalue()


def test_show_criteria_runs_through_main(capsys):
    assert main(["show-criteria", "--task", "regression"]) == 0
    assert "--min-test-r2" in capsys.readouterr().out


# -- run command guards --------------------------------------------------------


def test_a_missing_dataset_is_reported(capsys):
    code = main([
        "run", "--csv", "/nonexistent/path.csv",
        "--smiles-col", "smiles", "--activity-col", "activity",
        "--accept-suggested",
    ])
    assert code == 2
    assert "dataset not found" in capsys.readouterr().out


def test_a_missing_activity_column_is_reported(tmp_path, capsys):
    import pandas as pd

    path = tmp_path / "data.csv"
    pd.DataFrame({"smiles": ["CCO"], "other": [1.0]}).to_csv(path, index=False)
    code = main([
        "run", "--csv", str(path),
        "--smiles-col", "smiles", "--activity-col", "activity",
        "--accept-suggested",
    ])
    assert code == 2
    assert "not found" in capsys.readouterr().out


def test_a_missing_api_key_is_reported_before_any_work(tmp_path, capsys, monkeypatch):
    import pandas as pd

    for name in ("OPENAI_API_KEY", "QSAR_LLM_API_KEY", "QSAR_LLM_PROVIDER"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr("qsar_agent.cli.load_env_file", lambda *a, **k: False)

    path = tmp_path / "data.csv"
    pd.DataFrame(
        {"smiles": ["CCO"] * 3, "activity": [1.0, 2.0, 3.0]}
    ).to_csv(path, index=False)

    code = main([
        "run", "--csv", str(path),
        "--smiles-col", "smiles", "--activity-col", "activity",
        "--accept-suggested",
    ])
    assert code == 3
    assert "OPENAI_API_KEY" in capsys.readouterr().out
