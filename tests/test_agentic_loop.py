"""Tests for the iteration controller: stop conditions, honesty, and reporting."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from qsar_agent.agents.strategist import StrategistError
from qsar_agent.llm.mock import MockLLMClient
from qsar_agent.llm.provider import LLMResult
from qsar_agent.schemas.agentic import AgenticRunReport
from qsar_agent.services.agentic_loop import (
    AgenticConfig,
    AgenticQSARWorkflow,
    render_report,
    run_agentic_workflow,
)
from qsar_agent.tools.acceptance import AcceptanceCriteria

EXAMPLE_DIR = Path(__file__).resolve().parent.parent / "example"
REGRESSION_CSV = EXAMPLE_DIR / "agentic_regression_sample.csv"
CLASSIFICATION_CSV = EXAMPLE_DIR / "agentic_classification_sample.csv"


def base_config(tmp_path, **overrides) -> AgenticConfig:
    defaults = dict(
        smiles_column="smiles",
        activity_column="pIC50",
        id_column="compound_id",
        cv_folds=3,
        scramble_repeats=3,
        max_iterations=2,
        random_seed=42,
        output_dir=str(tmp_path / "outputs"),
        n_jobs=1,
    )
    defaults.update(overrides)
    return AgenticConfig(**defaults)


class QueueClient:
    """Replays a fixed sequence of strategist replies."""

    provider = "queue"
    model = "queue"

    def __init__(self, replies):
        self._replies = list(replies)
        self.contexts: list[dict] = []

    def complete_json(self, system, user, schema_hint, tools=None, max_tool_steps=6):
        self.contexts.append(json.loads(user))
        if not self._replies:
            raise AssertionError("QueueClient ran out of scripted replies")
        reply = self._replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return LLMResult(data=reply, raw_text=json.dumps(reply))


RIDGE_PLAN = {
    "features": {"blocks": ["rdkit_descriptors"], "univariate_top_k": 20},
    "model": {"estimator": "Ridge", "hyperparameters": {"alpha": 1.0}},
    "rationale": "Linear baseline.",
}

FOREST_REVISION = {
    "features": {"blocks": ["rdkit_descriptors"], "univariate_top_k": 30},
    "model": {"estimator": "RandomForest", "hyperparameters": {"n_estimators": 60}},
    "rationale": "More capacity to capture the nonlinear trend.",
    "addresses_diagnosis": "underfit",
}


# -- happy path ----------------------------------------------------------------


@pytest.mark.slow
def test_a_reachable_target_is_accepted_and_reported(tmp_path):
    report = run_agentic_workflow(
        dataset_path=REGRESSION_CSV,
        criteria=AcceptanceCriteria(min_test_r2=0.5, min_cv_q2=0.4),
        config=base_config(tmp_path, max_iterations=3),
        llm_client=MockLLMClient(),
    )
    assert isinstance(report, AgenticRunReport)
    assert report.accepted
    assert report.best_verdict.accepted
    assert report.task == "regression"
    assert report.iterations_run >= 1
    assert "criteria were met" in report.stop_reason


@pytest.mark.slow
def test_the_loop_iterates_when_the_first_plan_fails(tmp_path):
    client = QueueClient([RIDGE_PLAN, FOREST_REVISION])
    report = run_agentic_workflow(
        dataset_path=REGRESSION_CSV,
        criteria=AcceptanceCriteria(min_test_r2=0.55),
        config=base_config(tmp_path, max_iterations=2),
        llm_client=client,
    )
    assert report.iterations_run == 2
    assert report.iterations[0].plan.model.estimator == "Ridge"
    assert report.iterations[1].plan.model.estimator == "RandomForest"
    # The failed first iteration must carry a diagnosis that drove the revision.
    assert report.iterations[0].diagnosis is not None
    assert report.iterations[0].verdict.accepted is False
    assert report.accepted


@pytest.mark.slow
def test_the_diagnosis_is_passed_to_the_strategist(tmp_path):
    client = QueueClient([RIDGE_PLAN, FOREST_REVISION])
    run_agentic_workflow(
        dataset_path=REGRESSION_CSV,
        criteria=AcceptanceCriteria(min_test_r2=0.55),
        config=base_config(tmp_path, max_iterations=2),
        llm_client=client,
    )
    revision_context = client.contexts[1]
    assert revision_context["request"] == "revision"
    assert revision_context["diagnosis"]["label"] in {"underfit", "overfit", "unstable_cv",
                                                      "poor_generalisation"}
    assert revision_context["failed_criteria"]
    assert revision_context["tried_signatures"]
    assert revision_context["acceptance_criteria"]["min_test_r2"]["threshold"] == 0.55


# -- honest failure ------------------------------------------------------------


@pytest.mark.slow
def test_an_unreachable_target_is_reported_as_not_accepted(tmp_path):
    report = run_agentic_workflow(
        dataset_path=REGRESSION_CSV,
        # No model will reach R2 = 0.999 on noisy data.
        criteria=AcceptanceCriteria(min_test_r2=0.999),
        config=base_config(tmp_path, max_iterations=2),
        llm_client=MockLLMClient(),
    )
    assert not report.accepted
    assert report.best_verdict is not None
    assert report.best_verdict.failed_checks
    assert "without meeting the acceptance criteria" in report.stop_reason


@pytest.mark.slow
def test_thresholds_are_never_relaxed_to_manufacture_a_pass(tmp_path):
    criteria = AcceptanceCriteria(min_test_r2=0.999)
    report = run_agentic_workflow(
        dataset_path=REGRESSION_CSV,
        criteria=criteria,
        config=base_config(tmp_path, max_iterations=2),
        llm_client=MockLLMClient(),
    )
    assert report.criteria["min_test_r2"]["threshold"] == pytest.approx(0.999)
    for record in report.iterations:
        for check in record.verdict.checks:
            if check.name == "min_test_r2":
                assert check.threshold == pytest.approx(0.999)


@pytest.mark.slow
def test_the_iteration_cap_is_respected(tmp_path):
    report = run_agentic_workflow(
        dataset_path=REGRESSION_CSV,
        criteria=AcceptanceCriteria(min_test_r2=0.999),
        config=base_config(tmp_path, max_iterations=2),
        llm_client=MockLLMClient(),
    )
    assert report.iterations_run == 2
    assert report.max_iterations == 2


@pytest.mark.slow
def test_a_strategist_stop_ends_the_run_with_its_reason(tmp_path):
    client = QueueClient(
        [RIDGE_PLAN, {"stop": True, "stop_reason": "No revision I can justify remains."}]
    )
    report = run_agentic_workflow(
        dataset_path=REGRESSION_CSV,
        criteria=AcceptanceCriteria(min_test_r2=0.999),
        config=base_config(tmp_path, max_iterations=5),
        llm_client=client,
    )
    assert not report.accepted
    assert report.iterations_run == 1
    assert "No revision I can justify remains" in report.stop_reason


@pytest.mark.slow
def test_an_exhausted_time_budget_stops_the_loop(tmp_path):
    report = run_agentic_workflow(
        dataset_path=REGRESSION_CSV,
        criteria=AcceptanceCriteria(min_test_r2=0.999),
        config=base_config(tmp_path, max_iterations=5, budget_seconds=0.0),
        llm_client=MockLLMClient(),
    )
    assert report.iterations_run == 0
    assert "budget" in report.stop_reason.lower()


@pytest.mark.slow
def test_a_repeated_proposal_is_rejected_and_a_new_one_requested(tmp_path):
    client = QueueClient([RIDGE_PLAN, RIDGE_PLAN, FOREST_REVISION])
    report = run_agentic_workflow(
        dataset_path=REGRESSION_CSV,
        criteria=AcceptanceCriteria(min_test_r2=0.55),
        config=base_config(tmp_path, max_iterations=2),
        llm_client=client,
    )
    assert "rejected_proposal" in client.contexts[2]
    assert report.iterations[1].plan.model.estimator == "RandomForest"


@pytest.mark.slow
def test_a_persistently_repeating_strategist_stops_the_run(tmp_path):
    client = QueueClient([RIDGE_PLAN, RIDGE_PLAN, RIDGE_PLAN])
    report = run_agentic_workflow(
        dataset_path=REGRESSION_CSV,
        criteria=AcceptanceCriteria(min_test_r2=0.999),
        config=base_config(tmp_path, max_iterations=4),
        llm_client=client,
    )
    assert not report.accepted
    assert "already been tested" in report.stop_reason


@pytest.mark.slow
def test_an_execution_error_is_recorded_and_the_loop_continues(tmp_path):
    impossible = {
        # A correlation threshold of 0 removes every descriptor.
        "features": {"blocks": ["rdkit_descriptors"], "variance_threshold": 1e9},
        "model": {"estimator": "Ridge", "hyperparameters": {}},
        "rationale": "Deliberately impossible.",
    }
    client = QueueClient([impossible, FOREST_REVISION])
    report = run_agentic_workflow(
        dataset_path=REGRESSION_CSV,
        criteria=AcceptanceCriteria(min_test_r2=0.5),
        config=base_config(tmp_path, max_iterations=2),
        llm_client=client,
    )
    assert report.iterations[0].error is not None
    assert report.iterations[1].validation is not None


@pytest.mark.slow
def test_a_strategist_failure_stops_the_run_rather_than_falling_back(tmp_path):
    bad = {"model": {"estimator": "DoesNotExist", "hyperparameters": {}}}
    client = QueueClient([RIDGE_PLAN, bad, bad, bad])
    report = run_agentic_workflow(
        dataset_path=REGRESSION_CSV,
        criteria=AcceptanceCriteria(min_test_r2=0.999),
        config=base_config(tmp_path, max_iterations=4),
        llm_client=client,
    )
    assert not report.accepted
    assert "could not produce a valid plan" in report.stop_reason


def test_an_unusable_initial_plan_aborts_the_run(tmp_path):
    bad = {"model": {"estimator": "DoesNotExist", "hyperparameters": {}}}
    with pytest.raises(StrategistError):
        run_agentic_workflow(
            dataset_path=REGRESSION_CSV,
            criteria=AcceptanceCriteria(min_test_r2=0.6),
            config=base_config(tmp_path),
            llm_client=QueueClient([bad, bad, bad]),
        )


# -- configuration guards ------------------------------------------------------

def test_criteria_for_the_wrong_task_are_refused(tmp_path):
    # Classification thresholds cannot govern a regression endpoint.
    with pytest.raises(ValueError, match="No acceptance criteria are enabled"):
        run_agentic_workflow(
            dataset_path=REGRESSION_CSV,
            criteria=AcceptanceCriteria(min_test_roc_auc=0.7),
            config=base_config(tmp_path),
            llm_client=MockLLMClient(),
        )


def test_a_tiny_dataset_is_refused(tmp_path):
    frame = pd.read_csv(REGRESSION_CSV).head(5)
    path = tmp_path / "tiny.csv"
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match="usable compounds"):
        run_agentic_workflow(
            dataset_path=path,
            criteria=AcceptanceCriteria(min_test_r2=0.6),
            config=base_config(tmp_path),
            llm_client=MockLLMClient(),
        )


# -- classification ------------------------------------------------------------


@pytest.mark.slow
def test_a_classification_dataset_is_validated_on_class_metrics(tmp_path):
    report = run_agentic_workflow(
        dataset_path=CLASSIFICATION_CSV,
        criteria=AcceptanceCriteria(
            min_test_roc_auc=0.6, min_test_mcc=0.2, min_test_balanced_accuracy=0.55
        ),
        config=base_config(
            tmp_path, activity_column="active", split_method="stratified", max_iterations=2
        ),
        llm_client=MockLLMClient(),
    )
    assert report.task == "classification"
    assert report.best_validation.primary_metric == "roc_auc"
    assert "mcc" in report.best_validation.test_metrics
    assert "balanced_accuracy" in report.best_validation.test_metrics


# -- artifacts and reporting ---------------------------------------------------


@pytest.mark.slow
def test_artifacts_are_written_per_iteration(tmp_path):
    report = run_agentic_workflow(
        dataset_path=REGRESSION_CSV,
        criteria=AcceptanceCriteria(min_test_r2=0.5),
        config=base_config(tmp_path, max_iterations=2),
        llm_client=MockLLMClient(),
    )
    run_dir = Path(report.artifact_paths["run_dir"])
    assert (run_dir / "agentic_cleaned_dataset.csv").exists()
    assert (run_dir / "agentic_split_assignments.csv").exists()
    assert (run_dir / "agentic_run_report.json").exists()
    assert (run_dir / "agentic_run_report.md").exists()
    first = run_dir / "iteration_01"
    assert (first / "predictions.csv").exists()
    assert (first / "selected_features.json").exists()
    assert (first / "iteration_record.json").exists()


@pytest.mark.slow
def test_predictions_cover_train_and_test(tmp_path):
    report = run_agentic_workflow(
        dataset_path=REGRESSION_CSV,
        criteria=AcceptanceCriteria(min_test_r2=0.5),
        config=base_config(tmp_path, max_iterations=1),
        llm_client=MockLLMClient(),
    )
    predictions = pd.read_csv(
        Path(report.iterations[0].artifact_dir) / "predictions.csv"
    )
    assert set(predictions["split"]) == {"train", "test"}
    assert "residual" in predictions.columns


@pytest.mark.slow
def test_web_search_use_is_counted(tmp_path):
    report = run_agentic_workflow(
        dataset_path=REGRESSION_CSV,
        criteria=AcceptanceCriteria(min_test_r2=0.5),
        config=base_config(tmp_path, max_iterations=1),
        llm_client=MockLLMClient(),
    )
    # The mock exercises the search tool once per strategist call.
    assert report.web_searches >= 1


def test_the_rendered_report_states_failure_plainly():
    report = _synthetic_failed_report()
    rendered = render_report(report)
    assert "NOT ACCEPTED" in rendered
    assert "Honest assessment" in rendered
    assert "No threshold was relaxed" in rendered
    assert "Hold-out test R2 at least" in rendered


def test_the_rendered_report_lists_the_failing_criteria():
    rendered = render_report(_synthetic_failed_report())
    assert "observed 0.4000, required >= 0.9000" in rendered


def test_the_rendered_report_shows_the_iteration_history():
    rendered = render_report(_synthetic_failed_report())
    assert "## Iteration history" in rendered
    assert "rdkit_descriptors" in rendered


def _synthetic_failed_report() -> AgenticRunReport:
    from qsar_agent.schemas.agentic import (
        CVSummary,
        FeatureRecipe,
        IterationPlan,
        IterationRecord,
        ModelPlan,
        TaskDetection,
        ValidationReport,
    )

    plan = IterationPlan(
        features=FeatureRecipe(blocks=["rdkit_descriptors"]),
        model=ModelPlan(estimator="Ridge", hyperparameters={"alpha": 1.0}),
        rationale="Baseline.",
    )
    validation = ValidationReport(
        task="regression",
        primary_metric="r2",
        n_train=100,
        n_test=25,
        n_features=30,
        train_metrics={"r2": 0.55, "rmse": 0.7, "mae": 0.55},
        test_metrics={"r2": 0.40, "rmse": 0.9, "mae": 0.7},
        cv=CVSummary(
            folds=5,
            primary_metric="r2",
            mean_train_score=0.55,
            mean_cv_score=0.45,
            std_cv_score=0.05,
            train_cv_gap=0.10,
        ),
    )
    criteria = AcceptanceCriteria(min_test_r2=0.90)
    from qsar_agent.tools.acceptance import diagnose_failure, evaluate_acceptance

    verdict = evaluate_acceptance(criteria, validation)
    record = IterationRecord(
        index=1,
        plan=plan,
        validation=validation,
        verdict=verdict,
        diagnosis=diagnose_failure(validation, verdict),
    )
    return AgenticRunReport(
        run_id="testrun",
        dataset_path="test.csv",
        task="regression",
        task_detection=TaskDetection(task="regression", reason="continuous", n_unique_values=100),
        criteria=criteria.describe("regression"),
        n_compounds=125,
        n_train=100,
        n_test=25,
        split_method="random",
        iterations_run=1,
        max_iterations=3,
        accepted=False,
        stop_reason="Reached the maximum of 3 iterations without meeting the criteria.",
        best_iteration_index=1,
        best_validation=validation,
        best_verdict=verdict,
        best_plan=plan,
        iterations=[record],
    )


def test_the_rendered_report_celebrates_nothing_when_accepted():
    report = _synthetic_failed_report()
    report = report.model_copy(update={"accepted": True})
    rendered = render_report(report)
    assert "ACCEPTED" in rendered
    assert "Honest assessment" not in rendered


# -- workflow construction -----------------------------------------------------


def test_the_workflow_requires_an_llm_client(tmp_path, monkeypatch):
    from qsar_agent.llm.provider import LLMConfigurationError

    for name in ("OPENAI_API_KEY", "QSAR_LLM_API_KEY", "QSAR_LLM_PROVIDER"):
        monkeypatch.delenv(name, raising=False)
    with pytest.raises(LLMConfigurationError):
        AgenticQSARWorkflow(
            dataset_path=REGRESSION_CSV,
            criteria=AcceptanceCriteria(min_test_r2=0.6),
            config=base_config(tmp_path),
        )


def test_a_run_id_can_be_reused(tmp_path):
    workflow = AgenticQSARWorkflow(
        dataset_path=REGRESSION_CSV,
        criteria=AcceptanceCriteria(min_test_r2=0.6),
        config=base_config(tmp_path),
        llm_client=MockLLMClient(),
        run_id="fixedrun",
    )
    assert workflow.run_id == "fixedrun"
    assert workflow.run_dir.name == "fixedrun"


def test_progress_messages_are_forwarded(tmp_path):
    messages: list[str] = []
    workflow = AgenticQSARWorkflow(
        dataset_path=REGRESSION_CSV,
        criteria=AcceptanceCriteria(min_test_r2=0.6),
        config=base_config(tmp_path),
        llm_client=MockLLMClient(),
        progress=messages.append,
    )
    workflow.progress("hello")
    assert messages == ["hello"]
