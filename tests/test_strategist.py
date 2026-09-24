"""Tests for the Strategist: schema validation, repair, and hard failure."""

from __future__ import annotations

import json

import pytest

from qsar_agent.agents.strategist import Strategist, StrategistError
from qsar_agent.llm.provider import LLMResponseError, LLMResult
from qsar_agent.schemas.agentic import IterationPlan


class ScriptedClient:
    """Returns queued replies, recording the prompts it was given."""

    provider = "scripted"
    model = "scripted"

    def __init__(self, replies):
        self._replies = list(replies)
        self.prompts: list[str] = []
        self.tools_offered: list[list[str]] = []

    def complete_json(self, system, user, schema_hint, tools=None, max_tool_steps=6):
        self.prompts.append(user)
        self.tools_offered.append([spec.name for spec in (tools or [])])
        reply = self._replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return LLMResult(data=reply, raw_text=json.dumps(reply))


def _rejection(strategist: Strategist, prompt_index: int) -> dict:
    """The structured repair feedback the strategist sent on a retry."""
    context = json.loads(strategist.client.prompts[prompt_index])
    return context["rejected_previous_reply"]


def build_strategist(replies, task="regression"):
    return Strategist(
        client=ScriptedClient(replies),
        task=task,
        history_provider=lambda: [],
        summary_provider=lambda: {"n_compounds": 100},
        criteria_provider=lambda: {"min_test_r2": 0.6},
    )


VALID_PLAN = {
    "features": {"blocks": ["rdkit_descriptors"]},
    "model": {"estimator": "Ridge", "hyperparameters": {"alpha": 1.0}},
    "rationale": "Simple baseline.",
}


# -- happy paths ---------------------------------------------------------------


def test_a_valid_initial_plan_is_accepted():
    strategist = build_strategist([VALID_PLAN])
    plan = strategist.initial_plan({"task": "regression"})
    assert isinstance(plan, IterationPlan)
    assert plan.model.estimator == "Ridge"
    assert plan.features.blocks == ["rdkit_descriptors"]


def test_a_valid_revision_is_accepted():
    strategist = build_strategist(
        [{"model": {"estimator": "RandomForest", "hyperparameters": {"n_estimators": 100}},
          "rationale": "More capacity."}]
    )
    revision = strategist.revise({"diagnosis": {"label": "underfit"}})
    assert revision.model.estimator == "RandomForest"
    assert revision.features is None


def test_a_revision_may_change_features_only():
    strategist = build_strategist([{"features": {"blocks": ["morgan_fp"]}, "rationale": "Bits."}])
    revision = strategist.revise({})
    assert revision.features.blocks == ["morgan_fp"]
    assert revision.model is None


def test_a_stop_revision_needs_no_changes():
    strategist = build_strategist([{"stop": True, "stop_reason": "Nothing left to try."}])
    revision = strategist.revise({})
    assert revision.stop
    assert revision.stop_reason


def test_revision_to_plan_keeps_the_unchanged_half():
    strategist = build_strategist([{"model": {"estimator": "PLS", "hyperparameters": {}}}])
    revision = strategist.revise({})
    previous = IterationPlan.model_validate(VALID_PLAN)
    plan = revision.to_plan(previous)
    assert plan.model.estimator == "PLS"
    assert plan.features.blocks == previous.features.blocks


def test_the_full_toolset_is_offered():
    strategist = build_strategist([VALID_PLAN])
    strategist.initial_plan({"task": "regression"})
    offered = strategist.client.tools_offered[0]
    assert {"web_search", "list_model_zoo", "list_feature_blocks", "get_iteration_history",
            "get_dataset_summary", "get_acceptance_criteria"} == set(offered)


def test_the_context_is_passed_as_json():
    strategist = build_strategist([VALID_PLAN])
    strategist.initial_plan({"task": "regression", "dataset": {"n_compounds": 42}})
    context = json.loads(strategist.client.prompts[0])
    assert context["request"] == "initial_plan"
    assert context["dataset"]["n_compounds"] == 42


# -- validation and repair -----------------------------------------------------


def test_an_unknown_estimator_is_rejected_then_repaired():
    strategist = build_strategist(
        [
            {"features": {"blocks": ["rdkit_descriptors"]},
             "model": {"estimator": "QuantumOracle", "hyperparameters": {}}},
            VALID_PLAN,
        ]
    )
    plan = strategist.initial_plan({"task": "regression"})
    assert plan.model.estimator == "Ridge"
    assert "QuantumOracle" in _rejection(strategist, 1)["error"]


def test_an_unknown_hyperparameter_is_rejected_then_repaired():
    strategist = build_strategist(
        [
            {"features": {"blocks": ["rdkit_descriptors"]},
             "model": {"estimator": "Ridge", "hyperparameters": {"telepathy": 3}}},
            VALID_PLAN,
        ]
    )
    plan = strategist.initial_plan({"task": "regression"})
    assert plan.model.hyperparameters == {"alpha": 1.0}
    assert "telepathy" in _rejection(strategist, 1)["error"]


def test_an_unknown_feature_block_is_rejected():
    strategist = build_strategist(
        [
            {"features": {"blocks": ["astrology"]},
             "model": {"estimator": "Ridge", "hyperparameters": {}}},
            VALID_PLAN,
        ]
    )
    assert strategist.initial_plan({"task": "regression"}).features.blocks == [
        "rdkit_descriptors"
    ]


def test_a_classification_estimator_is_rejected_for_a_regression_task():
    strategist = build_strategist(
        [
            {"features": {"blocks": ["rdkit_descriptors"]},
             "model": {"estimator": "LogisticRegression", "hyperparameters": {}}},
            VALID_PLAN,
        ],
        task="regression",
    )
    plan = strategist.initial_plan({"task": "regression"})
    assert plan.model.estimator == "Ridge"
    assert "LogisticRegression" in _rejection(strategist, 1)["error"]


def test_an_empty_revision_is_rejected():
    strategist = build_strategist(
        [
            {"rationale": "I changed nothing."},
            {"model": {"estimator": "RandomForest", "hyperparameters": {}}},
        ]
    )
    revision = strategist.revise({})
    assert revision.model.estimator == "RandomForest"
    assert "must change" in _rejection(strategist, 1)["error"]


def test_a_malformed_reply_is_returned_for_repair():
    strategist = build_strategist([LLMResponseError("no JSON here"), VALID_PLAN])
    plan = strategist.initial_plan({"task": "regression"})
    assert plan.model.estimator == "Ridge"


# -- hard failure --------------------------------------------------------------


def test_persistent_invalid_replies_raise_rather_than_falling_back():
    bad = {"features": {"blocks": ["rdkit_descriptors"]},
           "model": {"estimator": "NotReal", "hyperparameters": {}}}
    strategist = build_strategist([bad, bad, bad])
    with pytest.raises(StrategistError, match="could not produce a valid plan"):
        strategist.initial_plan({"task": "regression"})


def test_the_repair_budget_is_respected():
    bad = {"nonsense": True}
    client = ScriptedClient([bad] * 5)
    strategist = Strategist(
        client=client,
        task="regression",
        history_provider=lambda: [],
        summary_provider=lambda: {},
        criteria_provider=lambda: {},
        max_repair_attempts=1,
    )
    with pytest.raises(StrategistError):
        strategist.revise({})
    assert len(client.prompts) == 2


def test_the_final_error_names_the_last_problem():
    bad = {"features": {"blocks": ["rdkit_descriptors"]},
           "model": {"estimator": "Ridge", "hyperparameters": {"bogus_param": 1}}}
    strategist = build_strategist([bad, bad, bad])
    with pytest.raises(StrategistError, match="bogus_param"):
        strategist.initial_plan({"task": "regression"})
