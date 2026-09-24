"""Tests for the LLM provider layer: fail-fast configuration and tool calling."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from qsar_agent.llm.mock import MockLLMClient, scripted_responder
from qsar_agent.llm.provider import (
    LLMConfigurationError,
    LLMResponseError,
    OpenAIClient,
    ToolSpec,
    get_llm_client,
    parse_json_object,
    register_provider,
    resolve_model_name,
    resolve_provider_name,
    strip_json_fence,
)

LLM_ENV_VARS = (
    "OPENAI_API_KEY",
    "QSAR_LLM_API_KEY",
    "QSAR_LLM_PROVIDER",
    "QSAR_LLM_MODEL",
    "OPENAI_MODEL",
    "QSAR_LLM_BASE_URL",
    "QSAR_LLM_TEMPERATURE",
)


@pytest.fixture
def clean_llm_env(monkeypatch):
    for name in LLM_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    return monkeypatch


# -- configuration -------------------------------------------------------------


def test_missing_api_key_fails_fast_with_instructions(clean_llm_env):
    with pytest.raises(LLMConfigurationError) as exc:
        get_llm_client()
    message = str(exc.value)
    assert "OPENAI_API_KEY" in message
    assert "requires a language model" in message


def test_the_error_points_at_the_mock_for_offline_use(clean_llm_env):
    with pytest.raises(LLMConfigurationError, match="--llm-provider mock"):
        get_llm_client()


def test_unknown_provider_lists_the_registered_ones(clean_llm_env):
    clean_llm_env.setenv("QSAR_LLM_PROVIDER", "telepathy")
    with pytest.raises(LLMConfigurationError, match="Registered providers"):
        get_llm_client()


def test_provider_defaults_to_openai(clean_llm_env):
    assert resolve_provider_name() == "openai"


def test_provider_can_be_set_by_environment_or_argument(clean_llm_env):
    clean_llm_env.setenv("QSAR_LLM_PROVIDER", "Mock")
    assert resolve_provider_name() == "mock"
    assert resolve_provider_name("openai") == "openai"


def test_model_name_resolution_prefers_the_qsar_variable(clean_llm_env):
    assert resolve_model_name() == "gpt-4o-mini"
    clean_llm_env.setenv("OPENAI_MODEL", "legacy-model")
    assert resolve_model_name() == "legacy-model"
    clean_llm_env.setenv("QSAR_LLM_MODEL", "preferred-model")
    assert resolve_model_name() == "preferred-model"


def test_quoted_environment_values_are_cleaned(clean_llm_env):
    clean_llm_env.setenv("QSAR_LLM_MODEL", '"quoted-model"')
    assert resolve_model_name() == "quoted-model"


def test_a_custom_provider_can_be_registered(clean_llm_env):
    sentinel = SimpleNamespace(provider="local", model="llama")
    register_provider("test_local", lambda model=None, **_: sentinel)
    clean_llm_env.setenv("QSAR_LLM_PROVIDER", "test_local")
    assert get_llm_client() is sentinel


def test_mock_provider_requires_explicit_opt_in(clean_llm_env):
    client = get_llm_client(provider="mock")
    assert isinstance(client, MockLLMClient)
    assert client.provider == "mock"


def test_openai_client_rejects_an_empty_key():
    with pytest.raises(LLMConfigurationError, match="non-empty API key"):
        OpenAIClient(api_key="")


# -- JSON parsing --------------------------------------------------------------


def test_fenced_json_is_unwrapped():
    assert strip_json_fence('```json\n{"a": 1}\n```') == '{"a": 1}'


def test_plain_json_parses():
    assert parse_json_object('{"a": 1}') == {"a": 1}


def test_json_embedded_in_prose_is_recovered():
    assert parse_json_object('Sure, here it is: {"a": 1} Hope that helps!') == {"a": 1}


def test_unparseable_output_raises():
    with pytest.raises(LLMResponseError, match="No JSON object"):
        parse_json_object("I would rather not.")


def test_a_json_array_is_not_an_object():
    with pytest.raises(LLMResponseError, match="Expected a JSON object"):
        parse_json_object("[1, 2, 3]")


# -- OpenAI tool-calling loop --------------------------------------------------


class FakeCompletions:
    """Replays a scripted sequence of OpenAI responses and records requests."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.requests = []

    def create(self, **kwargs):
        self.requests.append(kwargs)
        return self._responses.pop(0)


def make_response(content=None, tool_calls=None):
    message = SimpleNamespace(content=content, tool_calls=tool_calls)
    return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def make_tool_call(call_id, name, arguments):
    return SimpleNamespace(
        id=call_id,
        function=SimpleNamespace(name=name, arguments=json.dumps(arguments)),
    )


def build_client(monkeypatch, responses):
    client = OpenAIClient(api_key="test-key", model="test-model")
    fake = FakeCompletions(responses)
    monkeypatch.setattr(
        client, "_client", SimpleNamespace(chat=SimpleNamespace(completions=fake))
    )
    return client, fake


def test_a_direct_answer_is_returned(monkeypatch):
    client, fake = build_client(monkeypatch, [make_response(content='{"answer": 42}')])
    result = client.complete_json("sys", "user", "hint")
    assert result.data == {"answer": 42}
    assert result.tool_calls == []
    assert result.model == "test-model"
    assert len(fake.requests) == 1


def test_a_requested_tool_is_executed_and_fed_back(monkeypatch):
    calls = []

    def handler(query: str, max_results: int = 5):
        calls.append((query, max_results))
        return {"results": ["a paper"]}

    tool = ToolSpec(
        name="web_search",
        description="search",
        parameters={"type": "object", "properties": {}},
        handler=handler,
    )
    client, fake = build_client(
        monkeypatch,
        [
            make_response(
                tool_calls=[make_tool_call("c1", "web_search", {"query": "qsar", "max_results": 2})]
            ),
            make_response(content='{"done": true}'),
        ],
    )
    result = client.complete_json("sys", "user", "hint", tools=[tool])
    assert result.data == {"done": True}
    assert calls == [("qsar", 2)]
    assert result.tool_names() == ["web_search"]
    # The tool result must reach the model on the follow-up request.
    assert any(m["role"] == "tool" for m in fake.requests[1]["messages"])


def test_a_failing_tool_reports_the_error_instead_of_crashing(monkeypatch):
    def handler():
        raise RuntimeError("network down")

    tool = ToolSpec(
        name="broken",
        description="always fails",
        parameters={"type": "object", "properties": {}},
        handler=handler,
    )
    client, _fake = build_client(
        monkeypatch,
        [
            make_response(tool_calls=[make_tool_call("c1", "broken", {})]),
            make_response(content='{"ok": true}'),
        ],
    )
    result = client.complete_json("sys", "user", "hint", tools=[tool])
    assert result.data == {"ok": True}
    assert "network down" in result.tool_calls[0].result_preview


def test_an_unknown_tool_request_is_answered_with_an_error(monkeypatch):
    client, _fake = build_client(
        monkeypatch,
        [
            make_response(tool_calls=[make_tool_call("c1", "nonexistent", {})]),
            make_response(content='{"ok": true}'),
        ],
    )
    result = client.complete_json("sys", "user", "hint", tools=[])
    assert "Unknown tool" in result.tool_calls[0].result_preview


def test_endless_tool_requests_are_cut_off(monkeypatch):
    tool = ToolSpec(
        name="loop",
        description="loops",
        parameters={"type": "object", "properties": {}},
        handler=lambda: {"more": True},
    )
    responses = [make_response(tool_calls=[make_tool_call(f"c{i}", "loop", {})]) for i in range(5)]
    client, _fake = build_client(monkeypatch, responses)
    with pytest.raises(LLMResponseError, match="without answering"):
        client.complete_json("sys", "user", "hint", tools=[tool], max_tool_steps=3)


def test_tool_schemas_are_sent_in_openai_format(monkeypatch):
    tool = ToolSpec(
        name="thing",
        description="does a thing",
        parameters={"type": "object", "properties": {"a": {"type": "string"}}},
        handler=lambda a="": {},
    )
    client, fake = build_client(monkeypatch, [make_response(content="{}")])
    client.complete_json("sys", "user", "hint", tools=[tool])
    sent = fake.requests[0]["tools"][0]
    assert sent["type"] == "function"
    assert sent["function"]["name"] == "thing"


# -- mock client ---------------------------------------------------------------


def test_mock_client_requires_a_json_context():
    with pytest.raises(LLMResponseError, match="JSON context"):
        MockLLMClient().complete_json("sys", "not json", "hint")


def test_mock_client_records_the_contexts_it_saw():
    client = MockLLMClient()
    client.complete_json("sys", json.dumps({"request": "initial_plan", "task": "regression"}), "h")
    assert client.calls[0]["request"] == "initial_plan"


def test_mock_client_can_use_an_injected_responder():
    client = MockLLMClient(responder=lambda context: {"stop": True, "stop_reason": "scripted"})
    result = client.complete_json("sys", json.dumps({"request": "revision"}), "hint")
    assert result.data == {"stop": True, "stop_reason": "scripted"}


def test_mock_initial_plan_starts_simple():
    plan = scripted_responder({"request": "initial_plan", "task": "regression"})
    assert plan["model"]["estimator"] == "Ridge"
    assert plan["features"]["blocks"] == ["rdkit_descriptors"]


def test_mock_initial_plan_adapts_to_classification():
    plan = scripted_responder({"request": "initial_plan", "task": "classification"})
    assert plan["model"]["estimator"] == "LogisticRegression"


def test_mock_revision_targets_the_reported_diagnosis():
    revision = scripted_responder(
        {"request": "revision", "diagnosis": {"label": "underfit"}, "tried_signatures": []}
    )
    assert revision["addresses_diagnosis"] == "underfit"
    assert "morgan_fp" in revision["features"]["blocks"]


def test_mock_revision_avoids_repeating_a_tried_configuration():
    context = {"request": "revision", "diagnosis": {"label": "underfit"}, "tried_signatures": []}
    first = scripted_responder(context)
    signature = _signature_of(first)
    second = scripted_responder({**context, "tried_signatures": [signature]})
    assert _signature_of(second) != signature


def test_mock_eventually_stops_when_the_ladder_is_exhausted():
    from qsar_agent.llm.mock import _GENERIC_LADDER, _REVISION_LADDER

    context = {"request": "revision", "diagnosis": {"label": "underfit"}, "tried_signatures": []}
    tried = []
    for _ in range(len(_REVISION_LADDER["underfit"]) + len(_GENERIC_LADDER)):
        revision = scripted_responder({**context, "tried_signatures": tried})
        if revision.get("stop"):
            break
        tried.append(_signature_of(revision))
    final = scripted_responder({**context, "tried_signatures": tried})
    assert final["stop"] is True
    assert final["stop_reason"]


def _signature_of(revision: dict) -> str:
    from qsar_agent.schemas.agentic import FeatureRecipe, IterationPlan, ModelPlan

    return IterationPlan(
        features=FeatureRecipe.model_validate(revision["features"]),
        model=ModelPlan.model_validate(revision["model"]),
    ).signature()
