"""LLM provider interface for structured agent responses."""

from __future__ import annotations

import json
import logging
from typing import Any, Callable, Protocol, TypeVar

from pydantic import BaseModel, ValidationError

from qsar_agent.config import get_openai_api_key, get_openai_model

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


class AgentProvider(Protocol):
    def get_structured_response(
        self,
        *,
        agent_name: str,
        system_prompt: str,
        payload: dict[str, Any],
        response_model: type[T],
    ) -> T: ...


class ProviderError(RuntimeError):
    pass


class OpenAIAgentProvider:
    """OpenAI Chat Completions with JSON schema validation + one repair attempt."""

    def __init__(
        self,
        *,
        model: str | None = None,
        api_key: str | None = None,
        timeout: float = 60.0,
        max_retries: int = 1,
    ) -> None:
        self.model = model or get_openai_model()
        self.api_key = api_key if api_key is not None else get_openai_api_key()
        self.timeout = timeout
        self.max_retries = max_retries
        self.last_token_usage: dict[str, Any] | None = None
        self.last_error: str | None = None

    def get_structured_response(
        self,
        *,
        agent_name: str,
        system_prompt: str,
        payload: dict[str, Any],
        response_model: type[T],
    ) -> T:
        if not self.api_key:
            raise ProviderError("No OpenAI API key available")

        from openai import OpenAI

        client = OpenAI(api_key=self.api_key, timeout=self.timeout)
        user_content = json.dumps(payload, default=str)
        schema_hint = (
            f"Respond with JSON matching this schema: {response_model.model_json_schema()}"
        )
        messages = [
            {"role": "system", "content": system_prompt + "\n\n" + schema_hint},
            {"role": "user", "content": user_content},
        ]

        last_err: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                if attempt > 0 and last_err is not None:
                    messages = messages + [
                        {
                            "role": "user",
                            "content": (
                                "Previous response failed validation: "
                                f"{last_err}. Return corrected JSON only."
                            ),
                        }
                    ]
                resp = client.chat.completions.create(
                    model=self.model,
                    temperature=0,
                    response_format={"type": "json_object"},
                    messages=messages,
                )
                if resp.usage:
                    self.last_token_usage = {
                        "prompt_tokens": resp.usage.prompt_tokens,
                        "completion_tokens": resp.usage.completion_tokens,
                        "total_tokens": resp.usage.total_tokens,
                        "agent_name": agent_name,
                        "model": self.model,
                    }
                content = resp.choices[0].message.content or "{}"
                data = json.loads(content)
                return response_model.model_validate(data)
            except (ValidationError, json.JSONDecodeError, Exception) as exc:
                last_err = exc
                self.last_error = str(exc)
                logger.warning("Agent %s structured response attempt %s failed: %s", agent_name, attempt, exc)

        raise ProviderError(
            f"Structured response validation failed for {agent_name}: {last_err}"
        )


class MockAgentProvider:
    """Deterministic mock provider for tests."""

    def __init__(
        self,
        handlers: dict[str, Callable[[dict[str, Any]], BaseModel]] | None = None,
        default_factory: Callable[[str, type[BaseModel], dict[str, Any]], BaseModel] | None = None,
    ) -> None:
        self.handlers = handlers or {}
        self.default_factory = default_factory
        self.calls: list[dict[str, Any]] = []
        self.last_token_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    def get_structured_response(
        self,
        *,
        agent_name: str,
        system_prompt: str,
        payload: dict[str, Any],
        response_model: type[T],
    ) -> T:
        self.calls.append(
            {
                "agent_name": agent_name,
                "system_prompt": system_prompt,
                "payload": payload,
                "response_model": response_model.__name__,
            }
        )
        if agent_name in self.handlers:
            result = self.handlers[agent_name](payload)
            return response_model.model_validate(result.model_dump() if isinstance(result, BaseModel) else result)
        if self.default_factory is not None:
            result = self.default_factory(agent_name, response_model, payload)
            return response_model.model_validate(result.model_dump() if isinstance(result, BaseModel) else result)
        raise ProviderError(f"No mock handler for agent {agent_name}")


def get_provider(
    *,
    model: str | None = None,
    use_mock: bool = False,
    mock: MockAgentProvider | None = None,
) -> AgentProvider:
    if use_mock or mock is not None:
        return mock or MockAgentProvider()
    key = get_openai_api_key()
    if not key:
        raise ProviderError("No OpenAI API key available")
    return OpenAIAgentProvider(model=model, api_key=key)
