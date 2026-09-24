"""Thin, swappable LLM client interface.

The agentic workflow has a single execution path and it requires a language
model. There is deliberately no heuristic fallback: if no credential is
configured, :func:`get_llm_client` raises :class:`LLMConfigurationError` before
any work starts.

Providers live in a small registry so a local model can be added later with one
factory function. An OpenAI-compatible server (vLLM, Ollama, OpenRouter, Azure)
needs no code change at all — point ``QSAR_LLM_BASE_URL`` at it.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol, runtime_checkable

from qsar_agent.logging_utils import get_logger

logger = get_logger()

DEFAULT_OPENAI_MODEL = "gpt-4o-mini"
DEFAULT_MAX_TOOL_STEPS = 6


class LLMConfigurationError(RuntimeError):
    """Raised when no usable LLM provider is configured."""


class LLMResponseError(RuntimeError):
    """Raised when a provider reply cannot be parsed as the requested JSON."""


@dataclass
class ToolSpec:
    """A deterministic function the model may call while reasoning."""

    name: str
    description: str
    parameters: dict[str, Any]
    handler: Callable[..., Any]

    def to_openai_schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


@dataclass
class ToolCallRecord:
    """Audit trail entry for one tool invocation."""

    name: str
    arguments: dict[str, Any]
    result_preview: str


@dataclass
class LLMResult:
    """Parsed JSON object plus the trail of how the model got there."""

    data: dict[str, Any]
    raw_text: str = ""
    tool_calls: list[ToolCallRecord] = field(default_factory=list)
    model: str = ""
    provider: str = ""

    def tool_names(self) -> list[str]:
        return [call.name for call in self.tool_calls]


@runtime_checkable
class LLMClient(Protocol):
    """Minimal contract every provider must satisfy."""

    provider: str
    model: str

    def complete_json(
        self,
        system: str,
        user: str,
        schema_hint: str,
        tools: list[ToolSpec] | None = None,
        max_tool_steps: int = DEFAULT_MAX_TOOL_STEPS,
    ) -> LLMResult:
        """Return a JSON object produced by the model."""
        ...


def strip_json_fence(text: str) -> str:
    """Remove markdown fences some models wrap JSON in."""
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped
    lines = [ln for ln in stripped.split("\n") if not ln.strip().startswith("```")]
    return "\n".join(lines).strip()


def parse_json_object(text: str) -> dict[str, Any]:
    """Parse a JSON object from model output, tolerating fences and prose."""
    candidate = strip_json_fence(text)
    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError:
        start = candidate.find("{")
        end = candidate.rfind("}")
        if start == -1 or end <= start:
            raise LLMResponseError(f"No JSON object found in model reply: {text[:400]!r}") from None
        parsed = json.loads(candidate[start : end + 1])
    if not isinstance(parsed, dict):
        raise LLMResponseError(f"Expected a JSON object, got {type(parsed).__name__}")
    return parsed


def _preview(value: Any, limit: int = 600) -> str:
    try:
        text = value if isinstance(value, str) else json.dumps(value, default=str)
    except (TypeError, ValueError):
        text = str(value)
    return text if len(text) <= limit else text[:limit] + "...[truncated]"


class OpenAIClient:
    """Production client for OpenAI and OpenAI-compatible endpoints."""

    def __init__(
        self,
        api_key: str,
        model: str = DEFAULT_OPENAI_MODEL,
        base_url: str | None = None,
        temperature: float = 0.0,
        provider: str = "openai",
    ):
        if not api_key:
            raise LLMConfigurationError("OpenAIClient requires a non-empty API key.")
        self.model = model
        self.provider = provider
        self.temperature = temperature
        self._base_url = base_url
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise LLMConfigurationError(
                "The 'openai' package is required for the OpenAI provider. "
                "Install it with: pip install openai"
            ) from exc
        kwargs: dict[str, Any] = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url
        self._client = OpenAI(**kwargs)

    def complete_json(
        self,
        system: str,
        user: str,
        schema_hint: str,
        tools: list[ToolSpec] | None = None,
        max_tool_steps: int = DEFAULT_MAX_TOOL_STEPS,
    ) -> LLMResult:
        tool_map = {spec.name: spec for spec in (tools or [])}
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": f"{system}\n\nRespond with JSON only.\n{schema_hint}"},
            {"role": "user", "content": user},
        ]
        calls: list[ToolCallRecord] = []

        for _step in range(max(1, max_tool_steps)):
            request: dict[str, Any] = {
                "model": self.model,
                "messages": messages,
                "temperature": self.temperature,
            }
            if tool_map:
                request["tools"] = [spec.to_openai_schema() for spec in tool_map.values()]
            response = self._client.chat.completions.create(**request)
            message = response.choices[0].message
            requested = getattr(message, "tool_calls", None) or []

            if not requested:
                raw = message.content or ""
                return LLMResult(
                    data=parse_json_object(raw),
                    raw_text=raw,
                    tool_calls=calls,
                    model=self.model,
                    provider=self.provider,
                )

            messages.append(
                {
                    "role": "assistant",
                    "content": message.content or "",
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments,
                            },
                        }
                        for tc in requested
                    ],
                }
            )
            for tool_call in requested:
                name = tool_call.function.name
                try:
                    arguments = json.loads(tool_call.function.arguments or "{}")
                except json.JSONDecodeError:
                    arguments = {}
                spec = tool_map.get(name)
                if spec is None:
                    output: Any = {"error": f"Unknown tool: {name}"}
                else:
                    try:
                        output = spec.handler(**arguments)
                    except Exception as exc:  # tool errors must not kill the run
                        output = {"error": f"{type(exc).__name__}: {exc}"}
                preview = _preview(output)
                calls.append(
                    ToolCallRecord(name=name, arguments=arguments, result_preview=preview)
                )
                messages.append(
                    {"role": "tool", "tool_call_id": tool_call.id, "content": preview}
                )

        raise LLMResponseError(
            f"Model kept requesting tools without answering after {max_tool_steps} steps."
        )


ProviderFactory = Callable[..., LLMClient]

_PROVIDERS: dict[str, ProviderFactory] = {}


def register_provider(name: str, factory: ProviderFactory) -> None:
    """Register an LLM provider factory under ``name``."""
    _PROVIDERS[name.strip().lower()] = factory


def available_providers() -> list[str]:
    return sorted(_PROVIDERS)


def _env(name: str) -> str | None:
    value = os.environ.get(name)
    if value is None:
        return None
    cleaned = value.strip().strip('"').strip("'")
    return cleaned or None


def resolve_model_name() -> str:
    return _env("QSAR_LLM_MODEL") or _env("OPENAI_MODEL") or DEFAULT_OPENAI_MODEL


def resolve_provider_name(explicit: str | None = None) -> str:
    return (explicit or _env("QSAR_LLM_PROVIDER") or "openai").strip().lower()


def _temperature() -> float:
    raw = _env("QSAR_LLM_TEMPERATURE")
    if raw is None:
        return 0.0
    try:
        return float(raw)
    except ValueError:
        logger.warning("Ignoring invalid QSAR_LLM_TEMPERATURE=%r; using 0.0", raw)
        return 0.0


def _openai_factory(model: str | None = None, **_: Any) -> LLMClient:
    api_key = _env("QSAR_LLM_API_KEY") or _env("OPENAI_API_KEY")
    base_url = _env("QSAR_LLM_BASE_URL")
    if not api_key:
        raise LLMConfigurationError(
            "No LLM API key found. The agentic QSAR workflow requires a language model.\n"
            "Set one of:\n"
            "  export OPENAI_API_KEY=sk-...            # OpenAI\n"
            "  export QSAR_LLM_API_KEY=...             # OpenAI-compatible endpoint\n"
            "Optional: QSAR_LLM_MODEL (default "
            f"{DEFAULT_OPENAI_MODEL}), QSAR_LLM_BASE_URL for a local or self-hosted server.\n"
            "For tests and offline demos only, pass --llm-provider mock."
        )
    return OpenAIClient(
        api_key=api_key,
        model=model or resolve_model_name(),
        base_url=base_url,
        temperature=_temperature(),
        provider="openai_compatible" if base_url else "openai",
    )


register_provider("openai", _openai_factory)
register_provider("openai_compatible", _openai_factory)


def get_llm_client(provider: str | None = None, model: str | None = None) -> LLMClient:
    """Build the configured LLM client, or raise with actionable instructions."""
    name = resolve_provider_name(provider)
    if name == "mock":
        # Imported lazily so the test double never loads in a production run.
        from qsar_agent.llm.mock import mock_provider_factory

        return mock_provider_factory(model=model)
    factory = _PROVIDERS.get(name)
    if factory is None:
        raise LLMConfigurationError(
            f"Unknown LLM provider {name!r}. Registered providers: "
            f"{', '.join([*available_providers(), 'mock'])}."
        )
    return factory(model=model)
