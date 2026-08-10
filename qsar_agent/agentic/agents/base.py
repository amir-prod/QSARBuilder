"""Shared helpers for structured agent calls with repair + deterministic fallback."""

from __future__ import annotations

from typing import Any, Callable, TypeVar

from pydantic import BaseModel

from qsar_agent.agentic.provider import AgentProvider, ProviderError

T = TypeVar("T", bound=BaseModel)


def call_agent_structured(
    provider: AgentProvider | None,
    *,
    agent_name: str,
    system_prompt: str,
    payload: dict[str, Any],
    response_model: type[T],
    deterministic_fallback: Callable[[], T] | None = None,
) -> tuple[T, str]:
    """Return (response, decision_source)."""
    if provider is None:
        if deterministic_fallback is None:
            raise ProviderError(f"No provider and no fallback for {agent_name}")
        return deterministic_fallback(), "deterministic_fallback"

    try:
        result = provider.get_structured_response(
            agent_name=agent_name,
            system_prompt=system_prompt,
            payload=payload,
            response_model=response_model,
        )
        return result, "llm_agent"
    except Exception:
        if deterministic_fallback is not None:
            return deterministic_fallback(), "deterministic_fallback"
        raise
