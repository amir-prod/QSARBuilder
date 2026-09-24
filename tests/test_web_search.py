"""Tests for the web search tool, including its offline behaviour."""

from __future__ import annotations

import pytest

from qsar_agent.tools import web_search as web_search_module
from qsar_agent.tools.web_search import web_search

DUCKDUCKGO_HTML = """
<html><body>
<a class="result__a" href="/l/?uddg=https%3A%2F%2Fexample.org%2Fqsar">QSAR <b>review</b></a>
<a class="result__snippet">Descriptors for <b>solubility</b> models.</a>
<a class="result__a" href="https://direct.example.com/paper">Direct link</a>
<a class="result__snippet">Second snippet.</a>
</body></html>
"""


class FakeResponse:
    def __init__(self, text="", payload=None, status_ok=True):
        self.text = text
        self._payload = payload
        self._status_ok = status_ok

    def raise_for_status(self):
        if not self._status_ok:
            raise RuntimeError("HTTP 500")

    def json(self):
        return self._payload


@pytest.fixture
def no_search_keys(monkeypatch):
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    return monkeypatch


def test_an_empty_query_searches_nothing(no_search_keys):
    response = web_search("   ")
    assert response.provider == "none"
    assert response.results == []
    assert "Empty query" in response.note


def test_network_failure_returns_an_explanatory_note(no_search_keys, monkeypatch):
    def explode(*_args, **_kwargs):
        raise OSError("Name or service not known")

    monkeypatch.setattr(web_search_module, "_duckduckgo_search", explode)
    response = web_search("qsar descriptors")
    assert response.provider == "unavailable"
    assert response.results == []
    assert "Name or service not known" in response.note
    assert "Continue reasoning" in response.note


def test_offline_search_never_raises(no_search_keys, monkeypatch):
    monkeypatch.setattr(
        web_search_module,
        "_duckduckgo_search",
        lambda *a, **k: (_ for _ in ()).throw(ConnectionError("no route to host")),
    )
    assert web_search("anything").results == []


def test_duckduckgo_results_are_parsed(no_search_keys, monkeypatch):
    import requests

    monkeypatch.setattr(requests, "post", lambda *a, **k: FakeResponse(text=DUCKDUCKGO_HTML))
    response = web_search("qsar", max_results=5)
    assert response.provider == "duckduckgo"
    assert len(response.results) == 2
    first = response.results[0]
    assert first.title == "QSAR review"
    assert first.url == "https://example.org/qsar"
    assert "solubility" in first.snippet


def test_html_tags_are_stripped_from_titles(no_search_keys, monkeypatch):
    import requests

    monkeypatch.setattr(requests, "post", lambda *a, **k: FakeResponse(text=DUCKDUCKGO_HTML))
    assert "<b>" not in web_search("qsar").results[0].title


def test_direct_urls_pass_through_unchanged(no_search_keys, monkeypatch):
    import requests

    monkeypatch.setattr(requests, "post", lambda *a, **k: FakeResponse(text=DUCKDUCKGO_HTML))
    assert web_search("qsar").results[1].url == "https://direct.example.com/paper"


def test_result_count_is_capped(no_search_keys, monkeypatch):
    import requests

    monkeypatch.setattr(requests, "post", lambda *a, **k: FakeResponse(text=DUCKDUCKGO_HTML))
    assert len(web_search("qsar", max_results=1).results) == 1


def test_tavily_is_preferred_when_a_key_is_present(monkeypatch):
    monkeypatch.setenv("TAVILY_API_KEY", "test-key")
    import requests

    payload = {
        "results": [
            {"title": "Tavily hit", "url": "https://example.com/a", "content": "Some content."}
        ]
    }
    monkeypatch.setattr(requests, "post", lambda *a, **k: FakeResponse(payload=payload))
    response = web_search("qsar")
    assert response.provider == "tavily"
    assert response.results[0].title == "Tavily hit"


def test_tavily_failure_falls_back_to_duckduckgo(monkeypatch):
    monkeypatch.setenv("TAVILY_API_KEY", "test-key")
    monkeypatch.setattr(
        web_search_module,
        "_tavily_search",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("quota exceeded")),
    )
    monkeypatch.setattr(
        web_search_module,
        "_duckduckgo_search",
        lambda query, max_results, timeout: web_search_module.WebSearchResponse(
            query=query, provider="duckduckgo"
        ),
    )
    assert web_search("qsar").provider == "duckduckgo"


def test_tool_output_is_json_serialisable(no_search_keys, monkeypatch):
    import json

    monkeypatch.setattr(
        web_search_module,
        "_duckduckgo_search",
        lambda *a, **k: (_ for _ in ()).throw(OSError("offline")),
    )
    json.dumps(web_search("qsar").as_tool_output())
