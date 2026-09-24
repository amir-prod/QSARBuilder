"""Web search tool exposed to the reasoning agents.

Tries Tavily when ``TAVILY_API_KEY`` is set, then the DuckDuckGo HTML endpoint.
It never raises: a sandboxed or offline run gets an empty result set plus a note
explaining why, so the agent keeps reasoning from what it already knows instead
of crashing the workflow.
"""

from __future__ import annotations

import html
import os
import re
from urllib.parse import unquote

from pydantic import BaseModel, Field

from qsar_agent.logging_utils import get_logger

logger = get_logger()

DEFAULT_TIMEOUT = 10
DEFAULT_MAX_RESULTS = 5
DUCKDUCKGO_ENDPOINT = "https://html.duckduckgo.com/html/"
TAVILY_ENDPOINT = "https://api.tavily.com/search"


class SearchResult(BaseModel):
    title: str
    url: str
    snippet: str = ""


class WebSearchResponse(BaseModel):
    query: str
    provider: str
    results: list[SearchResult] = Field(default_factory=list)
    note: str = ""

    def as_tool_output(self) -> dict:
        return self.model_dump()


def _env(name: str) -> str | None:
    value = os.environ.get(name)
    if value is None:
        return None
    cleaned = value.strip().strip('"').strip("'")
    return cleaned or None


def _tavily_search(query: str, max_results: int, timeout: int) -> WebSearchResponse | None:
    api_key = _env("TAVILY_API_KEY")
    if not api_key:
        return None
    import requests

    response = requests.post(
        TAVILY_ENDPOINT,
        json={
            "api_key": api_key,
            "query": query,
            "max_results": max_results,
            "search_depth": "basic",
        },
        timeout=timeout,
    )
    response.raise_for_status()
    payload = response.json()
    results = [
        SearchResult(
            title=str(item.get("title", ""))[:300],
            url=str(item.get("url", "")),
            snippet=str(item.get("content", ""))[:600],
        )
        for item in payload.get("results", [])[:max_results]
    ]
    return WebSearchResponse(query=query, provider="tavily", results=results)


_RESULT_PATTERN = re.compile(
    r'<a[^>]*class="result__a"[^>]*href="(?P<href>[^"]+)"[^>]*>(?P<title>.*?)</a>',
    re.DOTALL,
)
_SNIPPET_PATTERN = re.compile(
    r'<a[^>]*class="result__snippet"[^>]*>(?P<snippet>.*?)</a>', re.DOTALL
)
_TAG_PATTERN = re.compile(r"<[^>]+>")


def _strip_html(text: str) -> str:
    return html.unescape(_TAG_PATTERN.sub("", text)).strip()


def _clean_duckduckgo_url(href: str) -> str:
    # DuckDuckGo wraps results as /l/?uddg=<urlencoded target>
    match = re.search(r"uddg=([^&]+)", href)
    return unquote(match.group(1)) if match else href


def _duckduckgo_search(query: str, max_results: int, timeout: int) -> WebSearchResponse:
    import requests

    response = requests.post(
        DUCKDUCKGO_ENDPOINT,
        data={"q": query},
        headers={"User-Agent": "Mozilla/5.0 (compatible; qsar-agent/1.0)"},
        timeout=timeout,
    )
    response.raise_for_status()
    body = response.text
    titles = _RESULT_PATTERN.findall(body)
    snippets = [_strip_html(s) for s in _SNIPPET_PATTERN.findall(body)]
    results = []
    for index, (href, title) in enumerate(titles[:max_results]):
        results.append(
            SearchResult(
                title=_strip_html(title)[:300],
                url=_clean_duckduckgo_url(href),
                snippet=snippets[index][:600] if index < len(snippets) else "",
            )
        )
    return WebSearchResponse(query=query, provider="duckduckgo", results=results)


def web_search(
    query: str,
    max_results: int = DEFAULT_MAX_RESULTS,
    timeout: int = DEFAULT_TIMEOUT,
) -> WebSearchResponse:
    """Search the web for ``query``, degrading to an empty result set when offline."""
    query = (query or "").strip()
    if not query:
        return WebSearchResponse(query="", provider="none", note="Empty query; nothing searched.")

    max_results = max(1, min(int(max_results), 10))
    try:
        tavily = _tavily_search(query, max_results, timeout)
        if tavily is not None:
            return tavily
    except Exception as exc:
        logger.warning("Tavily search failed (%s); trying DuckDuckGo.", exc)

    try:
        return _duckduckgo_search(query, max_results, timeout)
    except Exception as exc:
        note = (
            f"Web search unavailable ({type(exc).__name__}: {exc}). "
            "Continue reasoning from the information already provided."
        )
        logger.warning("Web search unavailable: %s", exc)
        return WebSearchResponse(query=query, provider="unavailable", note=note)
