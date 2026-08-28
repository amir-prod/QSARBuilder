"""Deterministic experiment identifiers."""

from __future__ import annotations

import hashlib
import json
from typing import Any


def canonical_json(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, default=str, separators=(",", ":"))


def make_experiment_id(
    *,
    tool_name: str,
    arguments: dict[str, Any],
    dataset_hash: str,
    development_split_hash: str,
    parent_experiment_id: str | None = None,
) -> str:
    """Stable 16-char SHA-256 prefix of the canonical experiment key."""
    key = {
        "tool_name": tool_name,
        "arguments": arguments or {},
        "dataset_hash": dataset_hash,
        "development_split_hash": development_split_hash,
        "parent_experiment_id": parent_experiment_id or "",
    }
    digest = hashlib.sha256(canonical_json(key).encode("utf-8")).hexdigest()
    return digest[:16]
