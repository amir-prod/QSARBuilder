"""Tests for .env loading."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from qsar_agent.config import _project_root, get_openai_api_key, get_openai_model, load_env_file

_ENV_PATH = _project_root() / ".env"


@pytest.mark.skipif(not _ENV_PATH.exists(), reason="No .env file in project root")
def test_load_env_file_from_project_root():
    loaded = load_env_file()
    assert loaded is True


@pytest.mark.skipif(not _ENV_PATH.exists(), reason="No .env file in project root")
def test_openai_key_available_after_load_env():
    load_env_file()
    key = get_openai_api_key()
    assert key is not None
    assert key.startswith("sk-")


@pytest.mark.skipif(not _ENV_PATH.exists(), reason="No .env file in project root")
def test_openai_model_from_env():
    load_env_file()
    model = get_openai_model()
    assert model
    assert model == os.environ.get("OPENAI_MODEL", model).strip().strip('"').strip("'")
