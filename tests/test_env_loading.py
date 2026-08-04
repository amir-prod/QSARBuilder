"""Tests for .env loading."""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from qsar_agent.config import (
    _project_root,
    get_openai_api_key,
    get_openai_api_key_source,
    get_openai_model,
    load_env_file,
)

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


def test_ui_api_key_used_when_env_missing():
    import sys

    mock_st = MagicMock()
    mock_st.secrets.get.return_value = None
    mock_st.session_state = {"openai_api_key": "sk-ui-test-key"}
    env_backup = os.environ.pop("OPENAI_API_KEY", None)
    try:
        with patch.dict(sys.modules, {"streamlit": mock_st}):
            key, source = get_openai_api_key_source()
        assert key == "sk-ui-test-key"
        assert source == "ui"
    finally:
        if env_backup is not None:
            os.environ["OPENAI_API_KEY"] = env_backup


def test_environment_api_key_wins_over_ui():
    with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-env-test-key"}, clear=False):
        key, source = get_openai_api_key_source()
    assert key == "sk-env-test-key"
    assert source == "environment"