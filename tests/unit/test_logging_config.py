"""Tests for logging_config."""

from __future__ import annotations

import logging
import os

import pytest

from src.ontodisco.utils import logging_config as lc


@pytest.fixture(autouse=True)
def reset_logging_state(monkeypatch):
    """Isolate env and allow re-configuration between tests."""
    monkeypatch.delenv(lc.ENV_VAR, raising=False)
    monkeypatch.delenv(lc.LIB_ENV_VAR, raising=False)
    lc._configured = False
    logging.getLogger().handlers.clear()
    yield
    lc._configured = False
    logging.getLogger().handlers.clear()


def test_app_debug_does_not_enable_pymongo_debug(monkeypatch, caplog):
    monkeypatch.setenv(lc.ENV_VAR, "DEBUG")
    lc.configure_logging(force=True)

    app_logger = logging.getLogger("src.ontodisco.utils.dedup_base")
    pymongo_logger = logging.getLogger("pymongo.topology")

    assert app_logger.getEffectiveLevel() == logging.DEBUG
    assert pymongo_logger.getEffectiveLevel() == logging.WARNING

    with caplog.at_level(logging.DEBUG):
        app_logger.debug("app debug message")
        pymongo_logger.debug("pymongo heartbeat")

    assert "app debug message" in caplog.text
    assert "pymongo heartbeat" not in caplog.text


def test_lib_log_level_override(monkeypatch):
    monkeypatch.setenv(lc.ENV_VAR, "INFO")
    monkeypatch.setenv(lc.LIB_ENV_VAR, "DEBUG")
    lc.configure_logging(force=True)

    assert logging.getLogger("pymongo").getEffectiveLevel() == logging.DEBUG
    assert logging.getLogger("src.ontodisco").getEffectiveLevel() == logging.INFO


def test_get_logger_uses_project_namespace():
    lc.configure_logging(force=True)
    logger = lc.get_logger("OpenAIUtils")
    assert logger.name == "src.ontodisco.OpenAIUtils"
