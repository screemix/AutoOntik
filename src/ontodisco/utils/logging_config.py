"""Central logging configuration.

``ONTODISCO_LOG_LEVEL`` controls verbosity for project code under
``src.ontodisco`` only. Third-party libraries are capped at WARNING by default
so DEBUG on your code does not flood the console with pymongo/urllib3 noise.

To debug a library, set ``ONTODISCO_LIB_LOG_LEVEL=DEBUG`` (applies to the
known noisy loggers listed in ``_QUIET_LOGGER_NAMES``).
"""

from __future__ import annotations

import logging
import os
from typing import Optional

ENV_VAR = "ONTODISCO_LOG_LEVEL"
LIB_ENV_VAR = "ONTODISCO_LIB_LOG_LEVEL"
DEFAULT_LEVEL_NAME = "INFO"
_VALID_LEVELS = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}

APP_LOGGER_ROOT = "src.ontodisco"

# Loggers that become very noisy when the root level is DEBUG.
_QUIET_LOGGER_NAMES = (
    "pymongo",
    "urllib3",
    "httpcore",
    "httpx",
    "huggingface_hub",
    "transformers",
    "filelock",
    "matplotlib",
    "fsspec",
    "asyncio",
    "openai",
    "charset_normalizer",
)

_configured = False


def resolve_log_level_name(level_name: Optional[str] = None) -> str:
    name = (level_name or os.getenv(ENV_VAR) or DEFAULT_LEVEL_NAME).upper()
    if name not in _VALID_LEVELS:
        return DEFAULT_LEVEL_NAME
    return name


def get_log_level(level_name: Optional[str] = None) -> int:
    return getattr(logging, resolve_log_level_name(level_name))


def _resolve_lib_log_level() -> int:
    lib_name = os.getenv(LIB_ENV_VAR)
    if lib_name and lib_name.upper() in _VALID_LEVELS:
        return getattr(logging, lib_name.upper())
    return logging.WARNING


def configure_logging(level_name: Optional[str] = None, *, force: bool = False) -> None:
    """Apply process-wide logging once (idempotent unless force=True)."""
    global _configured
    if _configured and not force:
        return

    app_level = get_log_level(level_name)
    lib_level = _resolve_lib_log_level()

    logging.basicConfig(
        level=app_level,
        format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
        datefmt="%H:%M:%S",
        force=True,
    )

    logging.getLogger(APP_LOGGER_ROOT).setLevel(app_level)

    for name in _QUIET_LOGGER_NAMES:
        logging.getLogger(name).setLevel(lib_level)

    _configured = True


def get_logger(name: str) -> logging.Logger:
    """Return a logger under the ``src.ontodisco`` namespace."""
    configure_logging()
    if name.startswith(f"{APP_LOGGER_ROOT}.") or name == APP_LOGGER_ROOT:
        return logging.getLogger(name)
    return logging.getLogger(f"{APP_LOGGER_ROOT}.{name}")
