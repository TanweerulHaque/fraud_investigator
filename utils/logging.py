"""Structured logging for agent tracing."""

from __future__ import annotations

import logging
import sys
from typing import Any

from utils.config import get_config


def setup_logging() -> None:
    cfg = get_config()
    level = getattr(logging, cfg.get("logging", {}).get("level", "INFO"))
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


def trace_agent(logger: logging.Logger, agent: str, payload: dict[str, Any]) -> None:
    cfg = get_config()
    if cfg.get("logging", {}).get("trace_agents", True):
        logger.info("[%s] %s", agent, payload)
