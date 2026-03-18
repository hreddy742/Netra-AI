"""
Structured JSON logging for all Netra services.
"""
from __future__ import annotations

import logging
import sys
from typing import Any

try:
    from loguru import logger as _loguru_logger
    _HAS_LOGURU = True
except ImportError:
    _HAS_LOGURU = False


def get_logger(name: str, level: str = "INFO") -> Any:
    if _HAS_LOGURU:
        _loguru_logger.remove()
        _loguru_logger.add(
            sys.stderr,
            level=level.upper(),
            format=(
                "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
                "<level>{level: <8}</level> | "
                "<cyan>{name}</cyan> | "
                "{message}"
            ),
            colorize=True,
            backtrace=True,
            diagnose=False,
        )
        return _loguru_logger.bind(service=name)
    else:
        logging.basicConfig(
            level=level.upper(),
            format="%(asctime)s %(levelname)-8s %(name)s %(message)s",
            handlers=[logging.StreamHandler(sys.stderr)],
        )
        return logging.getLogger(name)
