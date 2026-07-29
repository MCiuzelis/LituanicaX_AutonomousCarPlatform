"""ROS-flavoured console logging: `[INFO] [time] [module]: message`."""

from __future__ import annotations

import logging
import sys
import time

_LEVEL_COLORS = {
    logging.DEBUG: "\033[36m",
    logging.INFO: "\033[32m",
    logging.WARNING: "\033[33m",
    logging.ERROR: "\033[31m",
    logging.CRITICAL: "\033[1;31m",
}
_RESET = "\033[0m"

_configured = False


class _RosStyleFormatter(logging.Formatter):
    def __init__(self, colorize: bool) -> None:
        super().__init__()
        self.colorize = colorize

    def format(self, record: logging.LogRecord) -> str:
        level = record.levelname
        if self.colorize:
            color = _LEVEL_COLORS.get(record.levelno, "")
            level = f"{color}{level}{_RESET}"
        name = record.name.split(".")[-1]
        return f"[{level}] [{time.time():.9f}] [{name}]: {record.getMessage()}"


def configure(level: int = logging.INFO, colorize: bool | None = None) -> None:
    """Install the console handler once, for the whole process."""
    global _configured
    if _configured:
        return

    if colorize is None:
        colorize = sys.stderr.isatty()

    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(_RosStyleFormatter(colorize))

    root = logging.getLogger("lituanicax")
    root.setLevel(level)
    root.addHandler(handler)
    root.propagate = False
    _configured = True


def get_logger(name: str) -> logging.Logger:
    configure()
    return logging.getLogger(f"lituanicax.{name}")
