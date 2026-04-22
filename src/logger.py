"""Centralized logging utilities for the ALPR project."""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any, Optional


DEFAULT_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
DEFAULT_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


class BraceStyleLogger:
    """Small adapter that supports both logging `%s` and Loguru-style `{}` messages.

    The project originally used Loguru-style calls such as:
    `logger.info("Loaded model from {}", path)`.
    This adapter keeps those calls working while using Python's built-in
    `logging` module underneath.
    """

    def __init__(self, wrapped: logging.Logger) -> None:
        self._logger = wrapped

    def _format(self, message: Any, *args: Any) -> str:
        text = str(message)
        if args:
            try:
                if "{}" in text:
                    return text.format(*args)
                return text % args
            except Exception:
                return " ".join([text, *[str(arg) for arg in args]])
        return text

    def debug(self, message: Any, *args: Any, **kwargs: Any) -> None:
        self._logger.debug(self._format(message, *args), **kwargs)

    def info(self, message: Any, *args: Any, **kwargs: Any) -> None:
        self._logger.info(self._format(message, *args), **kwargs)

    def warning(self, message: Any, *args: Any, **kwargs: Any) -> None:
        self._logger.warning(self._format(message, *args), **kwargs)

    def error(self, message: Any, *args: Any, **kwargs: Any) -> None:
        self._logger.error(self._format(message, *args), **kwargs)

    def exception(self, message: Any, *args: Any, **kwargs: Any) -> None:
        self._logger.exception(self._format(message, *args), **kwargs)

    def critical(self, message: Any, *args: Any, **kwargs: Any) -> None:
        self._logger.critical(self._format(message, *args), **kwargs)

    def setLevel(self, level: int | str) -> None:
        self._logger.setLevel(level)

    def remove(self) -> None:
        """Compatibility helper for old Loguru-style `logger.remove()` calls."""

        for handler in list(self._logger.handlers):
            self._logger.removeHandler(handler)
            handler.close()

    def add(self, sink: Any, level: int | str = logging.INFO, **_: Any) -> None:
        """Compatibility helper for old Loguru-style `logger.add(...)` calls."""

        formatter = logging.Formatter(DEFAULT_LOG_FORMAT, datefmt=DEFAULT_DATE_FORMAT)
        if hasattr(sink, "write"):
            handler: logging.Handler = logging.StreamHandler(sink)
        else:
            path = Path(sink)
            path.parent.mkdir(parents=True, exist_ok=True)
            handler = logging.FileHandler(path, encoding="utf-8")
        handler.setLevel(level)
        handler.setFormatter(formatter)
        self._logger.addHandler(handler)


def _coerce_level(level: int | str) -> int:
    """Convert string log levels to logging constants."""

    if isinstance(level, int):
        return level
    value = getattr(logging, str(level).upper(), None)
    if not isinstance(value, int):
        raise ValueError(f"Invalid log level: {level}")
    return value


def get_logger(
    name: str = "alpr",
    level: int | str = logging.INFO,
    log_file: Optional[str | Path] = None,
    console: bool = True,
) -> logging.Logger:
    """Initialize and return a standard Python logger.

    The function is safe to call multiple times. Existing handlers are reused
    unless the requested file handler is missing.
    """

    logger = logging.getLogger(name)
    logger.setLevel(_coerce_level(level))
    logger.propagate = False

    formatter = logging.Formatter(DEFAULT_LOG_FORMAT, datefmt=DEFAULT_DATE_FORMAT)

    if console and not any(getattr(handler, "_alpr_console", False) for handler in logger.handlers):
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(_coerce_level(level))
        console_handler.setFormatter(formatter)
        console_handler._alpr_console = True  # type: ignore[attr-defined]
        logger.addHandler(console_handler)

    if log_file is not None:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        resolved = str(log_path.resolve())
        has_file_handler = any(getattr(handler, "_alpr_log_file", None) == resolved for handler in logger.handlers)
        if not has_file_handler:
            file_handler = logging.FileHandler(log_path, encoding="utf-8")
            file_handler.setLevel(_coerce_level(level))
            file_handler.setFormatter(formatter)
            file_handler._alpr_log_file = resolved  # type: ignore[attr-defined]
            logger.addHandler(file_handler)

    for handler in logger.handlers:
        handler.setLevel(_coerce_level(level))

    return logger


def setup_logger(
    log_name: str = "alpr.log",
    level: int | str = logging.INFO,
    log_dir: str | Path | None = None,
) -> BraceStyleLogger:
    """Project convenience initializer with console and file logging."""

    if log_dir is None:
        log_dir = Path(__file__).resolve().parents[1] / "outputs" / "logs"
    log_path = Path(log_dir) / log_name
    base_logger = get_logger("alpr", level=level, log_file=log_path, console=True)
    return BraceStyleLogger(base_logger)


logger = setup_logger()


__all__ = ["get_logger", "setup_logger", "logger", "BraceStyleLogger"]
