"""ALPR package."""

from __future__ import annotations

import sys

__version__ = "1.0.0"


def _check_python_version() -> None:
    """Fail fast on unsupported interpreters for the ML stack."""

    version = sys.version_info
    if version < (3, 10) or version >= (3, 13):
        raise RuntimeError(
            "ALPR Pro requires Python 3.10, 3.11, or 3.12. "
            f"Detected Python {version.major}.{version.minor}. "
            "Use the provided Docker setup or create a Python 3.11 virtual environment."
        )


_check_python_version()
