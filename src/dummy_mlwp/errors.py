"""Error types, each mapped to a distinct process exit code.

This application stands in for a real model while pipelines are built around it, so
failures must be unambiguous: the exit code says which layer rejected the run, and the
message says exactly what to change.
"""

from __future__ import annotations


class DummyMLWPError(Exception):
    """Base class for every error this application raises deliberately."""

    exit_code = 1


class ConfigError(DummyMLWPError):
    """An environment variable is missing, malformed, or internally inconsistent."""

    exit_code = 2


class InputError(DummyMLWPError):
    """The input store does not match what the configuration says to expect."""

    exit_code = 3


class DeviceError(DummyMLWPError):
    """The requested compute device is unavailable or unusable."""

    exit_code = 4


def format_problems(headline: str, problems: list[str]) -> str:
    """Render a collected list of problems as a single multi-line message."""
    return "\n".join([headline] + [f"  - {p}" for p in problems])
