"""A dummy deep-learning weather model: zarr in, zarr out."""

from __future__ import annotations


def _resolve_version() -> str:
    # Written by hatch-vcs at build time; present in any installed copy.
    try:
        from ._version import __version__ as v

        return v
    except ImportError:
        pass
    # Installed without the generated file (e.g. an editable install from a shallow
    # clone): fall back to the distribution metadata.
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version("dummy-mlwp-model")
    except PackageNotFoundError:
        return "0+unknown"


__version__ = _resolve_version()
