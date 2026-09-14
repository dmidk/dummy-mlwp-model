"""A dummy deep-learning weather model: zarr in, zarr out."""

from __future__ import annotations


def _resolve_version() -> str:
    """Find the installed version, however the package was installed.

    Returns
    -------
    str
        The version derived from the git tag by hatch-vcs at build time; the
        distribution metadata when the generated file is absent; ``'0+unknown'`` when
        the package is not installed at all.
    """
    try:
        from ._version import __version__ as v

        return v
    except ImportError:
        pass

    from importlib.metadata import PackageNotFoundError, version

    try:
        return version("dummy-mlwp-model")
    except PackageNotFoundError:
        return "0+unknown"


__version__ = _resolve_version()
