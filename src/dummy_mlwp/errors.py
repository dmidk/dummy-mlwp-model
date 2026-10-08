"""Error types, each mapped to a distinct process exit code.

This application stands in for a real model while pipelines are built around it, so
failures must be unambiguous: the exit code says which layer rejected the run, and the
message says exactly what to change.

==== ===================================================================================
Code Meaning
==== ===================================================================================
1    Anything unexpected: not one of these classes, so a bug, logged with its traceback
2    :class:`ConfigError` — the environment is wrong
3    :class:`InputError` — the input store is missing or does not match the configuration
4    :class:`DeviceError` — the requested compute device is unusable
5    :class:`StorageError` — a store could not be reached, read or written
==== ===================================================================================
"""

from __future__ import annotations


class DummyMLWPError(Exception):
    """Base class for every error this application raises deliberately.

    Attributes
    ----------
    exit_code : int
        Process exit code used when this error reaches the entrypoint.
    """

    exit_code = 1


class ConfigError(DummyMLWPError):
    """An environment variable is missing, malformed, or internally inconsistent.

    Attributes
    ----------
    exit_code : int
        Always 2.
    """

    exit_code = 2


class InputError(DummyMLWPError):
    """The input store does not match what the configuration says to expect.

    Attributes
    ----------
    exit_code : int
        Always 3.
    """

    exit_code = 3


class DeviceError(DummyMLWPError):
    """The requested compute device is unavailable or unusable.

    Attributes
    ----------
    exit_code : int
        Always 4.
    """

    exit_code = 4


class StorageError(DummyMLWPError):
    """A store could not be reached, read or written.

    This is the layer below the data — missing or rejected credentials, access denied,
    an unreachable endpoint, a destination that cannot be written — so it says nothing
    about the store's contents. An input store that simply does not exist stays an
    :class:`InputError`, since that is a wrong ``INPUT_ZARR`` rather than a failure to
    reach the store.

    Attributes
    ----------
    exit_code : int
        Always 5.
    """

    exit_code = 5


def format_problems(headline: str, problems: list[str]) -> str:
    """Render a collected list of problems as a single multi-line message.

    Parameters
    ----------
    headline : str
        Opening line describing what was being checked.
    problems : list of str
        Individual problems, rendered as an indented bullet list beneath the headline.

    Returns
    -------
    str
        The headline followed by one indented line per problem.

    Examples
    --------
    >>> print(format_problems("Two problems:", ["first", "second"]))
    Two problems:
      - first
      - second
    """
    return "\n".join([headline] + [f"  - {p}" for p in problems])
