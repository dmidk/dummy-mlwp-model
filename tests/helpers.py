"""Plain helpers shared by test modules. Fixtures belong in conftest.py."""

from __future__ import annotations

from dummy_mlwp.__main__ import main


def invoke(monkeypatch, env: dict[str, str]) -> int:
    """Run the entrypoint with ``env`` set on top of the current environment.

    ``LOG_LEVEL`` is cleared first, so the developer's shell cannot change what the run
    logs.

    Returns
    -------
    int
        The exit code ``main()`` returns.
    """
    monkeypatch.delenv("LOG_LEVEL", raising=False)
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    return main()
