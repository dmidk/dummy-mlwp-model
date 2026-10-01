"""Environment-variable configuration.

Every knob this application has lives here. Parsing happens once, up front, before the
input store is even opened — a bad deployment should fail in the first second, not after
a long read.

Problems are collected, not raised eagerly: every variable is checked, and a single
ConfigError (exit code 2) lists all of them, so a deployment with three bad variables is
fixed in one round trip rather than three. The individual parse helpers still raise
ConfigError; :meth:`Config.from_env` runs each one through :func:`_attempt`, which
records the message and carries on. A check that depends on a value which failed to
parse is skipped rather than run against a placeholder, so one mistake is reported once,
not echoed by every check downstream of it.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, TypeVar

import numpy as np
import pandas as pd

from .errors import ConfigError, format_problems
from .storage import storage_options
from .varspec import VarSpec, channel_layout, parse_level_coords, parse_var_specs

OUTPUT_MODES = ("random", "persistence", "constant", "zeros")
DEVICES = ("auto", "cuda", "cpu")
ZARR_FORMATS = ("auto", "2", "3")

_T = TypeVar("_T")


@dataclass(frozen=True)
class Config:
    """The fully parsed, validated run configuration.

    Parameters
    ----------
    input_zarr, output_zarr : str
        Store URIs. Local paths, ``s3://`` and ``gs://`` all work.
    input_variables, output_variables : list of VarSpec
        Variables to expect on input and to write on output.
    level_coords : dict of str to numpy.ndarray, optional
        Declared level coordinates, keyed by name.
    src_storage_options, dst_storage_options : dict, optional
        fsspec options for the input and output stores, kept separate so the two can
        live on different object stores or accounts.
    n_input_timesteps : int or None, optional
        How many of the input's timesteps to feed the model: ``None`` for all, a
        positive ``n`` for the first ``n``, a negative ``n`` for the last ``|n|``.
    n_forecast_steps : int, optional
        ``-1`` to predict on the input timesteps, or a positive number of forecast
        steps at the input's time resolution.
    forecast_timestep : pandas.Timedelta or None, optional
        Explicit forecast resolution, needed only when the input has a single timestep.
    time_coord, x_coord, y_coord : str or None, optional
        Coordinate name overrides; ``None`` means detect from the input.
    output_mode : {'random', 'persistence', 'constant', 'zeros'}, optional
        What the output arrays contain. The forward pass runs regardless.
    random_seed : int, optional
        Seeds the network weights, making runs reproducible.
    constant_value : float, optional
        Fill value used by ``constant`` mode.
    device : {'auto', 'cuda', 'cpu'}, optional
        Compute device. ``'cuda'`` fails if no device is visible.
    model_hidden_channels, model_layers : int, optional
        Network width and depth — how much work the GPU is given.
    zarr_format : {'auto', '2', '3'}, optional
        Output store format. ``'auto'`` matches the input.
    log_level : str, optional
        loguru level name.
    """

    input_zarr: str
    output_zarr: str
    input_variables: list[VarSpec]
    output_variables: list[VarSpec]
    level_coords: dict[str, np.ndarray] = field(default_factory=dict)

    src_storage_options: dict[str, Any] = field(default_factory=dict)
    dst_storage_options: dict[str, Any] = field(default_factory=dict)

    n_input_timesteps: int | None = None
    n_forecast_steps: int = -1
    forecast_timestep: pd.Timedelta | None = None

    time_coord: str | None = None
    x_coord: str | None = None
    y_coord: str | None = None

    output_mode: str = "random"
    random_seed: int = 0
    constant_value: float = 0.0

    device: str = "auto"
    model_hidden_channels: int = 64
    model_layers: int = 4

    zarr_format: str = "auto"
    log_level: str = "INFO"

    @property
    def n_input_channels(self) -> int:
        """int: Number of 2D fields the network consumes."""
        return len(channel_layout(self.input_variables, self.level_coords))

    @property
    def n_output_channels(self) -> int:
        """int: Number of 2D fields the network produces."""
        return len(channel_layout(self.output_variables, self.level_coords))

    @property
    def predicts_input_timesteps(self) -> bool:
        """bool: True when N_FORECAST_STEPS is -1, i.e. one prediction per input step."""
        return self.n_forecast_steps == -1

    def provenance(self) -> dict[str, str]:
        """Summarise the configuration for the output store's attributes.

        Returns
        -------
        dict of str to str
            Key configuration values, recorded so a stored result can be traced back
            to the run that produced it.
        """
        return {
            "input_variables": ",".join(str(s) for s in self.input_variables),
            "output_variables": ",".join(str(s) for s in self.output_variables),
            "n_input_timesteps": "all"
            if self.n_input_timesteps is None
            else str(self.n_input_timesteps),
            "n_forecast_steps": str(self.n_forecast_steps),
            "output_mode": self.output_mode,
            "random_seed": str(self.random_seed),
            "source": self.input_zarr,
        }

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> Config:
        """Build a configuration from environment variables.

        Parameters
        ----------
        env : mapping of str to str, optional
            Environment to read. Defaults to :data:`os.environ`; tests pass a dict.

        Returns
        -------
        Config
            The parsed, validated configuration.

        Raises
        ------
        ConfigError
            If a required variable is missing or any value is malformed or out of
            range. Every variable is checked first, so the message lists all the
            problems at once; a lone problem is reported as its bare message. Every
            check here runs before the input store is opened.

        Notes
        -----
        A check that depends on another value is skipped when that value is itself
        invalid, since its problem has already been recorded:

        * A broken LEVEL_COORDS does not stop INPUT_VARIABLES and OUTPUT_VARIABLES
          from being parsed, but their ``@`` references are not checked against it —
          otherwise every reference would be reported as undeclared.
        * A range check (``N_FORECAST_STEPS``, ``MODEL_LAYERS``, ...) runs only when
          the value parsed as a number.
        * A side's storage options are parsed only when its store URI is set, because
          the URI's scheme decides which options apply.

        Each variable reports at most one problem: the parsers stop at the first thing
        wrong with a value.
        """
        env = os.environ if env is None else env
        problems: list[str] = []

        level_coords = _attempt(problems, lambda: parse_level_coords(env.get("LEVEL_COORDS", "")))
        # When LEVEL_COORDS failed, level_coords is None, which parse_var_specs reads as
        # "declared set unknown" and so skips only the @reference check.
        input_variables = _attempt(
            problems,
            lambda: parse_var_specs(
                _require(env, "INPUT_VARIABLES"), level_coords, "INPUT_VARIABLES"
            ),
        )
        output_variables = _attempt(
            problems,
            lambda: parse_var_specs(
                _require(env, "OUTPUT_VARIABLES"), level_coords, "OUTPUT_VARIABLES"
            ),
        )

        n_input_timesteps = _attempt(problems, lambda: _get_optional_int(env, "N_INPUT_TIMESTEPS"))
        if n_input_timesteps == 0:
            problems.append(
                "N_INPUT_TIMESTEPS must be a positive number (the first n timesteps) or "
                "a negative number (the last |n|); leave it unset to use all of them"
            )

        n_forecast_steps = _attempt(problems, lambda: _get_int(env, "N_FORECAST_STEPS", -1))
        if n_forecast_steps is not None and (n_forecast_steps == 0 or n_forecast_steps < -1):
            problems.append(
                f"N_FORECAST_STEPS must be -1 (predict on the input timesteps) or a "
                f"positive number of forecast steps, got {n_forecast_steps}"
            )

        forecast_timestep = _attempt(problems, lambda: _get_timedelta(env, "FORECAST_TIMESTEP"))
        if forecast_timestep is not None and forecast_timestep <= pd.Timedelta(0):
            problems.append(f"FORECAST_TIMESTEP must be positive, got {forecast_timestep}")

        model_hidden_channels = _attempt(
            problems, lambda: _get_int(env, "MODEL_HIDDEN_CHANNELS", 64)
        )
        if model_hidden_channels is not None and model_hidden_channels < 1:
            problems.append(f"MODEL_HIDDEN_CHANNELS must be >= 1, got {model_hidden_channels}")
        model_layers = _attempt(problems, lambda: _get_int(env, "MODEL_LAYERS", 4))
        if model_layers is not None and model_layers < 2:
            problems.append(
                f"MODEL_LAYERS must be >= 2 (an input and an output convolution), "
                f"got {model_layers}"
            )

        output_mode = _attempt(
            problems, lambda: _get_choice(env, "OUTPUT_MODE", OUTPUT_MODES, "random")
        )
        random_seed = _attempt(problems, lambda: _get_int(env, "RANDOM_SEED", 0))
        constant_value = _attempt(problems, lambda: _get_float(env, "CONSTANT_VALUE", 0.0))
        device = _attempt(problems, lambda: _get_choice(env, "DEVICE", DEVICES, "auto"))
        zarr_format = _attempt(
            problems, lambda: _get_choice(env, "ZARR_FORMAT", ZARR_FORMATS, "auto")
        )

        input_zarr = _attempt(problems, lambda: _require(env, "INPUT_ZARR"))
        output_zarr = _attempt(problems, lambda: _require(env, "OUTPUT_ZARR"))
        # The URI's scheme decides which storage options apply, so a side whose URI is
        # missing (already reported above) has nothing meaningful to be parsed against.
        src_storage_options = (
            None
            if input_zarr is None
            else _attempt(problems, lambda: storage_options(env, input_zarr, "SRC"))
        )
        dst_storage_options = (
            None
            if output_zarr is None
            else _attempt(problems, lambda: storage_options(env, output_zarr, "DST"))
        )

        if len(problems) == 1:
            raise ConfigError(problems[0])
        if problems:
            raise ConfigError(
                format_problems(
                    f"The environment configuration has {len(problems)} problems:", problems
                )
            )

        return cls(
            input_zarr=input_zarr,
            output_zarr=output_zarr,
            input_variables=input_variables,
            output_variables=output_variables,
            level_coords=level_coords,
            src_storage_options=src_storage_options,
            dst_storage_options=dst_storage_options,
            n_input_timesteps=n_input_timesteps,
            n_forecast_steps=n_forecast_steps,
            forecast_timestep=forecast_timestep,
            time_coord=_get_optional(env, "TIME_COORD"),
            x_coord=_get_optional(env, "X_COORD"),
            y_coord=_get_optional(env, "Y_COORD"),
            output_mode=output_mode,
            random_seed=random_seed,
            constant_value=constant_value,
            device=device,
            model_hidden_channels=model_hidden_channels,
            model_layers=model_layers,
            zarr_format=zarr_format,
            log_level=env.get("LOG_LEVEL", "INFO").strip().upper() or "INFO",
        )


def _attempt(problems: list[str], parse: Callable[[], _T]) -> _T | None:
    """Run one parse step, recording its failure instead of raising it.

    Parameters
    ----------
    problems : list of str
        Problems collected so far. A failing step appends its message here.
    parse : callable
        A zero-argument callable that returns the parsed value or raises
        :class:`ConfigError`.

    Returns
    -------
    object or None
        Whatever ``parse`` returned, or ``None`` if it raised :class:`ConfigError`.
        ``None`` is only a placeholder: :meth:`Config.from_env` raises before it can
        reach a :class:`Config`, and checks that depend on the value skip it.

    Notes
    -----
    Only :class:`ConfigError` is caught. Anything else is a bug rather than a
    configuration problem, and should surface as one.
    """
    try:
        return parse()
    except ConfigError as exc:
        problems.append(str(exc))
        return None


def _require(env: Mapping[str, str], key: str) -> str:
    """Read a required environment variable.

    Parameters
    ----------
    env : mapping of str to str
        Environment to read.
    key : str
        Variable name.

    Returns
    -------
    str
        The stripped value.

    Raises
    ------
    ConfigError
        If the variable is unset or blank.
    """
    value = env.get(key, "").strip()
    if not value:
        raise ConfigError(f"{key} is required but not set")
    return value


def _get_optional(env: Mapping[str, str], key: str) -> str | None:
    """Read an optional environment variable.

    Parameters
    ----------
    env : mapping of str to str
        Environment to read.
    key : str
        Variable name.

    Returns
    -------
    str or None
        The stripped value, or ``None`` when unset or blank.
    """
    value = env.get(key, "").strip()
    return value or None


def _get_int(env: Mapping[str, str], key: str, default: int) -> int:
    """Read an integer environment variable.

    Parameters
    ----------
    env : mapping of str to str
        Environment to read.
    key : str
        Variable name.
    default : int
        Value used when the variable is unset.

    Returns
    -------
    int
        The parsed value.

    Raises
    ------
    ConfigError
        If the value is not an integer.
    """
    raw = _get_optional(env, key)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ConfigError(f"{key} must be an integer, got {raw!r}") from exc


def _get_optional_int(env: Mapping[str, str], key: str) -> int | None:
    """Read an integer environment variable that has no default.

    Parameters
    ----------
    env : mapping of str to str
        Environment to read.
    key : str
        Variable name.

    Returns
    -------
    int or None
        The parsed value, or ``None`` when unset — which the caller is free to read as
        "no limit" rather than as a number.

    Raises
    ------
    ConfigError
        If the value is not an integer.
    """
    raw = _get_optional(env, key)
    if raw is None:
        return None
    try:
        return int(raw)
    except ValueError as exc:
        raise ConfigError(f"{key} must be an integer, got {raw!r}") from exc


def _get_float(env: Mapping[str, str], key: str, default: float) -> float:
    """Read a floating-point environment variable.

    Parameters
    ----------
    env : mapping of str to str
        Environment to read.
    key : str
        Variable name.
    default : float
        Value used when the variable is unset.

    Returns
    -------
    float
        The parsed value.

    Raises
    ------
    ConfigError
        If the value is not a number.
    """
    raw = _get_optional(env, key)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ConfigError(f"{key} must be a number, got {raw!r}") from exc


def _get_choice(env: Mapping[str, str], key: str, choices: tuple[str, ...], default: str) -> str:
    """Read an environment variable constrained to a fixed set of values.

    Matching is case-insensitive, so ``DEVICE=CPU`` and ``DEVICE=cpu`` agree.

    Parameters
    ----------
    env : mapping of str to str
        Environment to read.
    key : str
        Variable name.
    choices : tuple of str
        Permitted values, lowercase.
    default : str
        Value used when the variable is unset.

    Returns
    -------
    str
        The lowercased, validated value.

    Raises
    ------
    ConfigError
        If the value is not one of ``choices``.
    """
    raw = _get_optional(env, key)
    if raw is None:
        return default
    value = raw.lower()
    if value not in choices:
        raise ConfigError(f"{key} must be one of {', '.join(choices)}, got {raw!r}")
    return value


def _get_timedelta(env: Mapping[str, str], key: str) -> pd.Timedelta | None:
    """Read a duration environment variable.

    Parameters
    ----------
    env : mapping of str to str
        Environment to read.
    key : str
        Variable name.

    Returns
    -------
    pandas.Timedelta or None
        The parsed duration, or ``None`` when unset.

    Raises
    ------
    ConfigError
        If the value is not a duration pandas recognises.

    Notes
    -----
    pandas accepts ISO 8601 durations (``'PT6H'``) as well as friendlier forms
    (``'6h'``), so both are supported without an extra dependency.
    """
    raw = _get_optional(env, key)
    if raw is None:
        return None
    try:
        return pd.Timedelta(raw)
    except ValueError as exc:
        raise ConfigError(
            f"{key} must be an ISO 8601 duration such as 'PT6H' (or '6h'), got {raw!r}"
        ) from exc
