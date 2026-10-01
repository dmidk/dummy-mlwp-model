"""Entrypoint: read zarr, run the dummy model, write zarr.

Configuration comes entirely from the environment — there is no command line, so the
container behaves identically however it is launched.
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np
from loguru import logger

from . import __version__
from .config import Config
from .errors import DummyMLWPError
from .grid import detect_coords
from .inputs import (
    detect_zarr_format,
    open_input,
    select_input_timesteps,
    stack_channels,
    validate_input,
)
from .model import (
    DummyNet,
    channel_stats,
    feedback_index,
    output_stats,
    predict,
    select_device,
)
from .outputs import apply_output_mode, build_output_dataset, write_output
from .timeaxis import build_output_times
from .varspec import channel_layout

LOG_FORMAT = (
    "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | "
    "<cyan>{name}</cyan> - <level>{message}</level>"
)


def configure_logging(level: str) -> None:
    """Point loguru at stderr.

    Parameters
    ----------
    level : str
        A loguru level name. An unrecognised level warns and falls back to ``INFO``,
        rather than failing a run over a typo in a log setting.
    """
    logger.remove()
    try:
        logger.add(sys.stderr, level=level, format=LOG_FORMAT, colorize=sys.stderr.isatty())
    except ValueError:
        logger.add(sys.stderr, level="INFO", format=LOG_FORMAT, colorize=sys.stderr.isatty())
        logger.warning(f"LOG_LEVEL={level!r} is not a known level; using INFO")


def describe_config(config: Config) -> dict[str, str]:
    """Render the effective configuration for the startup log.

    Parameters
    ----------
    config : Config
        The parsed run configuration.

    Returns
    -------
    dict of str to str
        One entry per setting the run will use, defaults included, keyed by the
        environment variable that controls it, in the order the README documents them.
        Values are written in the grammar the variables accept (variable specs and
        level coordinates as declared, durations in ISO 8601), so a log line maps
        straight back to the deployment. An optional setting left unset reads
        ``unset (...)``, naming what happens instead.

    Notes
    -----
    The storage options are deliberately left out. They are assembled from many
    variables rather than one, and :func:`storage.storage_options` has already logged
    them, credentials masked, while the configuration was parsed. One place rendering
    them means one place that has to get the redaction right.
    """
    timestep = config.forecast_timestep
    return {
        "INPUT_ZARR": config.input_zarr,
        "OUTPUT_ZARR": config.output_zarr,
        "INPUT_VARIABLES": ",".join(str(spec) for spec in config.input_variables),
        "OUTPUT_VARIABLES": ",".join(str(spec) for spec in config.output_variables),
        "LEVEL_COORDS": _render_level_coords(config.level_coords),
        "N_INPUT_TIMESTEPS": _or_unset(config.n_input_timesteps, "all"),
        "N_FORECAST_STEPS": str(config.n_forecast_steps),
        "FORECAST_TIMESTEP": _or_unset(
            None if timestep is None else timestep.isoformat(), "inferred from input"
        ),
        "TIME_COORD": _or_unset(config.time_coord, "auto-detect"),
        "X_COORD": _or_unset(config.x_coord, "auto-detect"),
        "Y_COORD": _or_unset(config.y_coord, "auto-detect"),
        "OUTPUT_MODE": config.output_mode,
        "RANDOM_SEED": str(config.random_seed),
        "CONSTANT_VALUE": str(config.constant_value),
        "DEVICE": config.device,
        "MODEL_HIDDEN_CHANNELS": str(config.model_hidden_channels),
        "MODEL_LAYERS": str(config.model_layers),
        "ZARR_FORMAT": config.zarr_format,
        "LOG_LEVEL": config.log_level,
    }


def _render_level_coords(level_coords: dict[str, np.ndarray]) -> str:
    """Write level coordinates back in the ``LEVEL_COORDS`` grammar.

    Parameters
    ----------
    level_coords : dict of str to numpy.ndarray
        Declared level coordinates, keyed by name.

    Returns
    -------
    str
        ``name:v1/v2,name2:v1`` in declaration order, e.g.
        ``'isobaricInhPa:850/500/250'``, or ``'unset (none)'`` when none are declared.
    """
    if not level_coords:
        return "unset (none)"
    return ",".join(
        f"{name}:{'/'.join(str(v) for v in values.tolist())}"
        for name, values in level_coords.items()
    )


def _or_unset(value: object, meaning: str) -> str:
    """Render an optional setting, spelling out what an unset one means.

    Parameters
    ----------
    value : object
        The setting's value, or ``None`` when it was left unset.
    meaning : str
        What the run does instead when the setting is unset.

    Returns
    -------
    str
        ``str(value)``, or ``'unset (<meaning>)'`` for ``None``. The parenthesised form
        cannot be mistaken for a value someone actually configured.
    """
    return f"unset ({meaning})" if value is None else str(value)


def log_config(config: Config) -> None:
    """Log the effective configuration at INFO, one setting per line.

    Parameters
    ----------
    config : Config
        The parsed run configuration, rendered by :func:`describe_config`.

    Notes
    -----
    One record per setting rather than a single multi-line message: container log
    collectors split on newlines, so the continuation lines of one message would lose
    their timestamp and level, while separate records each keep theirs and stay
    greppable by variable name.
    """
    settings = describe_config(config)
    width = max(len(name) for name in settings)
    logger.info("Effective configuration (defaults included):")
    for name, value in settings.items():
        logger.info(f"  {name:<{width}} = {value}")


def run(config: Config) -> None:
    """Execute one forecast, start to finish.

    Parameters
    ----------
    config : Config
        The parsed run configuration.

    Raises
    ------
    InputError
        If the input store does not match the configuration.
    DeviceError
        If the requested device is unusable.
    ConfigError
        If the forecast resolution cannot be determined from the input.
    """
    started = time.perf_counter()

    ds = open_input(config.input_zarr, config.src_storage_options)
    coords = detect_coords(ds, config.time_coord, config.y_coord, config.x_coord)
    validate_input(ds, config, coords)
    ds = select_input_timesteps(ds, coords, config.n_input_timesteps)

    input_fields = stack_channels(ds, config.input_variables, config, coords)
    times, lead_times, reference_time = build_output_times(
        ds[coords.time].values, config.n_forecast_steps, config.forecast_timestep
    )

    in_layout = channel_layout(config.input_variables, config.level_coords)
    out_layout = channel_layout(config.output_variables, config.level_coords)
    in_mean, in_std = channel_stats(input_fields)
    out_mean, out_std = output_stats(in_layout, out_layout, in_mean, in_std)

    device = select_device(config.device)
    net = DummyNet(
        in_channels=len(in_layout),
        out_channels=len(out_layout),
        hidden_channels=config.model_hidden_channels,
        n_layers=config.model_layers,
        seed=config.random_seed,
    )
    predicted = predict(
        input_fields,
        net,
        device,
        config.n_forecast_steps,
        in_mean,
        in_std,
        out_mean,
        out_std,
        feedback_index(in_layout, out_layout),
    )

    fields = apply_output_mode(config, predicted, input_fields, in_layout, out_layout)
    out_ds = build_output_dataset(
        config, coords, ds, fields, times, lead_times, reference_time, device.type
    )

    zarr_format = _resolve_zarr_format(config)
    write_output(out_ds, config, zarr_format, coords)

    logger.info(f"Done in {time.perf_counter() - started:.2f} s")


def _resolve_zarr_format(config: Config) -> int:
    """Decide which zarr format to write.

    Parameters
    ----------
    config : Config
        The run configuration.

    Returns
    -------
    {2, 3}
        The explicit ZARR_FORMAT when set, otherwise the input store's own format.
        Falls back to 3, with a warning, when detection fails.
    """
    if config.zarr_format != "auto":
        return int(config.zarr_format)
    detected = detect_zarr_format(config.input_zarr, config.src_storage_options)
    if detected is None:
        logger.warning("Could not detect the input's zarr format; writing format 3")
        return 3
    logger.info(f"Input store is zarr format {detected}; writing the output in the same format")
    return detected


def main() -> int:
    """Run the application from the environment and return a process exit code.

    Returns
    -------
    int
        0 on success; 2 for a configuration error, 3 for an input error, 4 for a
        device error, and 1 for anything unexpected, whose traceback is logged.

    Notes
    -----
    The configuration is parsed before any store is touched, so a bad deployment
    fails in the first second rather than after a long read.

    With no command line and no config file, the log is the only record of what a run
    used: the version is logged first, before parsing, so even a run that fails on its
    configuration says which version failed, and the effective configuration follows
    as soon as it parses.
    """
    configure_logging(os.environ.get("LOG_LEVEL", "INFO").strip().upper() or "INFO")
    logger.info(f"Starting dummy-mlwp-model {__version__}")
    try:
        config = Config.from_env()
    except DummyMLWPError as exc:
        logger.error(f"{exc}")
        return exc.exit_code

    configure_logging(config.log_level)
    log_config(config)
    try:
        run(config)
    except DummyMLWPError as exc:
        logger.error(f"{exc}")
        return exc.exit_code
    except Exception:
        logger.opt(exception=True).error("Unexpected failure")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
