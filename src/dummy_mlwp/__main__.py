"""Entrypoint: read zarr, run the dummy model, write zarr.

Configuration comes entirely from the environment — there is no command line, so the
container behaves identically however it is launched.
"""

from __future__ import annotations

import os
import sys
import time

from loguru import logger

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
    StorageError
        If the input store cannot be read or the output store cannot be written.
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
        device error, 5 for a storage error (a store could not be reached, read or
        written), and 1 for anything unexpected, whose traceback is logged.

    Notes
    -----
    The configuration is parsed before any store is touched, so a bad deployment
    fails in the first second rather than after a long read.
    """
    configure_logging(os.environ.get("LOG_LEVEL", "INFO").strip().upper() or "INFO")
    try:
        config = Config.from_env()
    except DummyMLWPError as exc:
        logger.error(f"{exc}")
        return exc.exit_code

    configure_logging(config.log_level)
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
