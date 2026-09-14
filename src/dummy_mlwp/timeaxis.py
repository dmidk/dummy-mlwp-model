"""Forecast time construction.

Two regimes, selected by N_FORECAST_STEPS:

* ``-1`` — one prediction per input timestep (a regression/classification framing).
  The output time coordinate *is* the input time coordinate.
* ``K > 0`` — K future steps at the input's own time resolution, starting one dt after
  the last input time.

Either way the output carries ``forecastReferenceTime`` (the last input time, i.e. the
analysis time) and a ``leadTime`` coordinate, so downstream code sees something shaped
like a real forecast product. In the ``-1`` regime lead times are zero or negative,
which is the honest description of a diagnostic evaluated on its own input times.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger

from .errors import ConfigError, InputError


def infer_dt(times: np.ndarray) -> pd.Timedelta:
    """Infer the timestep of an evenly spaced, strictly increasing time axis."""
    if times.size < 2:
        raise InputError(
            "Cannot infer the input timestep from a single timestep; "
            "set FORECAST_TIMESTEP (e.g. 'PT6H') to say what the output resolution is"
        )
    deltas = np.diff(times.astype("datetime64[ns]").astype("int64"))
    if np.any(deltas <= 0):
        first_bad = int(np.argmax(deltas <= 0))
        raise InputError(
            f"Input time coordinate is not strictly increasing: "
            f"{times[first_bad]} is followed by {times[first_bad + 1]}"
        )
    if not np.all(deltas == deltas[0]):
        worst = int(np.argmax(np.abs(deltas - deltas[0])))
        raise InputError(
            f"Input time coordinate is not evenly spaced: the first step is "
            f"{pd.Timedelta(int(deltas[0]))} but the step at index {worst} is "
            f"{pd.Timedelta(int(deltas[worst]))}"
        )
    return pd.Timedelta(int(deltas[0]))


def validate_times(times: np.ndarray) -> list[str]:
    """Collect (rather than raise) time-axis problems, for batched input validation."""
    if times.size < 2:
        return []
    try:
        infer_dt(times)
    except InputError as exc:
        return [str(exc)]
    return []


def build_output_times(
    input_times: np.ndarray,
    n_forecast_steps: int,
    forecast_timestep: pd.Timedelta | None = None,
) -> tuple[np.ndarray, np.ndarray, np.datetime64]:
    """Return ``(times, lead_times, reference_time)`` for the output store."""
    input_times = input_times.astype("datetime64[ns]")
    reference_time = input_times[-1]

    if n_forecast_steps == -1:
        times = input_times
    else:
        dt = _output_dt(input_times, forecast_timestep)
        step = np.timedelta64(dt.value, "ns")
        times = reference_time + step * np.arange(1, n_forecast_steps + 1)
        logger.info(f"Forecasting {n_forecast_steps} step(s) of {dt} from {reference_time}")

    return times, times - reference_time, reference_time


def _output_dt(input_times: np.ndarray, forecast_timestep: pd.Timedelta | None) -> pd.Timedelta:
    if input_times.size < 2:
        if forecast_timestep is None:
            raise ConfigError(
                "The input has a single timestep, so the forecast resolution cannot be "
                "inferred. Set FORECAST_TIMESTEP (e.g. 'PT6H')."
            )
        return forecast_timestep

    inferred = infer_dt(input_times)
    if forecast_timestep is not None and forecast_timestep != inferred:
        logger.warning(
            f"FORECAST_TIMESTEP={forecast_timestep} overrides the input's own "
            f"timestep of {inferred}"
        )
        return forecast_timestep
    return inferred
