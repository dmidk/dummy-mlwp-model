from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from dummy_mlwp.errors import ConfigError, InputError
from dummy_mlwp.timeaxis import build_output_times, infer_dt, validate_times


def times(n: int = 4, freq: str = "6h") -> np.ndarray:
    return pd.date_range("2024-01-01", periods=n, freq=freq).values


def test_infers_the_timestep():
    assert infer_dt(times()) == pd.Timedelta("6h")
    assert infer_dt(times(freq="1h")) == pd.Timedelta("1h")


def test_single_timestep_cannot_give_a_timestep():
    with pytest.raises(InputError, match="set FORECAST_TIMESTEP"):
        infer_dt(times(n=1))


def test_uneven_spacing_is_rejected():
    uneven = np.concatenate([times(3), times(1, freq="6h") + np.timedelta64(31, "h")])
    with pytest.raises(InputError, match="not evenly spaced"):
        infer_dt(uneven)


def test_non_increasing_times_are_rejected():
    backwards = times()[::-1]
    with pytest.raises(InputError, match="not strictly increasing"):
        infer_dt(backwards)


def test_validate_times_collects_rather_than_raises():
    assert validate_times(times()) == []
    assert validate_times(times(n=1)) == []  # a single step is fine unless we forecast
    (problem,) = validate_times(times()[::-1])
    assert "not strictly increasing" in problem


def test_minus_one_keeps_the_input_time_axis():
    input_times = times()
    out, lead, reference = build_output_times(input_times, -1)

    assert np.array_equal(out, input_times)
    assert reference == input_times[-1]
    # Diagnostics on the input's own times: lead times run up to zero, not beyond.
    assert lead[-1] == np.timedelta64(0, "ns")
    assert (lead <= np.timedelta64(0, "ns")).all()


def test_positive_steps_extend_beyond_the_last_input_time():
    input_times = times()
    out, lead, reference = build_output_times(input_times, 8)

    assert out.size == 8
    assert reference == input_times[-1]
    assert out[0] == input_times[-1] + np.timedelta64(6, "h")
    assert np.array_equal(np.diff(out), np.full(7, np.timedelta64(6, "h")))
    assert lead[0] == np.timedelta64(6, "h")
    assert lead[-1] == np.timedelta64(48, "h")


def test_forecast_timestep_covers_a_single_timestep_input():
    input_times = times(n=1)
    out, _, _ = build_output_times(input_times, 3, pd.Timedelta("1h"))

    assert out.size == 3
    assert out[0] == input_times[-1] + np.timedelta64(1, "h")


def test_single_timestep_without_a_declared_resolution_is_a_config_error():
    with pytest.raises(ConfigError, match="Set FORECAST_TIMESTEP"):
        build_output_times(times(n=1), 3)


def test_forecast_timestep_overrides_the_inferred_one():
    out, _, _ = build_output_times(times(), 2, pd.Timedelta("1h"))
    assert np.diff(out)[0] == np.timedelta64(1, "h")
