"""N_INPUT_TIMESTEPS: how much of the input's time axis the model actually sees."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from dummy_mlwp.__main__ import main
from dummy_mlwp.config import Config
from dummy_mlwp.errors import ConfigError, InputError
from dummy_mlwp.grid import detect_coords
from dummy_mlwp.inputs import select_input_timesteps

MINIMAL = {
    "INPUT_ZARR": "/in.zarr",
    "OUTPUT_ZARR": "/out.zarr",
    "INPUT_VARIABLES": "t2m",
    "OUTPUT_VARIABLES": "t2m:K",
}


def invoke(monkeypatch, env):
    monkeypatch.delenv("N_INPUT_TIMESTEPS", raising=False)
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    return main()


def test_unset_means_all():
    assert Config.from_env(MINIMAL).n_input_timesteps is None


def test_zero_is_rejected():
    with pytest.raises(ConfigError, match="leave it unset to use all of them"):
        Config.from_env(MINIMAL | {"N_INPUT_TIMESTEPS": "0"})


def test_non_integer_is_rejected():
    with pytest.raises(ConfigError, match="must be an integer"):
        Config.from_env(MINIMAL | {"N_INPUT_TIMESTEPS": "two"})


def test_none_selects_everything(make_dataset):
    ds = make_dataset(nt=5)
    assert select_input_timesteps(ds, detect_coords(ds), None).sizes["time"] == 5


def test_positive_takes_the_first_n(make_dataset):
    ds = make_dataset(nt=5)
    selected = select_input_timesteps(ds, detect_coords(ds), 2)
    assert selected.sizes["time"] == 2
    assert np.array_equal(selected.time.values, ds.time.values[:2])


def test_negative_takes_the_last_n(make_dataset):
    ds = make_dataset(nt=5)
    selected = select_input_timesteps(ds, detect_coords(ds), -2)
    assert selected.sizes["time"] == 2
    assert np.array_equal(selected.time.values, ds.time.values[-2:])


def test_asking_for_exactly_what_exists_is_fine(make_dataset):
    ds = make_dataset(nt=4)
    assert select_input_timesteps(ds, detect_coords(ds), 4).sizes["time"] == 4
    assert select_input_timesteps(ds, detect_coords(ds), -4).sizes["time"] == 4


@pytest.mark.parametrize("n", [5, -5, 99])
def test_asking_for_more_than_exists_is_an_error(make_dataset, n):
    ds = make_dataset(nt=4)
    with pytest.raises(InputError, match="but the input has only 4"):
        select_input_timesteps(ds, detect_coords(ds), n)


def test_forecast_starts_after_the_selected_window(monkeypatch, base_env):
    """With the first two timesteps selected, the forecast runs on from the second."""
    env = base_env | {"N_INPUT_TIMESTEPS": "2", "N_FORECAST_STEPS": "3"}
    assert invoke(monkeypatch, env) == 0

    source = xr.open_zarr(env["INPUT_ZARR"])
    out = xr.open_zarr(env["OUTPUT_ZARR"], decode_timedelta=True)

    step = np.timedelta64(6, "h")
    assert out.forecastReferenceTime.values == source.time.values[1]
    assert out.time.values[0] == source.time.values[1] + step


def test_last_window_uses_the_end_of_the_input(monkeypatch, base_env):
    env = base_env | {"N_INPUT_TIMESTEPS": "-2", "N_FORECAST_STEPS": "1"}
    assert invoke(monkeypatch, env) == 0

    source = xr.open_zarr(env["INPUT_ZARR"])
    out = xr.open_zarr(env["OUTPUT_ZARR"], decode_timedelta=True)
    assert out.forecastReferenceTime.values == source.time.values[-1]


def test_diagnostic_mode_outputs_only_the_selected_timesteps(monkeypatch, base_env):
    env = base_env | {"N_INPUT_TIMESTEPS": "2", "N_FORECAST_STEPS": "-1"}
    assert invoke(monkeypatch, env) == 0

    source = xr.open_zarr(env["INPUT_ZARR"])
    out = xr.open_zarr(env["OUTPUT_ZARR"])
    assert out.sizes["time"] == 2
    assert np.array_equal(out.time.values, source.time.values[:2])


def test_too_few_timesteps_exits_3(monkeypatch, base_env):
    assert invoke(monkeypatch, base_env | {"N_INPUT_TIMESTEPS": "99"}) == 3


def test_zero_exits_2(monkeypatch, base_env):
    assert invoke(monkeypatch, base_env | {"N_INPUT_TIMESTEPS": "0"}) == 2


def test_single_selected_timestep_needs_a_declared_resolution(monkeypatch, base_env):
    """One timestep gives no dt to infer, so a forecast needs FORECAST_TIMESTEP."""
    env = base_env | {"N_INPUT_TIMESTEPS": "-1", "N_FORECAST_STEPS": "2"}
    assert invoke(monkeypatch, env) == 2

    env = env | {"FORECAST_TIMESTEP": "PT1H"}
    assert invoke(monkeypatch, env) == 0
    out = xr.open_zarr(env["OUTPUT_ZARR"], decode_timedelta=True)
    assert np.diff(out.time.values)[0] == np.timedelta64(1, "h")
