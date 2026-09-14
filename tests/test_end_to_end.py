"""Whole-application tests: environment in, zarr store out, exit code checked."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from dummy_mlwp.__main__ import main

LEVELS = "isobaricInhPa:850/500/250"


def invoke(monkeypatch, env: dict[str, str]) -> int:
    monkeypatch.delenv("LOG_LEVEL", raising=False)
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    return main()


def open_output(env: dict[str, str]) -> xr.Dataset:
    return xr.open_zarr(env["OUTPUT_ZARR"], decode_timedelta=True)


def test_forecast_run(monkeypatch, base_env):
    env = base_env | {"N_FORECAST_STEPS": "8"}
    assert invoke(monkeypatch, env) == 0

    source = xr.open_zarr(env["INPUT_ZARR"])
    out = open_output(env)

    assert set(out.data_vars) == {"t2m", "tp", "crs"}
    assert out.t2m.dims == ("time", "y", "x")
    assert out.t2m.attrs["units"] == "K"
    assert out.tp.attrs["units"] == "mm"
    assert out.sizes["time"] == 8

    step = np.timedelta64(6, "h")
    assert out.time.values[0] == source.time.values[-1] + step
    assert np.array_equal(np.diff(out.time.values), np.full(7, step))
    assert out.forecastReferenceTime.values == source.time.values[-1]
    assert out.leadTime.values[0] == step

    assert np.array_equal(out.x.values, source.x.values)
    assert np.array_equal(out.y.values, source.y.values)
    assert np.isfinite(out.t2m.values).all()


def test_diagnostic_run_reuses_the_input_time_axis(monkeypatch, base_env):
    env = base_env | {"N_FORECAST_STEPS": "-1"}
    assert invoke(monkeypatch, env) == 0

    source = xr.open_zarr(env["INPUT_ZARR"])
    out = open_output(env)
    assert np.array_equal(out.time.values, source.time.values)
    assert out.leadTime.values[-1] == np.timedelta64(0, "ns")


def test_levels_round_trip(monkeypatch, base_env):
    env = base_env | {
        "LEVEL_COORDS": LEVELS,
        "INPUT_VARIABLES": "t2m,t:K@isobaricInhPa",
        "OUTPUT_VARIABLES": "z:m2s-2@isobaricInhPa,t2m:K",
        "N_FORECAST_STEPS": "3",
    }
    assert invoke(monkeypatch, env) == 0

    out = open_output(env)
    assert out.z.dims == ("time", "isobaricInhPa", "y", "x")
    assert list(out.isobaricInhPa.values) == [850, 500, 250]
    assert out.t2m.dims == ("time", "y", "x")


def test_latlon_input(monkeypatch, tmp_path, make_input):
    source = make_input(kind="latlon")
    env = {
        "INPUT_ZARR": str(source),
        "OUTPUT_ZARR": str(tmp_path / "out.zarr"),
        "INPUT_VARIABLES": "t2m,u10,v10",
        "OUTPUT_VARIABLES": "t2m:K",
        "DEVICE": "cpu",
        "MODEL_HIDDEN_CHANNELS": "8",
        "MODEL_LAYERS": "2",
    }
    assert invoke(monkeypatch, env) == 0

    out = open_output(env)
    assert out.t2m.dims == ("time", "latitude", "longitude")
    assert out.attrs["grid_type"] == "latlon"


def test_projected_run_keeps_the_crs(monkeypatch, base_env):
    assert invoke(monkeypatch, base_env) == 0
    out = open_output(base_env)
    assert "crs" in out.variables
    assert out.t2m.attrs["grid_mapping"] == "crs"


def test_transposed_input_dimensions_are_accepted(monkeypatch, tmp_path, make_input):
    source = make_input()
    ds = xr.open_zarr(source).transpose("x", "time", "y")
    shuffled = tmp_path / "shuffled.zarr"
    ds.to_zarr(shuffled, mode="w", consolidated=True, zarr_format=3)

    env = {
        "INPUT_ZARR": str(shuffled),
        "OUTPUT_ZARR": str(tmp_path / "out.zarr"),
        "INPUT_VARIABLES": "t2m,u10",
        "OUTPUT_VARIABLES": "t2m:K",
        "DEVICE": "cpu",
        "MODEL_HIDDEN_CHANNELS": "8",
        "MODEL_LAYERS": "2",
    }
    assert invoke(monkeypatch, env) == 0
    assert open_output(env).t2m.dims == ("time", "y", "x")


@pytest.mark.parametrize("mode", ["random", "persistence", "constant", "zeros"])
def test_every_output_mode_writes_a_store(monkeypatch, base_env, mode):
    env = base_env | {"OUTPUT_MODE": mode, "CONSTANT_VALUE": "3.5", "N_FORECAST_STEPS": "2"}
    assert invoke(monkeypatch, env) == 0
    values = open_output(env).t2m.values
    assert values.shape[0] == 2
    assert np.isfinite(values).all()

    if mode == "zeros":
        assert (values == 0).all()
    elif mode == "constant":
        assert (values == np.float32(3.5)).all()


def test_persistence_repeats_the_last_input_timestep(monkeypatch, base_env):
    env = base_env | {"OUTPUT_MODE": "persistence", "N_FORECAST_STEPS": "3"}
    assert invoke(monkeypatch, env) == 0

    source = xr.open_zarr(env["INPUT_ZARR"])
    out = open_output(env)
    expected = source.t2m.isel(time=-1).values
    for step in range(3):
        assert np.allclose(out.t2m.isel(time=step).values, expected)


def test_persistence_falls_back_for_unknown_variables(monkeypatch, base_env):
    """tp is not in the input, so it cannot persist — it takes the network output."""
    env = base_env | {"OUTPUT_MODE": "persistence", "N_FORECAST_STEPS": "2"}
    assert invoke(monkeypatch, env) == 0

    out = open_output(env)
    assert not np.allclose(out.tp.isel(time=0).values, out.tp.isel(time=1).values)


def test_runs_are_reproducible(monkeypatch, base_env, tmp_path):
    first = base_env | {"OUTPUT_ZARR": str(tmp_path / "a.zarr"), "N_FORECAST_STEPS": "2"}
    second = base_env | {"OUTPUT_ZARR": str(tmp_path / "b.zarr"), "N_FORECAST_STEPS": "2"}
    assert invoke(monkeypatch, first) == 0
    assert invoke(monkeypatch, second) == 0
    assert np.array_equal(open_output(first).t2m.values, open_output(second).t2m.values)


def test_seed_changes_the_output(monkeypatch, base_env, tmp_path):
    first = base_env | {"OUTPUT_ZARR": str(tmp_path / "a.zarr"), "RANDOM_SEED": "1"}
    second = base_env | {"OUTPUT_ZARR": str(tmp_path / "b.zarr"), "RANDOM_SEED": "2"}
    assert invoke(monkeypatch, first) == 0
    assert invoke(monkeypatch, second) == 0
    assert not np.allclose(open_output(first).t2m.values, open_output(second).t2m.values)


@pytest.mark.parametrize("zarr_format", [2, 3])
def test_output_format_matches_the_input(monkeypatch, tmp_path, make_input, zarr_format):
    source = make_input(zarr_format=zarr_format)
    env = {
        "INPUT_ZARR": str(source),
        "OUTPUT_ZARR": str(tmp_path / "out.zarr"),
        "INPUT_VARIABLES": "t2m",
        "OUTPUT_VARIABLES": "t2m:K",
        "DEVICE": "cpu",
        "MODEL_HIDDEN_CHANNELS": "8",
        "MODEL_LAYERS": "2",
    }
    assert invoke(monkeypatch, env) == 0

    out = tmp_path / "out.zarr"
    if zarr_format == 3:
        assert (out / "zarr.json").exists()
    else:
        assert (out / ".zgroup").exists()


def test_zarr_format_can_be_forced(monkeypatch, tmp_path, make_input):
    source = make_input(zarr_format=3)
    env = {
        "INPUT_ZARR": str(source),
        "OUTPUT_ZARR": str(tmp_path / "out.zarr"),
        "INPUT_VARIABLES": "t2m",
        "OUTPUT_VARIABLES": "t2m:K",
        "ZARR_FORMAT": "2",
        "DEVICE": "cpu",
        "MODEL_HIDDEN_CHANNELS": "8",
        "MODEL_LAYERS": "2",
    }
    assert invoke(monkeypatch, env) == 0
    assert (tmp_path / "out.zarr" / ".zgroup").exists()


def test_one_chunk_per_timestep(monkeypatch, base_env):
    env = base_env | {"N_FORECAST_STEPS": "4"}
    assert invoke(monkeypatch, env) == 0
    out = open_output(env)
    assert out.t2m.encoding["chunks"][0] == 1


# --- failure paths -------------------------------------------------------------------


def test_missing_required_variable_exits_2(monkeypatch, base_env):
    env = dict(base_env)
    del env["OUTPUT_VARIABLES"]
    monkeypatch.delenv("OUTPUT_VARIABLES", raising=False)
    assert invoke(monkeypatch, env) == 2


def test_bad_output_mode_exits_2(monkeypatch, base_env):
    assert invoke(monkeypatch, base_env | {"OUTPUT_MODE": "vibes"}) == 2


def test_undeclared_level_coordinate_exits_2(monkeypatch, base_env):
    env = base_env | {"OUTPUT_VARIABLES": "z@isobaricInhPa"}
    assert invoke(monkeypatch, env) == 2


def test_missing_input_variable_exits_3(monkeypatch, base_env):
    env = base_env | {"INPUT_VARIABLES": "t2m,notAVariable"}
    assert invoke(monkeypatch, env) == 3


def test_level_mismatch_exits_3(monkeypatch, base_env):
    env = base_env | {
        "LEVEL_COORDS": "isobaricInhPa:900/400",
        "INPUT_VARIABLES": "t@isobaricInhPa",
    }
    assert invoke(monkeypatch, env) == 3


def test_wrong_dimensionality_exits_3(monkeypatch, base_env):
    """t2m is 2D, so declaring it with levels must be rejected."""
    env = base_env | {
        "LEVEL_COORDS": LEVELS,
        "INPUT_VARIABLES": "t2m@isobaricInhPa",
    }
    assert invoke(monkeypatch, env) == 3


def test_irregular_grid_exits_3(monkeypatch, tmp_path, make_input):
    source = make_input()
    ds = xr.open_zarr(source).load()
    stretched = ds.x.values.copy()
    stretched[5:] += 1234.0
    ds = ds.assign_coords(x=stretched)
    irregular = tmp_path / "irregular.zarr"
    ds.to_zarr(irregular, mode="w", consolidated=True, zarr_format=3)

    env = {
        "INPUT_ZARR": str(irregular),
        "OUTPUT_ZARR": str(tmp_path / "out.zarr"),
        "INPUT_VARIABLES": "t2m",
        "OUTPUT_VARIABLES": "t2m:K",
        "DEVICE": "cpu",
    }
    assert invoke(monkeypatch, env) == 3


def test_missing_store_exits_3(monkeypatch, base_env, tmp_path):
    assert invoke(monkeypatch, base_env | {"INPUT_ZARR": str(tmp_path / "nope.zarr")}) == 3


def test_cuda_without_a_device_exits_4(monkeypatch, base_env):
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert invoke(monkeypatch, base_env | {"DEVICE": "cuda"}) == 4


def test_all_input_problems_are_reported_together(monkeypatch, base_env, capsys):
    """One run of a misconfigured pipeline should report every problem at once."""
    env = base_env | {"INPUT_VARIABLES": "missingOne,missingTwo,t2m"}
    assert invoke(monkeypatch, env) == 3

    stderr = capsys.readouterr().err
    assert "missingOne" in stderr
    assert "missingTwo" in stderr
