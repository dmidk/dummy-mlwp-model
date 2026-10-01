from __future__ import annotations

import pandas as pd
import pytest

from dummy_mlwp.config import Config
from dummy_mlwp.errors import ConfigError

MINIMAL = {
    "INPUT_ZARR": "/in.zarr",
    "OUTPUT_ZARR": "/out.zarr",
    "INPUT_VARIABLES": "t2m,u10",
    "OUTPUT_VARIABLES": "t2m:K",
}


def test_defaults():
    config = Config.from_env(MINIMAL)
    assert config.n_forecast_steps == -1
    assert config.predicts_input_timesteps
    assert config.output_mode == "random"
    assert config.device == "auto"
    assert config.zarr_format == "auto"
    assert config.random_seed == 0
    assert config.forecast_timestep is None
    assert config.time_coord is None


def test_channel_counts_account_for_levels():
    config = Config.from_env(
        MINIMAL
        | {
            "LEVEL_COORDS": "isobaricInhPa:850/500/250",
            "INPUT_VARIABLES": "t2m,t@isobaricInhPa",
            "OUTPUT_VARIABLES": "z@isobaricInhPa",
        }
    )
    assert config.n_input_channels == 4
    assert config.n_output_channels == 3


@pytest.mark.parametrize("key", sorted(MINIMAL))
def test_required_variables(key):
    env = {k: v for k, v in MINIMAL.items() if k != key}
    with pytest.raises(ConfigError, match=f"{key} is required"):
        Config.from_env(env)


def test_blank_counts_as_missing():
    with pytest.raises(ConfigError, match="INPUT_ZARR is required"):
        Config.from_env(MINIMAL | {"INPUT_ZARR": "   "})


@pytest.mark.parametrize("value", ["0", "-2", "-100"])
def test_rejects_invalid_forecast_step_counts(value):
    with pytest.raises(ConfigError, match="N_FORECAST_STEPS must be -1"):
        Config.from_env(MINIMAL | {"N_FORECAST_STEPS": value})


def test_rejects_non_integer_forecast_steps():
    with pytest.raises(ConfigError, match="must be an integer"):
        Config.from_env(MINIMAL | {"N_FORECAST_STEPS": "eight"})


def test_positive_forecast_steps():
    config = Config.from_env(MINIMAL | {"N_FORECAST_STEPS": "8"})
    assert config.n_forecast_steps == 8
    assert not config.predicts_input_timesteps


@pytest.mark.parametrize(("value", "expected"), [("PT6H", "6h"), ("6h", "6h"), ("P1D", "1d")])
def test_forecast_timestep_accepts_iso_durations(value, expected):
    config = Config.from_env(MINIMAL | {"FORECAST_TIMESTEP": value})
    assert config.forecast_timestep == pd.Timedelta(expected)


def test_rejects_malformed_forecast_timestep():
    with pytest.raises(ConfigError, match="ISO 8601 duration"):
        Config.from_env(MINIMAL | {"FORECAST_TIMESTEP": "soon"})


def test_rejects_negative_forecast_timestep():
    with pytest.raises(ConfigError, match="must be positive"):
        Config.from_env(MINIMAL | {"FORECAST_TIMESTEP": "-PT6H"})


@pytest.mark.parametrize(
    ("key", "value", "message"),
    [
        ("OUTPUT_MODE", "vibes", "OUTPUT_MODE must be one of"),
        ("DEVICE", "tpu", "DEVICE must be one of"),
        ("ZARR_FORMAT", "1", "ZARR_FORMAT must be one of"),
        ("MODEL_HIDDEN_CHANNELS", "0", "MODEL_HIDDEN_CHANNELS must be >= 1"),
        ("MODEL_LAYERS", "1", "MODEL_LAYERS must be >= 2"),
        ("CONSTANT_VALUE", "warm", "must be a number"),
    ],
)
def test_rejects_bad_values(key, value, message):
    with pytest.raises(ConfigError, match=message):
        Config.from_env(MINIMAL | {key: value})


def test_choices_are_case_insensitive():
    config = Config.from_env(MINIMAL | {"OUTPUT_MODE": "Persistence", "DEVICE": "CPU"})
    assert config.output_mode == "persistence"
    assert config.device == "cpu"


def test_coordinate_overrides_are_read():
    config = Config.from_env(
        MINIMAL | {"TIME_COORD": "valid_time", "X_COORD": "longitude", "Y_COORD": "latitude"}
    )
    assert (config.time_coord, config.x_coord, config.y_coord) == (
        "valid_time",
        "longitude",
        "latitude",
    )


def test_provenance_round_trips_the_spec_strings():
    config = Config.from_env(
        MINIMAL | {"LEVEL_COORDS": "isobaricInhPa:850", "OUTPUT_VARIABLES": "z:m2s-2@isobaricInhPa"}
    )
    provenance = config.provenance()
    assert provenance["output_variables"] == "z:m2s-2@isobaricInhPa"
    assert provenance["source"] == "/in.zarr"


def test_output_only_level_coordinate_must_have_a_known_cf_description():
    env = MINIMAL | {"LEVEL_COORDS": "myLevels:1/2", "OUTPUT_VARIABLES": "z@myLevels"}
    with pytest.raises(ConfigError, match="not one of the known level coordinates"):
        Config.from_env(env)


def test_known_output_only_level_coordinate_is_accepted():
    env = MINIMAL | {
        "LEVEL_COORDS": "heightAboveGround:10/100",
        "OUTPUT_VARIABLES": "u@heightAboveGround",
    }
    assert Config.from_env(env).n_output_channels == 2


def test_level_coordinate_read_from_the_input_needs_no_known_description():
    env = MINIMAL | {
        "LEVEL_COORDS": "myLevels:1/2",
        "INPUT_VARIABLES": "q@myLevels",
        "OUTPUT_VARIABLES": "q@myLevels",
    }
    assert Config.from_env(env).n_output_channels == 2
