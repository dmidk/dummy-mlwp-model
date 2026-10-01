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


# --- problems are collected, not raised one at a time --------------------------------


def config_error(env: dict[str, str]) -> str:
    with pytest.raises(ConfigError) as excinfo:
        Config.from_env(env)
    return str(excinfo.value)


def test_every_bad_variable_is_reported_in_one_error():
    env = {k: v for k, v in MINIMAL.items() if k != "OUTPUT_ZARR"} | {
        "N_FORECAST_STEPS": "0",
        "OUTPUT_MODE": "vibes",
        "MODEL_LAYERS": "1",
        "FORECAST_TIMESTEP": "soon",
    }
    message = config_error(env)
    lines = message.splitlines()
    assert lines[0] == "The environment configuration has 5 problems:"
    assert len(lines) == 6
    assert all(line.startswith("  - ") for line in lines[1:])
    for expected in (
        "N_FORECAST_STEPS must be -1",
        "OUTPUT_MODE must be one of",
        "MODEL_LAYERS must be >= 2",
        "FORECAST_TIMESTEP must be an ISO 8601 duration",
        "OUTPUT_ZARR is required",
    ):
        assert expected in message


def test_a_single_problem_is_reported_as_its_bare_message():
    message = config_error(MINIMAL | {"OUTPUT_MODE": "vibes"})
    assert message == "OUTPUT_MODE must be one of random, persistence, constant, zeros, got 'vibes'"


def test_an_unparseable_number_is_not_also_range_checked():
    """'eight' is one problem, not 'must be an integer' plus an out-of-range complaint."""
    message = config_error(MINIMAL | {"N_FORECAST_STEPS": "eight", "MODEL_LAYERS": "two"})
    assert message.splitlines()[0] == "The environment configuration has 2 problems:"
    assert "must be -1" not in message
    assert "must be >= 2" not in message


def test_broken_level_coords_do_not_flag_every_level_reference():
    """Each @reference would otherwise be reported as undeclared, burying the real fault."""
    message = config_error(
        MINIMAL
        | {
            "LEVEL_COORDS": "isobaricInhPa:850/high",
            "INPUT_VARIABLES": "t2m,t@isobaricInhPa",
            "OUTPUT_VARIABLES": "z@isobaricInhPa",
        }
    )
    assert message.startswith("LEVEL_COORDS: level coordinate 'isobaricInhPa' has a non-numeric")
    assert "not declared" not in message


def test_broken_level_coords_still_let_variable_grammar_be_checked():
    message = config_error(
        MINIMAL
        | {
            "LEVEL_COORDS": "isobaricInhPa:850/high",
            "INPUT_VARIABLES": "t@isobaricInhPa,bad name",
        }
    )
    assert message.splitlines()[0] == "The environment configuration has 2 problems:"
    assert "LEVEL_COORDS: level coordinate 'isobaricInhPa'" in message
    assert "INPUT_VARIABLES: variable name 'bad name' is not a valid name" in message
    assert "not declared" not in message


def test_undeclared_level_reference_is_still_reported_alongside_other_problems():
    message = config_error(MINIMAL | {"OUTPUT_VARIABLES": "z@isobaricInhPa", "DEVICE": "tpu"})
    assert "which is not declared in LEVEL_COORDS" in message
    assert "DEVICE must be one of" in message


def test_storage_option_problems_are_collected():
    message = config_error(
        MINIMAL
        | {
            "OUTPUT_ZARR": "s3://bucket/out.zarr",
            "SRC_STORAGE_OPTIONS": "not json",
            "DST_S3_ANON": "maybe",
            "N_INPUT_TIMESTEPS": "0",
        }
    )
    assert message.splitlines()[0] == "The environment configuration has 3 problems:"
    assert "SRC_STORAGE_OPTIONS must be valid JSON" in message
    assert "DST_S3_ANON must be a boolean" in message
    assert "N_INPUT_TIMESTEPS must be a positive number" in message


def test_storage_options_are_not_parsed_without_a_store_uri():
    """The URI's scheme decides which options apply; a missing URI is the one problem."""
    env = {k: v for k, v in MINIMAL.items() if k != "INPUT_ZARR"}
    message = config_error(env | {"SRC_STORAGE_OPTIONS": "not json"})
    assert message == "INPUT_ZARR is required but not set"


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
