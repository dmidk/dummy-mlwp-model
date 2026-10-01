"""The startup log: the version first, then every setting the run will use."""

from __future__ import annotations

import re
from dataclasses import fields

import pytest

from dummy_mlwp import __version__
from dummy_mlwp.__main__ import describe_config, log_config, main
from dummy_mlwp.config import Config

MINIMAL = {
    "INPUT_ZARR": "/in.zarr",
    "OUTPUT_ZARR": "/out.zarr",
    "INPUT_VARIABLES": "t2m,u10",
    "OUTPUT_VARIABLES": "t2m:K",
}

#: Settings whose defaults the end-to-end tests assert on; cleared so a developer's own
#: shell cannot leak into the run.
DEFAULTED = ("LOG_LEVEL", "N_INPUT_TIMESTEPS", "FORECAST_TIMESTEP", "TIME_COORD", "ZARR_FORMAT")


def invoke(monkeypatch, env: dict[str, str]) -> int:
    for key in DEFAULTED:
        monkeypatch.delenv(key, raising=False)
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    return main()


# --- the renderer --------------------------------------------------------------------


def test_every_setting_is_labelled_with_its_environment_variable():
    """A new Config field must show up in the log, under the variable that sets it."""
    storage = {"src_storage_options", "dst_storage_options"}
    expected = {f.name.upper() for f in fields(Config) if f.name not in storage}
    assert set(describe_config(Config.from_env(MINIMAL))) == expected


def test_defaults_are_rendered():
    settings = describe_config(Config.from_env(MINIMAL))
    assert settings["N_INPUT_TIMESTEPS"] == "unset (all)"
    assert settings["N_FORECAST_STEPS"] == "-1"
    assert settings["FORECAST_TIMESTEP"] == "unset (inferred from input)"
    assert settings["LEVEL_COORDS"] == "unset (none)"
    assert settings["TIME_COORD"] == "unset (auto-detect)"
    assert settings["X_COORD"] == "unset (auto-detect)"
    assert settings["Y_COORD"] == "unset (auto-detect)"
    assert settings["OUTPUT_MODE"] == "random"
    assert settings["RANDOM_SEED"] == "0"
    assert settings["CONSTANT_VALUE"] == "0.0"
    assert settings["DEVICE"] == "auto"
    assert settings["MODEL_HIDDEN_CHANNELS"] == "64"
    assert settings["MODEL_LAYERS"] == "4"
    assert settings["ZARR_FORMAT"] == "auto"
    assert settings["LOG_LEVEL"] == "INFO"


def test_values_are_rendered_in_the_declared_grammar():
    config = Config.from_env(
        MINIMAL
        | {
            "LEVEL_COORDS": "isobaricInhPa: 850/500/250, hybrid:0.5/1",
            "INPUT_VARIABLES": " t2m , t:K@isobaricInhPa ",
            "OUTPUT_VARIABLES": "u:m s-1@hybrid,tp:mm",
            "N_INPUT_TIMESTEPS": "-2",
            "N_FORECAST_STEPS": "8",
            "FORECAST_TIMESTEP": "PT6H",
            "TIME_COORD": "valid_time",
            "DEVICE": "CPU",
        }
    )
    settings = describe_config(config)
    assert settings["INPUT_ZARR"] == "/in.zarr"
    assert settings["LEVEL_COORDS"] == "isobaricInhPa:850/500/250,hybrid:0.5/1.0"
    assert settings["INPUT_VARIABLES"] == "t2m,t:K@isobaricInhPa"
    assert settings["OUTPUT_VARIABLES"] == "u:m s-1@hybrid,tp:mm"
    assert settings["N_INPUT_TIMESTEPS"] == "-2"
    assert settings["N_FORECAST_STEPS"] == "8"
    assert settings["FORECAST_TIMESTEP"] == "P0DT6H0M0S"
    assert settings["TIME_COORD"] == "valid_time"
    assert settings["DEVICE"] == "cpu"


def test_no_secrets_are_rendered():
    env = MINIMAL | {
        "INPUT_ZARR": "s3://bucket/in.zarr",
        "OUTPUT_ZARR": "s3://bucket/out.zarr",
        "SRC_AWS_ACCESS_KEY_ID": "AKIAsrcKey",
        "SRC_AWS_SECRET_ACCESS_KEY": "srcSecret",
        "DST_AWS_SESSION_TOKEN": "dstToken",
        "DST_STORAGE_OPTIONS": '{"secret": "jsonSecret", "password": "hunter2"}',
    }
    rendered = " ".join(f"{k}={v}" for k, v in describe_config(Config.from_env(env)).items())
    for secret in ("AKIAsrcKey", "srcSecret", "dstToken", "jsonSecret", "hunter2"):
        assert secret not in rendered


def test_one_log_record_per_setting():
    """Separate records keep their timestamp and level once a collector splits lines."""
    from loguru import logger

    config = Config.from_env(MINIMAL)
    messages: list[str] = []
    sink = logger.add(lambda m: messages.append(m.record["message"]), level="INFO")
    try:
        log_config(config)
    finally:
        logger.remove(sink)

    settings = describe_config(config)
    assert len(messages) == 1 + len(settings)
    assert not any("\n" in message for message in messages)
    for message, (name, value) in zip(messages[1:], settings.items(), strict=True):
        assert re.fullmatch(rf"  {name} += {re.escape(value)}", message)


# --- end to end ----------------------------------------------------------------------


def test_successful_run_logs_the_version_and_the_configuration(monkeypatch, base_env, capsys):
    env = base_env | {"N_FORECAST_STEPS": "2"}
    assert invoke(monkeypatch, env) == 0

    lines = capsys.readouterr().err.splitlines()
    assert f"Starting dummy-mlwp-model {__version__}" in lines[0]

    stderr = "\n".join(lines)
    expected = {
        "INPUT_ZARR": env["INPUT_ZARR"],
        "OUTPUT_ZARR": env["OUTPUT_ZARR"],
        "INPUT_VARIABLES": "t2m,u10,v10",
        "OUTPUT_VARIABLES": "t2m:K,tp:mm",
        "N_FORECAST_STEPS": "2",
        "N_INPUT_TIMESTEPS": "unset (all)",
        "TIME_COORD": "unset (auto-detect)",
        "DEVICE": "cpu",
        "MODEL_HIDDEN_CHANNELS": "8",
        "ZARR_FORMAT": "auto",
    }
    for name, value in expected.items():
        assert re.search(rf"\b{name} += {re.escape(value)}$", stderr, re.MULTILINE), name

    # Logged before anything touches the input store.
    assert stderr.index("Effective configuration") < stderr.index("Opening input store")


@pytest.mark.parametrize(
    "change",
    [{"OUTPUT_MODE": "vibes"}, {"N_FORECAST_STEPS": "0"}, {"OUTPUT_VARIABLES": ""}],
)
def test_config_error_still_logs_the_version_first(monkeypatch, base_env, capsys, change):
    assert invoke(monkeypatch, base_env | change) == 2

    lines = capsys.readouterr().err.splitlines()
    assert f"Starting dummy-mlwp-model {__version__}" in lines[0]
    assert "ERROR" in lines[-1]
    assert not any("Effective configuration" in line for line in lines)
