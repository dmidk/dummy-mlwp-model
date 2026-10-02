"""Units declared in INPUT_VARIABLES are asserted against the store, as exact strings."""

from __future__ import annotations

import pytest

from dummy_mlwp.__main__ import main
from dummy_mlwp.config import Config
from dummy_mlwp.errors import InputError
from dummy_mlwp.grid import detect_coords
from dummy_mlwp.inputs import _validate_variables, validate_input

LEVELS = "isobaricInhPa:850/500/250"

MINIMAL = {
    "INPUT_ZARR": "/in.zarr",
    "OUTPUT_ZARR": "/out.zarr",
    "OUTPUT_VARIABLES": "t2m:K",
    "LEVEL_COORDS": LEVELS,
}


def problems_for(ds, input_variables: str) -> list[str]:
    config = Config.from_env(MINIMAL | {"INPUT_VARIABLES": input_variables})
    return _validate_variables(ds, config, detect_coords(ds))


def invoke(monkeypatch, env: dict[str, str]) -> int:
    monkeypatch.delenv("LOG_LEVEL", raising=False)
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    return main()


# --- unit ----------------------------------------------------------------------------


def test_matching_units_pass(make_dataset):
    ds = make_dataset(levels=[850, 500, 250])
    assert problems_for(ds, "t2m:K,u10:m s-1,t:K@isobaricInhPa") == []


def test_undeclared_units_are_not_checked(make_dataset):
    ds = make_dataset()
    ds.t2m.attrs["units"] = "furlongs"
    del ds.u10.attrs["units"]
    assert problems_for(ds, "t2m,u10") == []


def test_mismatched_units_are_reported(make_dataset):
    problems = problems_for(make_dataset(), "t2m:degC")
    assert len(problems) == 1
    assert "'t2m'" in problems[0]
    assert "units 'K'" in problems[0]
    assert "declares units 'degC'" in problems[0]


@pytest.mark.parametrize("declared", ["m/s", "m s**-1", "m  s-1", "M S-1"])
def test_units_are_compared_as_exact_strings(make_dataset, declared):
    """No unit algebra or normalisation: only the store's literal 'm s-1' matches."""
    problems = problems_for(make_dataset(), f"u10:{declared}")
    assert len(problems) == 1
    assert "units 'm s-1'" in problems[0]


def test_missing_units_attribute_is_reported(make_dataset):
    ds = make_dataset()
    del ds.t2m.attrs["units"]
    problems = problems_for(ds, "t2m:K")
    assert len(problems) == 1
    assert "'t2m' has no 'units' attribute" in problems[0]
    assert "declares units 'K'" in problems[0]


def test_non_string_units_attribute_never_matches(make_dataset):
    ds = make_dataset()
    ds.t2m.attrs["units"] = 1
    problems = problems_for(ds, "t2m:1")
    assert len(problems) == 1
    assert "units 1 " in problems[0]


def test_wrong_dimensions_and_wrong_units_are_both_reported(make_dataset):
    problems = problems_for(make_dataset(levels=[850, 500, 250]), "t2m:degC@isobaricInhPa")
    assert len(problems) == 2
    assert any("dimensions" in p for p in problems)
    assert any("declares units 'degC'" in p for p in problems)


def test_missing_variable_gets_no_units_message(make_dataset):
    problems = problems_for(make_dataset(), "notAVariable:K")
    assert len(problems) == 1
    assert "not in the input" in problems[0]


def test_validate_input_raises_with_every_units_problem(make_dataset):
    ds = make_dataset()
    config = Config.from_env(MINIMAL | {"INPUT_VARIABLES": "t2m:degC,u10:m/s,v10:m s-1"})
    with pytest.raises(InputError) as info:
        validate_input(ds, config, detect_coords(ds))
    message = str(info.value)
    assert "declares units 'degC'" in message
    assert "declares units 'm/s'" in message
    assert "'v10'" not in message


# --- end to end ----------------------------------------------------------------------


def test_matching_units_run_succeeds(monkeypatch, base_env):
    env = base_env | {
        "LEVEL_COORDS": LEVELS,
        "INPUT_VARIABLES": "t2m:K,u10:m s-1,t:K@isobaricInhPa",
    }
    assert invoke(monkeypatch, env) == 0


def test_mismatched_units_exit_3(monkeypatch, base_env, capsys):
    env = base_env | {"INPUT_VARIABLES": "t2m:degC,u10,v10"}
    assert invoke(monkeypatch, env) == 3

    stderr = capsys.readouterr().err
    assert "'t2m' has units 'K'" in stderr
    assert "declares units 'degC'" in stderr


def test_units_problem_is_reported_with_other_problems(monkeypatch, base_env, capsys):
    """A units mismatch must not hide, or be hidden by, any other problem in the run."""
    env = base_env | {"INPUT_VARIABLES": "t2m:degC,notAVariable,u10:m/s"}
    assert invoke(monkeypatch, env) == 3

    stderr = capsys.readouterr().err
    assert "declares units 'degC'" in stderr
    assert "'notAVariable' (from INPUT_VARIABLES) is not in the input" in stderr
    assert "declares units 'm/s'" in stderr
