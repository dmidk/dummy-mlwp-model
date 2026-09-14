from __future__ import annotations

import numpy as np
import pytest

from dummy_mlwp.errors import ConfigError
from dummy_mlwp.varspec import (
    VarSpec,
    channel_layout,
    parse_level_coords,
    parse_var_specs,
)


def test_parses_plain_names():
    specs = parse_var_specs("t2m,u10,v10", {}, "INPUT_VARIABLES")
    assert specs == [VarSpec("t2m"), VarSpec("u10"), VarSpec("v10")]


def test_parses_units_and_levels():
    levels = parse_level_coords("isobaricInhPa:850/500/250")
    specs = parse_var_specs("t2m:K,t:K@isobaricInhPa", levels, "OUTPUT_VARIABLES")
    assert specs[0] == VarSpec("t2m", units="K")
    assert specs[1] == VarSpec("t", units="K", level_coord="isobaricInhPa")


def test_units_may_contain_spaces_and_slashes():
    (spec,) = parse_var_specs("u:m s-1", {}, "OUTPUT_VARIABLES")
    assert spec.units == "m s-1"
    (spec,) = parse_var_specs("tp:kg/m2", {}, "OUTPUT_VARIABLES")
    assert spec.units == "kg/m2"


def test_whitespace_around_entries_is_ignored():
    specs = parse_var_specs(" t2m , u10 ", {}, "INPUT_VARIABLES")
    assert [s.name for s in specs] == ["t2m", "u10"]


def test_integral_levels_stay_integers():
    levels = parse_level_coords("isobaricInhPa:850/500/250")
    assert levels["isobaricInhPa"].dtype.kind == "i"
    assert list(levels["isobaricInhPa"]) == [850, 500, 250]


def test_fractional_levels_stay_floats():
    levels = parse_level_coords("sigma:0.2/0.5/0.9")
    assert levels["sigma"].dtype.kind == "f"
    assert np.allclose(levels["sigma"], [0.2, 0.5, 0.9])


def test_several_level_coordinates():
    levels = parse_level_coords("isobaricInhPa:850/500,heightAboveGround:10/100")
    assert set(levels) == {"isobaricInhPa", "heightAboveGround"}


def test_empty_level_coords_is_valid():
    assert parse_level_coords("") == {}


def test_str_roundtrips():
    levels = parse_level_coords("isobaricInhPa:850/500")
    (spec,) = parse_var_specs("t:K@isobaricInhPa", levels, "OUTPUT_VARIABLES")
    assert str(spec) == "t:K@isobaricInhPa"


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("t2m,t2m", "more than once"),
        ("", "no variables declared"),
        ("t2m:", "no units"),
        ("t2m@", "trailing '@'"),
        ("2wet:K", "not a valid name"),
        ("t@unknownCoord", "not declared in LEVEL_COORDS"),
    ],
)
def test_rejects_malformed_specs(text, message):
    with pytest.raises(ConfigError, match=message):
        parse_var_specs(text, {}, "OUTPUT_VARIABLES")


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("isobaricInhPa", "has no ':'"),
        ("isobaricInhPa:", "has no values"),
        ("isobaricInhPa:850/nope", "non-numeric"),
        ("isobaricInhPa:850/850", "duplicate values"),
        ("isobaricInhPa:850,isobaricInhPa:500", "declared more than once"),
    ],
)
def test_rejects_malformed_level_coords(text, message):
    with pytest.raises(ConfigError, match=message):
        parse_level_coords(text)


def test_channel_layout_expands_levels_in_order():
    levels = parse_level_coords("isobaricInhPa:850/500/250")
    specs = parse_var_specs("t2m,t@isobaricInhPa,u10", levels, "INPUT_VARIABLES")
    layout = channel_layout(specs, levels)
    assert [(s.name, i) for s, i in layout] == [
        ("t2m", None),
        ("t", 0),
        ("t", 1),
        ("t", 2),
        ("u10", None),
    ]


def test_dims_follow_the_level_declaration():
    levels = parse_level_coords("isobaricInhPa:850/500")
    (flat,) = parse_var_specs("t2m", {}, "INPUT_VARIABLES")
    (upper,) = parse_var_specs("t@isobaricInhPa", levels, "INPUT_VARIABLES")
    assert flat.dims("time", "y", "x") == ("time", "y", "x")
    assert upper.dims("time", "y", "x") == ("time", "isobaricInhPa", "y", "x")
    assert flat.n_levels(levels) == 1
    assert upper.n_levels(levels) == 2
