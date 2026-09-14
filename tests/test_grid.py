from __future__ import annotations

import numpy as np
import pytest

from dummy_mlwp.errors import InputError
from dummy_mlwp.grid import detect_coords, validate_grid


def test_detects_projected_grid(make_dataset):
    coords = detect_coords(make_dataset(kind="projected"))
    assert (coords.time, coords.y, coords.x) == ("time", "y", "x")
    assert coords.kind == "projected"


def test_detects_latlon_grid(make_dataset):
    coords = detect_coords(make_dataset(kind="latlon"))
    assert (coords.time, coords.y, coords.x) == ("time", "latitude", "longitude")
    assert coords.kind == "latlon"
    assert coords.horizontal == ("latitude", "longitude")


def test_detects_grid_without_cf_attributes(make_dataset):
    ds = make_dataset(kind="projected")
    for name in ("time", "y", "x"):
        ds[name].attrs.clear()
    coords = detect_coords(ds)
    assert (coords.time, coords.y, coords.x) == ("time", "y", "x")


def test_detects_renamed_time_coordinate(make_dataset):
    ds = make_dataset().rename({"time": "valid_time"})
    assert detect_coords(ds).time == "valid_time"


def test_overrides_win_over_detection(make_dataset):
    ds = make_dataset(kind="latlon").rename({"latitude": "lat", "longitude": "lon"})
    coords = detect_coords(ds, y="lon", x="lat")
    assert (coords.y, coords.x) == ("lon", "lat")


def test_override_naming_a_missing_variable_is_an_error(make_dataset):
    with pytest.raises(InputError, match="X_COORD='easting'"):
        detect_coords(make_dataset(), x="easting")


def test_unidentifiable_coordinate_names_the_env_var(make_dataset):
    ds = make_dataset(kind="projected").rename({"x": "column"})
    ds["column"].attrs.clear()
    with pytest.raises(InputError, match="Set X_COORD explicitly"):
        detect_coords(ds)


def test_regular_grid_passes_validation(make_dataset):
    ds = make_dataset()
    assert validate_grid(ds, detect_coords(ds)) == []


def test_irregular_spacing_is_reported(make_dataset):
    ds = make_dataset()
    coords = detect_coords(ds)
    stretched = ds.x.values.copy()
    stretched[5:] += 1000.0
    ds = ds.assign_coords(x=stretched)

    (problem,) = validate_grid(ds, coords)
    assert "not evenly spaced" in problem
    assert "index 4 and 5" in problem


def test_non_monotonic_axis_is_reported(make_dataset):
    ds = make_dataset()
    coords = detect_coords(ds)
    shuffled = ds.y.values.copy()
    shuffled[[2, 7]] = shuffled[[7, 2]]
    ds = ds.assign_coords(y=shuffled)

    (problem,) = validate_grid(ds, coords)
    assert "not monotonic" in problem


def test_repeated_coordinate_values_are_reported(make_dataset):
    ds = make_dataset()
    coords = detect_coords(ds)
    repeated = ds.x.values.copy()
    repeated[3] = repeated[2]
    ds = ds.assign_coords(x=repeated)

    (problem,) = validate_grid(ds, coords)
    assert "repeated values" in problem


def test_descending_axis_is_accepted(make_dataset):
    """Latitude descending north-to-south is normal; only irregularity is a problem."""
    ds = make_dataset(kind="latlon")
    coords = detect_coords(ds)
    ds = ds.assign_coords(latitude=ds.latitude.values[::-1])
    assert validate_grid(ds, coords) == []


def test_curvilinear_grid_is_rejected(make_dataset):
    ds = make_dataset()
    coords = detect_coords(ds)
    two_d = np.tile(ds.x.values, (ds.sizes["y"], 1))
    ds = ds.assign_coords(x=(("y", "x"), two_d))

    (problem,) = validate_grid(ds, coords)
    assert "curvilinear grids are not supported" in problem
