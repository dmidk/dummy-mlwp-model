from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from dummy_mlwp.errors import InputError
from dummy_mlwp.grid import detect_coords, is_vertical_coord, validate_grid


def test_detects_projected_grid(make_dataset):
    coords = detect_coords(make_dataset(kind="projected"))
    assert (coords.time, coords.y, coords.x) == ("time", "y", "x")
    assert coords.kind == "projected"


def test_detects_latlon_grid(make_dataset):
    coords = detect_coords(make_dataset(kind="latlon"))
    assert (coords.time, coords.y, coords.x) == ("time", "latitude", "longitude")
    assert coords.kind == "latlon"
    assert coords.horizontal == ("latitude", "longitude")


def test_grid_without_cf_attributes_is_rejected(make_dataset):
    """Obvious names are not enough: a coordinate must say what it is."""
    ds = make_dataset(kind="projected")
    for name in ("time", "y", "x"):
        ds[name].attrs.clear()
    with pytest.raises(InputError) as excinfo:
        detect_coords(ds)
    message = str(excinfo.value)
    for axis in ("time", "y", "x"):
        assert f"identified as the {axis} coordinate" in message


def test_coordinates_are_found_by_attributes_not_names(make_dataset):
    ds = make_dataset(kind="projected").rename({"time": "when", "y": "northing", "x": "easting"})
    coords = detect_coords(ds)
    assert (coords.time, coords.y, coords.x) == ("when", "northing", "easting")


def test_axis_attribute_alone_identifies_a_coordinate(make_dataset):
    ds = make_dataset(kind="projected")
    ds["x"].attrs = {"axis": "X"}
    assert detect_coords(ds).x == "x"


def test_latlon_identified_by_units_alone(make_dataset):
    ds = make_dataset(kind="latlon")
    ds["latitude"].attrs = {"units": "degrees_north"}
    ds["longitude"].attrs = {"units": "degrees_east"}
    coords = detect_coords(ds)
    assert (coords.y, coords.x, coords.kind) == ("latitude", "longitude", "latlon")


def test_time_identified_by_cf_units_alone(make_input):
    """On decode, xarray moves CF time units into .encoding; that still counts."""
    ds = xr.open_zarr(make_input())
    ds["time"].attrs.clear()
    assert ds["time"].encoding["units"].startswith("hours since")
    assert detect_coords(ds).time == "time"


def test_projected_axes_win_over_auxiliary_latlon(make_dataset):
    ds = make_dataset(kind="projected")
    shape = (ds.sizes["y"], ds.sizes["x"])
    ds = ds.assign_coords(
        lat=(("y", "x"), np.zeros(shape), {"standard_name": "latitude"}),
        lon=(("y", "x"), np.zeros(shape), {"standard_name": "longitude"}),
    )
    coords = detect_coords(ds)
    assert (coords.y, coords.x, coords.kind) == ("y", "x", "projected")


def test_two_time_coordinates_are_ambiguous(make_dataset):
    ds = make_dataset()
    ds = ds.assign_coords(valid_time=("time", ds.time.values, {"standard_name": "time"}))
    with pytest.raises(InputError, match="Set TIME_COORD to pick one"):
        detect_coords(ds)
    assert detect_coords(ds, time="valid_time").time == "valid_time"


def test_override_must_be_cf_identified_as_that_axis(make_dataset):
    """An override picks between CF candidates; it cannot vouch for a coordinate."""
    ds = make_dataset(kind="latlon")
    with pytest.raises(InputError, match="X_COORD='latitude' but its attributes"):
        detect_coords(ds, x="latitude")


def test_override_naming_a_missing_variable_is_an_error(make_dataset):
    with pytest.raises(InputError, match="X_COORD='easting'"):
        detect_coords(make_dataset(), x="easting")


def test_unidentifiable_coordinate_names_what_is_needed(make_dataset):
    ds = make_dataset(kind="projected")
    ds["x"].attrs = {"long_name": "column"}
    with pytest.raises(InputError, match="axis='X'"):
        detect_coords(ds)


def test_vertical_coordinate_identification(make_dataset):
    ds = make_dataset(levels=[850.0, 500.0])
    assert is_vertical_coord(ds, "isobaricInhPa")
    ds["isobaricInhPa"].attrs = {"units": "hPa"}
    assert is_vertical_coord(ds, "isobaricInhPa")
    ds["isobaricInhPa"].attrs = {"positive": "up"}
    assert is_vertical_coord(ds, "isobaricInhPa")
    ds["isobaricInhPa"].attrs = {"long_name": "levels"}
    assert not is_vertical_coord(ds, "isobaricInhPa")


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
