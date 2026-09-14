"""Generate a synthetic input zarr store for exercising the dummy model.

Examples::

    python scripts/make_test_input.py /tmp/in.zarr --kind latlon
    python scripts/make_test_input.py /tmp/in.zarr --kind projected --levels 850 500 250
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import xarray as xr

SURFACE_VARS = {
    "t2m": ("K", 283.0, 8.0, "air_temperature"),
    "u10": ("m s-1", 0.0, 5.0, "eastward_wind"),
    "v10": ("m s-1", 0.0, 5.0, "northward_wind"),
}
UPPER_AIR_VARS = {"t": ("K", 250.0, 12.0, "air_temperature")}


def build(
    kind: str,
    nt: int,
    ny: int,
    nx: int,
    levels: list[float] | None,
    freq: str,
    seed: int,
) -> xr.Dataset:
    rng = np.random.default_rng(seed)
    times = pd.date_range("2024-01-01T00:00", periods=nt, freq=freq)

    if kind == "latlon":
        y = np.linspace(54.0, 58.0, ny, dtype="float64")
        x = np.linspace(7.0, 15.0, nx, dtype="float64")
        y_attrs = {"standard_name": "latitude", "units": "degrees_north", "axis": "Y"}
        x_attrs = {"standard_name": "longitude", "units": "degrees_east", "axis": "X"}
        y_name, x_name = "latitude", "longitude"
    else:
        y = np.arange(ny, dtype="float64") * 2500.0
        x = np.arange(nx, dtype="float64") * 2500.0
        y_attrs = {"standard_name": "projection_y_coordinate", "units": "m", "axis": "Y"}
        x_attrs = {"standard_name": "projection_x_coordinate", "units": "m", "axis": "X"}
        y_name, x_name = "y", "x"

    coords = {
        "time": ("time", times, {"standard_name": "time"}),
        y_name: (y_name, y, y_attrs),
        x_name: (x_name, x, x_attrs),
    }
    if levels:
        coords["isobaricInhPa"] = (
            "isobaricInhPa",
            np.array(levels, dtype="int32"),
            {"standard_name": "air_pressure", "units": "hPa", "positive": "down"},
        )

    data_vars = {}
    for name, (units, mean, spread, standard_name) in SURFACE_VARS.items():
        values = _smooth_field(rng, (nt, ny, nx), mean, spread)
        data_vars[name] = (
            ("time", y_name, x_name),
            values,
            {"units": units, "standard_name": standard_name},
        )
    if levels:
        for name, (units, mean, spread, standard_name) in UPPER_AIR_VARS.items():
            values = _smooth_field(rng, (nt, len(levels), ny, nx), mean, spread)
            data_vars[name] = (
                ("time", "isobaricInhPa", y_name, x_name),
                values,
                {"units": units, "standard_name": standard_name},
            )

    ds = xr.Dataset(data_vars=data_vars, coords=coords)

    if kind == "projected":
        ds["crs"] = xr.DataArray(
            np.int32(0),
            attrs={
                "grid_mapping_name": "lambert_conformal_conic",
                "standard_parallel": [55.0, 55.0],
                "longitude_of_central_meridian": 10.0,
                "latitude_of_projection_origin": 55.0,
            },
        )
        for name in ds.data_vars:
            if name != "crs":
                ds[name].attrs["grid_mapping"] = "crs"

    ds.attrs = {"Conventions": "CF-1.10", "title": "Synthetic input for dummy-mlwp-model"}
    return ds


def _smooth_field(
    rng: np.random.Generator, shape: tuple[int, ...], mean: float, spread: float
) -> np.ndarray:
    """Spatially correlated noise — smooth enough to look like a field when plotted."""
    field = rng.standard_normal(shape)
    for axis in (-2, -1):
        for _ in range(3):
            field = (np.roll(field, 1, axis=axis) + field + np.roll(field, -1, axis=axis)) / 3.0
    field /= field.std()
    return (mean + spread * field).astype("float32")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", help="path of the zarr store to write")
    parser.add_argument("--kind", choices=("latlon", "projected"), default="latlon")
    parser.add_argument("--nt", type=int, default=4, help="number of timesteps")
    parser.add_argument("--ny", type=int, default=64)
    parser.add_argument("--nx", type=int, default=96)
    parser.add_argument(
        "--levels",
        type=float,
        nargs="*",
        default=None,
        help="pressure levels for the 3D variable (omit for surface fields only)",
    )
    parser.add_argument("--freq", default="6h", help="time resolution, e.g. '6h' or '1h'")
    parser.add_argument("--zarr-format", type=int, choices=(2, 3), default=3)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    ds = build(args.kind, args.nt, args.ny, args.nx, args.levels, args.freq, args.seed)
    ds.to_zarr(args.output, mode="w", consolidated=True, zarr_format=args.zarr_format)
    print(f"Wrote {args.output}")
    print(ds)


if __name__ == "__main__":
    main()
