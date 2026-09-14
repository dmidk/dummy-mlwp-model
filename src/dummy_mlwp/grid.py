"""Coordinate discovery and the "2D regular grid" assumption, made explicit.

cf-xarray does the CF-convention work (standard_name, axis, units, positive attrs);
this module only resolves its answers into a single name per axis, applies env-var
overrides, and enforces that the horizontal grid really is regular.
"""

from __future__ import annotations

from dataclasses import dataclass

import cf_xarray  # noqa: F401  (registers the .cf accessor)
import numpy as np
import xarray as xr
from loguru import logger

from .errors import InputError

#: Relative tolerance on grid spacing. Coordinates are usually float32 degrees or
#: metres, so exact equality of successive deltas is too strict to be useful.
GRID_RTOL = 1e-3

#: Last-resort name matching, for stores with no usable CF attributes at all.
_FALLBACK_NAMES = {
    "time": ("time", "valid_time", "t"),
    "y": ("y", "latitude", "lat", "yc"),
    "x": ("x", "longitude", "lon", "xc"),
}

_LATLON_STANDARD_NAMES = {"latitude", "longitude"}


@dataclass(frozen=True)
class CoordNames:
    """The resolved coordinate names, and which kind of horizontal grid they describe."""

    time: str
    y: str
    x: str
    kind: str  # "latlon" | "projected"

    @property
    def horizontal(self) -> tuple[str, str]:
        return (self.y, self.x)


def detect_coords(
    ds: xr.Dataset,
    time: str | None = None,
    y: str | None = None,
    x: str | None = None,
) -> CoordNames:
    """Resolve the time/y/x coordinate names, preferring explicit overrides.

    Order of preference: env-var override, then cf-xarray's axes, then cf-xarray's
    coordinates, then plain-name matching. Ambiguity is an error naming both the
    candidates and the env var that settles it.
    """
    # guess_coord_axis fills in axis/standard_name attrs for common names, which lets
    # the cf accessor answer for stores that are only loosely CF-compliant.
    guessed = ds.cf.guess_coord_axis()
    axes = guessed.cf.axes
    coordinates = guessed.cf.coordinates

    resolved_time = _resolve(ds, "time", time, axes.get("T"), coordinates.get("time"))
    resolved_y = _resolve(ds, "y", y, axes.get("Y"), coordinates.get("latitude"))
    resolved_x = _resolve(ds, "x", x, axes.get("X"), coordinates.get("longitude"))

    kind = _grid_kind(ds, resolved_y, resolved_x, coordinates)
    logger.info(f"Coordinates: time={resolved_time} y={resolved_y} x={resolved_x} ({kind} grid)")
    return CoordNames(time=resolved_time, y=resolved_y, x=resolved_x, kind=kind)


def _resolve(
    ds: xr.Dataset,
    axis: str,
    override: str | None,
    from_axes: list[str] | None,
    from_coordinates: list[str] | None,
) -> str:
    env_var = f"{axis.upper()}_COORD"

    if override is not None:
        if override not in ds.variables:
            raise InputError(
                f"{env_var}={override!r} but the input has no such variable. "
                f"Available: {', '.join(sorted(map(str, ds.variables)))}"
            )
        return override

    for candidates in (from_axes, from_coordinates):
        if not candidates:
            continue
        unique = sorted({str(c) for c in candidates if c in ds.variables})
        if len(unique) == 1:
            return unique[0]
        if len(unique) > 1:
            raise InputError(
                f"Ambiguous {axis} coordinate: cf-xarray matched {', '.join(unique)}. "
                f"Set {env_var} to pick one."
            )

    for name in _FALLBACK_NAMES[axis]:
        if name in ds.variables:
            return name

    raise InputError(
        f"Could not identify the {axis} coordinate in the input. Set {env_var} explicitly. "
        f"Available: {', '.join(sorted(map(str, ds.variables)))}"
    )


def _grid_kind(ds: xr.Dataset, y_name: str, x_name: str, coordinates: dict[str, list[str]]) -> str:
    for name in (y_name, x_name):
        if ds[name].attrs.get("standard_name") in _LATLON_STANDARD_NAMES:
            return "latlon"
        if "degree" in str(ds[name].attrs.get("units", "")).lower():
            return "latlon"
    if y_name in coordinates.get("latitude", []) or x_name in coordinates.get("longitude", []):
        return "latlon"
    return "projected"


def validate_grid(ds: xr.Dataset, coords: CoordNames) -> list[str]:
    """Check the horizontal grid is 1D, monotonic and evenly spaced. Returns problems."""
    problems: list[str] = []
    for axis, name in (("y", coords.y), ("x", coords.x)):
        problems.extend(_check_axis(ds, name, axis))
    return problems


def _check_axis(ds: xr.Dataset, name: str, axis: str) -> list[str]:
    values = ds[name].values
    if values.ndim != 1:
        return [
            f"{axis} coordinate {name!r} is {values.ndim}-dimensional; this model assumes a "
            "regular 2D grid with 1D x and y coordinates (curvilinear grids are not supported)"
        ]
    if values.size < 2:
        return [f"{axis} coordinate {name!r} has {values.size} point(s); at least 2 are needed"]

    deltas = np.diff(values.astype("float64"))
    if np.any(deltas == 0):
        return [f"{axis} coordinate {name!r} has repeated values; it must be monotonic"]
    if not (np.all(deltas > 0) or np.all(deltas < 0)):
        return [f"{axis} coordinate {name!r} is not monotonic"]

    mean = deltas.mean()
    worst = int(np.argmax(np.abs(deltas - mean)))
    if np.abs(deltas[worst] - mean) > GRID_RTOL * abs(mean):
        return [
            f"{axis} coordinate {name!r} is not evenly spaced: spacing is {mean:g} on average "
            f"but {deltas[worst]:g} between index {worst} and {worst + 1} "
            f"(relative tolerance {GRID_RTOL:g})"
        ]
    return []
