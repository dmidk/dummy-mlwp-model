"""Coordinate discovery from CF attributes, and the "2D regular grid" assumption.

Every coordinate this model relies on must *say what it is* through its CF attributes —
``axis``, ``standard_name``, ``units`` or ``positive`` — and cf-xarray reads them.
Nothing is guessed from a name: a store whose ``time`` variable carries no CF metadata
is not a valid input, however obvious the name looks. The TIME_COORD / Y_COORD /
X_COORD overrides only choose *between* CF-identified candidates; they cannot vouch for
a coordinate that does not describe itself.

This module resolves cf-xarray's answers into a single name per axis, identifies
vertical coordinates, and enforces that the horizontal grid really is regular.
"""

from __future__ import annotations

from dataclasses import dataclass

import cf_xarray  # noqa: F401  (registers the .cf accessor)
import numpy as np
import xarray as xr
from loguru import logger

from .errors import InputError, format_problems

#: Relative tolerance on grid spacing. Coordinates are usually float32 degrees or
#: metres, so exact equality of successive deltas is too strict to be useful.
GRID_RTOL = 1e-3

_LATLON_STANDARD_NAMES = {"latitude", "longitude"}

#: cf-xarray keys consulted for each axis: first its ``axes`` mapping, then its
#: ``coordinates`` mapping. Axes win, so a projected store that also carries 2D
#: latitude/longitude auxiliary coordinates resolves to its projected x and y.
_CF_KEYS = {
    "time": ("T", "time"),
    "y": ("Y", "latitude"),
    "x": ("X", "longitude"),
}

#: What an axis needs to be recognised, for error messages.
_CF_REQUIREMENTS = {
    "time": "axis='T', standard_name='time', or CF time units ('<unit> since <date>')",
    "y": (
        "axis='Y', standard_name='latitude' / 'projection_y_coordinate' / "
        "'grid_latitude', or units='degrees_north'"
    ),
    "x": (
        "axis='X', standard_name='longitude' / 'projection_x_coordinate' / "
        "'grid_longitude', or units='degrees_east'"
    ),
}

#: Units by which CF identifies a vertical coordinate even without ``positive``.
_PRESSURE_UNITS = {"pa", "hpa", "kpa", "mbar", "millibar", "bar", "decibar", "dbar", "atm"}


@dataclass(frozen=True)
class CoordNames:
    """The resolved coordinate names, and which kind of horizontal grid they describe.

    Parameters
    ----------
    time : str
        Name of the time coordinate.
    y : str
        Name of the northward coordinate (latitude, or projected y).
    x : str
        Name of the eastward coordinate (longitude, or projected x).
    kind : {'latlon', 'projected'}
        Which kind of horizontal grid the coordinates describe.
    """

    time: str
    y: str
    x: str
    kind: str

    @property
    def horizontal(self) -> tuple[str, str]:
        """Return the horizontal coordinate names as a ``(y, x)`` pair."""
        return (self.y, self.x)


def detect_coords(
    ds: xr.Dataset,
    time: str | None = None,
    y: str | None = None,
    x: str | None = None,
) -> CoordNames:
    """Resolve the time/y/x coordinate names from their CF attributes.

    Parameters
    ----------
    ds : xarray.Dataset
        The opened input store.
    time, y, x : str or None, optional
        Explicit choices from TIME_COORD / Y_COORD / X_COORD. ``None`` means detect.
        A choice must still be a coordinate CF identifies as that axis.

    Returns
    -------
    CoordNames
        The resolved names and grid kind.

    Raises
    ------
    InputError
        If no coordinate is CF-identified as one of the axes, if more than one is and
        no override picks between them, or if an override names a variable that is
        absent or not CF-identified as that axis. Every axis is resolved before
        raising, so a store missing all its CF metadata reports all three at once.
    """
    candidates = cf_axis_candidates(ds)
    problems: list[str] = []
    resolved: dict[str, str] = {}
    for axis, override in (("time", time), ("y", y), ("x", x)):
        try:
            resolved[axis] = _resolve(ds, axis, override, candidates[axis])
        except InputError as exc:
            problems.append(str(exc))
    if problems:
        if len(problems) == 1:
            raise InputError(problems[0])
        raise InputError(format_problems("Could not resolve the input's coordinates:", problems))

    kind = _grid_kind(ds, resolved["y"], resolved["x"], ds.cf.coordinates)
    logger.info(
        f"Coordinates (from CF attributes): time={resolved['time']} y={resolved['y']} "
        f"x={resolved['x']} ({kind} grid)"
    )
    return CoordNames(time=resolved["time"], y=resolved["y"], x=resolved["x"], kind=kind)


def cf_axis_candidates(ds: xr.Dataset) -> dict[str, list[list[str]]]:
    """List the coordinates CF attributes identify as each of time, y and x.

    Parameters
    ----------
    ds : xarray.Dataset
        The dataset to inspect.

    Returns
    -------
    dict of str to list of list of str
        For each axis, candidate groups in order of preference: cf-xarray's ``axes``
        match, then its ``coordinates`` match. For time, a datetime-typed variable
        decoded from CF time units (``'hours since ...'``, which xarray moves into
        ``.encoding``) is added to the second group, since CF identifies time by its
        units alone.
    """
    axes = ds.cf.axes
    coordinates = ds.cf.coordinates
    out: dict[str, list[list[str]]] = {}
    for axis, (axis_key, coord_key) in _CF_KEYS.items():
        out[axis] = [
            sorted({str(c) for c in axes.get(axis_key, []) if c in ds.variables}),
            sorted({str(c) for c in coordinates.get(coord_key, []) if c in ds.variables}),
        ]
    decoded_time = [
        str(name)
        for name, var in ds.variables.items()
        if np.issubdtype(var.dtype, np.datetime64)
        and " since " in str(var.encoding.get("units", var.attrs.get("units", "")))
    ]
    out["time"][1] = sorted(set(out["time"][1]) | set(decoded_time))
    return out


def _resolve(ds: xr.Dataset, axis: str, override: str | None, candidates: list[list[str]]) -> str:
    """Resolve one axis to a single CF-identified coordinate name.

    Parameters
    ----------
    ds : xarray.Dataset
        The opened input store.
    axis : {'time', 'y', 'x'}
        Which axis is being resolved; also determines the env var named in errors.
    override : str or None
        Explicit choice, which wins when set — provided CF identifies it as ``axis``.
    candidates : list of list of str
        Candidate groups from :func:`cf_axis_candidates`, most preferred first.

    Returns
    -------
    str
        The resolved coordinate name.

    Raises
    ------
    InputError
        If the override is absent or not CF-identified, the candidates are ambiguous,
        or nothing is CF-identified as this axis.
    """
    env_var = f"{axis.upper()}_COORD"
    identified = sorted({name for group in candidates for name in group})

    if override is not None:
        if override not in ds.variables:
            raise InputError(
                f"{env_var}={override!r} but the input has no such variable. "
                f"Available: {', '.join(sorted(map(str, ds.variables)))}"
            )
        if override not in identified:
            raise InputError(
                f"{env_var}={override!r} but its attributes do not identify it as the "
                f"{axis} coordinate ({_describe(ds, override)}); it needs "
                f"{_CF_REQUIREMENTS[axis]}"
            )
        return override

    for group in candidates:
        if len(group) == 1:
            return group[0]
        if len(group) > 1:
            raise InputError(
                f"Ambiguous {axis} coordinate: CF attributes identify {', '.join(group)}. "
                f"Set {env_var} to pick one."
            )

    raise InputError(
        f"No coordinate in the input is identified as the {axis} coordinate by its CF "
        f"attributes; it needs {_CF_REQUIREMENTS[axis]}. Coordinates present: "
        f"{'; '.join(_describe(ds, str(c)) for c in sorted(map(str, ds.coords))) or '(none)'}"
    )


def _describe(ds: xr.Dataset, name: str) -> str:
    """Summarise a variable's CF-relevant attributes for an error message.

    Parameters
    ----------
    ds : xarray.Dataset
        The dataset holding the variable.
    name : str
        Variable name.

    Returns
    -------
    str
        ``name`` followed by whichever of ``axis``, ``standard_name``, ``units`` and
        ``positive`` it carries, or ``"no CF attributes"``.
    """
    attrs = ds[name].attrs
    shown = [
        f"{k}={attrs[k]!r}" for k in ("axis", "standard_name", "units", "positive") if k in attrs
    ]
    return f"{name!r} ({', '.join(shown) if shown else 'no CF attributes'})"


def is_vertical_coord(ds: xr.Dataset, name: str) -> bool:
    """Say whether CF attributes identify a variable as a vertical coordinate.

    Parameters
    ----------
    ds : xarray.Dataset
        The dataset holding the variable.
    name : str
        Variable name.

    Returns
    -------
    bool
        ``True`` when cf-xarray matches it as ``Z`` or ``vertical`` (``axis='Z'``, a
        vertical ``standard_name``, or a ``positive`` attribute), or its units are a
        pressure, which CF accepts as identifying on their own.
    """
    found = set(ds.cf.axes.get("Z", [])) | set(ds.cf.coordinates.get("vertical", []))
    if name in found:
        return True
    return str(ds[name].attrs.get("units", "")).strip().lower() in _PRESSURE_UNITS


def _grid_kind(ds: xr.Dataset, y_name: str, x_name: str, coordinates: dict[str, list[str]]) -> str:
    """Decide whether the horizontal grid is geographic or projected.

    Parameters
    ----------
    ds : xarray.Dataset
        The opened input store.
    y_name, x_name : str
        Resolved horizontal coordinate names.
    coordinates : dict of str to list of str
        cf-xarray's coordinate mapping.

    Returns
    -------
    {'latlon', 'projected'}
        ``'latlon'`` when the coordinates carry latitude/longitude standard names,
        degree units, or are identified as such by cf-xarray.
    """
    for name in (y_name, x_name):
        if ds[name].attrs.get("standard_name") in _LATLON_STANDARD_NAMES:
            return "latlon"
        if "degree" in str(ds[name].attrs.get("units", "")).lower():
            return "latlon"
    if y_name in coordinates.get("latitude", []) or x_name in coordinates.get("longitude", []):
        return "latlon"
    return "projected"


def validate_grid(ds: xr.Dataset, coords: CoordNames) -> list[str]:
    """Check the horizontal grid is 1D, monotonic and evenly spaced.

    Parameters
    ----------
    ds : xarray.Dataset
        The opened input store.
    coords : CoordNames
        Resolved coordinate names.

    Returns
    -------
    list of str
        One message per problem found; empty when the grid is regular. Problems are
        returned rather than raised so the caller can report every failure at once.
    """
    problems: list[str] = []
    for axis, name in (("y", coords.y), ("x", coords.x)):
        problems.extend(_check_axis(ds, name, axis))
    return problems


def _check_axis(ds: xr.Dataset, name: str, axis: str) -> list[str]:
    """Check one horizontal axis against the regular-grid assumption.

    Parameters
    ----------
    ds : xarray.Dataset
        The opened input store.
    name : str
        Coordinate name to check.
    axis : {'y', 'x'}
        Which axis this is, for the message text.

    Returns
    -------
    list of str
        At most one message, describing the first problem found: not 1D, too short,
        repeated values, not monotonic, or unevenly spaced. Uneven spacing reports the
        offending index and the conflicting delta.
    """
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
