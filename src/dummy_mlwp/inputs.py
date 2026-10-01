"""Opening the input store and asserting it matches the configuration.

Validation collects *every* problem before raising, so one run of a misconfigured
pipeline reports all of it rather than one failure per debugging cycle.
"""

from __future__ import annotations

from typing import Any

import fsspec
import numpy as np
import xarray as xr
from loguru import logger

from .config import Config
from .errors import InputError, format_problems
from .grid import CoordNames, is_vertical_coord, validate_grid
from .timeaxis import validate_times
from .varspec import VarSpec, channel_layout


def open_input(uri: str, storage_options: dict[str, Any] | None = None) -> xr.Dataset:
    """Open a zarr store, local or remote.

    Parameters
    ----------
    uri : str
        Store location. Local paths, ``s3://`` and ``gs://`` all work, the latter two
        through fsspec.
    storage_options : dict, optional
        fsspec options for this store — credentials, profile, endpoint. ``None`` or an
        empty dict leaves fsspec to pick up ambient environment credentials.

    Returns
    -------
    xarray.Dataset
        The opened store, lazily loaded.

    Raises
    ------
    InputError
        If the store is missing or cannot be read as zarr.

    Notes
    -----
    No ``chunks=`` is requested: the whole field stack is loaded into a numpy array
    anyway, and asking for chunks would drag in a dask dependency for no benefit.
    """
    logger.info(f"Opening input store {uri}")
    try:
        return xr.open_dataset(
            uri,
            engine="zarr",
            decode_timedelta=True,
            storage_options=storage_options or None,
        )
    except (FileNotFoundError, KeyError, ValueError) as exc:
        raise InputError(f"Could not open input store {uri!r}: {exc}") from exc


def detect_zarr_format(uri: str, storage_options: dict[str, Any] | None = None) -> int | None:
    """Detect whether a store is zarr format 2 or 3, so the output can match it.

    Parameters
    ----------
    uri : str
        Store location.
    storage_options : dict, optional
        fsspec options for this store.

    Returns
    -------
    int or None
        ``3`` if the store has a ``zarr.json``, ``2`` if it has a ``.zgroup``, and
        ``None`` when neither is found or the store cannot be inspected. Detection is
        best-effort by design: a failure here should not fail the run.
    """
    try:
        fs, path = fsspec.core.url_to_fs(uri, **(storage_options or {}))
        if fs.exists(f"{path.rstrip('/')}/zarr.json"):
            return 3
        if fs.exists(f"{path.rstrip('/')}/.zgroup"):
            return 2
    except Exception as exc:  # noqa: BLE001 - detection is best-effort by design
        logger.warning(f"Could not detect the zarr format of {uri} ({exc}); defaulting")
    return None


def validate_input(ds: xr.Dataset, config: Config, coords: CoordNames) -> None:
    """Assert the input matches INPUT_VARIABLES, the grid assumption and the time axis.

    This includes the CF metadata: level coordinates must identify themselves as
    vertical, and any ``standard_name`` or ``units`` declared in INPUT_VARIABLES must
    match the variable's attributes.

    Parameters
    ----------
    ds : xarray.Dataset
        The opened input store.
    config : Config
        The run configuration.
    coords : CoordNames
        Resolved coordinate names.

    Raises
    ------
    InputError
        If anything does not match. Every check runs first, so the message lists all
        the problems at once rather than stopping at the first.
    """
    problems: list[str] = []
    problems.extend(validate_grid(ds, coords))
    problems.extend(validate_times(ds[coords.time].values))
    problems.extend(_validate_level_coords(ds, config))
    problems.extend(_validate_variables(ds, config, coords))

    if problems:
        raise InputError(
            format_problems(
                f"Input store {config.input_zarr!r} does not match the configured expectations:",
                problems,
            )
        )
    logger.info(
        f"Input validated: {len(config.input_variables)} variable(s), "
        f"{config.n_input_channels} channel(s), {ds.sizes[coords.time]} timestep(s), "
        f"grid {ds.sizes[coords.y]}x{ds.sizes[coords.x]}"
    )


def select_input_timesteps(ds: xr.Dataset, coords: CoordNames, n: int | None) -> xr.Dataset:
    """Narrow the input to the timesteps the model should actually see.

    Parameters
    ----------
    ds : xarray.Dataset
        The opened, validated input store.
    coords : CoordNames
        Resolved coordinate names.
    n : int or None
        ``None`` for every timestep, a positive ``n`` for the first ``n``, a negative
        ``n`` for the last ``|n|``.

    Returns
    -------
    xarray.Dataset
        The input, sliced along time. Returned unchanged when ``n`` is ``None``.

    Raises
    ------
    InputError
        If the store has fewer timesteps than were asked for. Silently taking what is
        available would change the forecast's meaning without saying so.

    Notes
    -----
    Slicing a contiguous run off either end of an evenly spaced axis leaves it evenly
    spaced, so the time-axis validation done before this still holds afterwards.
    """
    if n is None:
        return ds

    available = ds.sizes[coords.time]
    if abs(n) > available:
        raise InputError(
            f"N_INPUT_TIMESTEPS={n} asks for {abs(n)} timestep(s) but the input has "
            f"only {available}"
        )

    selection = slice(0, n) if n > 0 else slice(available + n, available)
    selected = ds.isel({coords.time: selection})
    times = selected[coords.time].values
    logger.info(
        f"Using {abs(n)} of {available} input timestep(s) "
        f"({'first' if n > 0 else 'last'}): {times[0]} to {times[-1]}"
    )
    return selected


def _referenced_level_coords(config: Config) -> set[str]:
    """List the level coordinates the input variables actually use.

    Parameters
    ----------
    config : Config
        The run configuration.

    Returns
    -------
    set of str
        Level coordinate names referenced by INPUT_VARIABLES. Coordinates declared in
        LEVEL_COORDS but used only on output are not checked against the input.
    """
    return {s.level_coord for s in config.input_variables if s.level_coord is not None}


def _validate_level_coords(ds: xr.Dataset, config: Config) -> list[str]:
    """Check declared level values against the ones in the store.

    Parameters
    ----------
    ds : xarray.Dataset
        The opened input store.
    config : Config
        The run configuration.

    Returns
    -------
    list of str
        One message per level coordinate that is missing, is not identified as
        vertical by its CF attributes, or whose values differ from the declaration.
    """
    problems: list[str] = []
    for name in sorted(_referenced_level_coords(config)):
        expected = config.level_coords[name]
        if name not in ds.variables:
            problems.append(
                f"level coordinate {name!r} (from LEVEL_COORDS) is not present in the input"
            )
            continue
        if not is_vertical_coord(ds, name):
            problems.append(
                f"level coordinate {name!r} is not identified as a vertical coordinate by "
                "its CF attributes; it needs axis='Z', positive='up'/'down', a vertical "
                "standard_name such as 'air_pressure' or 'height', or pressure units"
            )
        actual = ds[name].values
        if actual.shape != expected.shape or not np.allclose(
            actual.astype("float64"), expected.astype("float64")
        ):
            problems.append(
                f"level coordinate {name!r} is {_fmt(actual)} in the input but "
                f"LEVEL_COORDS declares {_fmt(expected)}"
            )
    return problems


def _validate_variables(ds: xr.Dataset, config: Config, coords: CoordNames) -> list[str]:
    """Check each declared input variable exists with the implied dimensions and attributes.

    Dimension *order* is not checked — a store may hold ``(x, time, y)`` and it is
    transposed later — but the set of dimensions must match exactly. A declared
    ``standard_name`` or ``units`` must match the variable's attribute exactly; units
    are compared as strings, so ``'m s-1'`` and ``'m/s'`` are different on purpose.

    Parameters
    ----------
    ds : xarray.Dataset
        The opened input store.
    config : Config
        The run configuration.
    coords : CoordNames
        Resolved coordinate names.

    Returns
    -------
    list of str
        One message per missing variable, dimension mismatch, or CF attribute that is
        missing or differs from the declaration.
    """
    problems: list[str] = []
    for spec in config.input_variables:
        if spec.name not in ds.data_vars:
            problems.append(
                f"variable {spec.name!r} (from INPUT_VARIABLES) is not in the input; "
                f"available: {', '.join(sorted(map(str, ds.data_vars)))}"
            )
            continue

        expected = set(spec.dims(coords.time, coords.y, coords.x))
        actual = set(map(str, ds[spec.name].dims))
        if actual != expected:
            problems.append(
                f"variable {spec.name!r} has dimensions {tuple(map(str, ds[spec.name].dims))} "
                f"but {spec} implies {spec.dims(coords.time, coords.y, coords.x)}"
            )
        problems.extend(_validate_attr(ds[spec.name], spec, "standard_name", spec.standard_name))
        problems.extend(_validate_attr(ds[spec.name], spec, "units", spec.units))
    return problems


def _validate_attr(da: xr.DataArray, spec: VarSpec, key: str, expected: str | None) -> list[str]:
    """Check one declared CF attribute of an input variable.

    Parameters
    ----------
    da : xarray.DataArray
        The input variable.
    spec : VarSpec
        Its declaration, for the message.
    key : {'standard_name', 'units'}
        Attribute to check.
    expected : str or None
        Declared value; ``None`` means nothing was declared, so nothing is checked.

    Returns
    -------
    list of str
        At most one message: the attribute is missing, or differs from ``expected``.
    """
    if expected is None:
        return []
    actual = da.attrs.get(key)
    if actual is None:
        return [f"variable {spec.name!r} has no {key} attribute but {spec} declares {expected!r}"]
    if str(actual) != expected:
        return [f"variable {spec.name!r} has {key}={actual!r} but {spec} declares {expected!r}"]
    return []


def _fmt(values: np.ndarray) -> str:
    """Format coordinate values compactly for an error message.

    Parameters
    ----------
    values : numpy.ndarray
        Values to render.

    Returns
    -------
    str
        A bracketed, comma-separated list using general numeric formatting.
    """
    return "[" + ", ".join(f"{v:g}" for v in np.atleast_1d(values)) + "]"


def stack_channels(
    ds: xr.Dataset, specs: list[VarSpec], config: Config, coords: CoordNames
) -> np.ndarray:
    """Pack the declared variables into a single dense array for the network.

    Channel order follows :func:`~dummy_mlwp.varspec.channel_layout`, which is also
    what the output side uses to unpack — that shared ordering is what keeps the two
    halves in step.

    Parameters
    ----------
    ds : xarray.Dataset
        The opened, validated input store.
    specs : list of VarSpec
        Variables to pack, in declaration order.
    config : Config
        The run configuration, for the level declarations.
    coords : CoordNames
        Resolved coordinate names.

    Returns
    -------
    numpy.ndarray
        A ``(time, channel, y, x)`` float32 array. Non-finite values are replaced with
        zero, with a warning, so they cannot poison the forward pass.
    """
    layout = channel_layout(specs, config.level_coords)
    n_time = ds.sizes[coords.time]
    shape = (n_time, len(layout), ds.sizes[coords.y], ds.sizes[coords.x])
    out = np.empty(shape, dtype="float32")

    for channel, (spec, level_index) in enumerate(layout):
        da = ds[spec.name]
        if level_index is not None:
            da = da.isel({spec.level_coord: level_index})
        da = da.transpose(coords.time, coords.y, coords.x)
        out[:, channel] = da.values.astype("float32")

    if not np.isfinite(out).all():
        n_bad = int((~np.isfinite(out)).sum())
        logger.warning(f"Input contains {n_bad} non-finite value(s); replacing them with 0")
        out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    return out


def find_grid_mapping_vars(ds: xr.Dataset, specs: list[VarSpec]) -> list[str]:
    """Find the CRS/grid-mapping variables referenced by the input variables.

    Copying these through means the projection survives into the output store, which
    matters for anything downstream that reprojects or plots the result.

    Parameters
    ----------
    ds : xarray.Dataset
        The opened input store.
    specs : list of VarSpec
        Variables whose ``grid_mapping`` attributes should be followed.

    Returns
    -------
    list of str
        Names of the referenced grid-mapping variables that exist in the store, in
        first-seen order and without duplicates.
    """
    names: list[str] = []
    for spec in specs:
        if spec.name not in ds.variables:
            continue
        mapping = ds[spec.name].attrs.get("grid_mapping")
        if mapping and mapping in ds.variables and mapping not in names:
            names.append(str(mapping))
    return names
