"""Opening the input store and asserting it matches the configuration.

Validation collects *every* problem before raising, so one run of a misconfigured
pipeline reports all of it rather than one failure per debugging cycle.
"""

from __future__ import annotations

import fsspec
import numpy as np
import xarray as xr
from loguru import logger

from .config import Config
from .errors import InputError, format_problems
from .grid import CoordNames, validate_grid
from .timeaxis import validate_times
from .varspec import VarSpec, channel_layout


def open_input(uri: str) -> xr.Dataset:
    """Open a zarr store, local or remote (``s3://``, ``gs://``, ... via fsspec)."""
    logger.info(f"Opening input store {uri}")
    try:
        # No chunks= here: the whole field stack is loaded into a numpy array anyway,
        # and asking for chunks would drag in a dask dependency for no benefit.
        return xr.open_dataset(uri, engine="zarr", decode_timedelta=True)
    except (FileNotFoundError, KeyError, ValueError) as exc:
        raise InputError(f"Could not open input store {uri!r}: {exc}") from exc


def detect_zarr_format(uri: str) -> int | None:
    """Detect whether a store is zarr format 2 or 3, so the output can match it."""
    try:
        fs, path = fsspec.core.url_to_fs(uri)
        if fs.exists(f"{path.rstrip('/')}/zarr.json"):
            return 3
        if fs.exists(f"{path.rstrip('/')}/.zgroup"):
            return 2
    except Exception as exc:  # noqa: BLE001 - detection is best-effort by design
        logger.warning(f"Could not detect the zarr format of {uri} ({exc}); defaulting")
    return None


def validate_input(ds: xr.Dataset, config: Config, coords: CoordNames) -> None:
    """Assert the input matches INPUT_VARIABLES, the grid assumption and the time axis."""
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


def _referenced_level_coords(config: Config) -> set[str]:
    return {s.level_coord for s in config.input_variables if s.level_coord is not None}


def _validate_level_coords(ds: xr.Dataset, config: Config) -> list[str]:
    problems: list[str] = []
    for name in sorted(_referenced_level_coords(config)):
        expected = config.level_coords[name]
        if name not in ds.variables:
            problems.append(
                f"level coordinate {name!r} (from LEVEL_COORDS) is not present in the input"
            )
            continue
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
    return problems


def _fmt(values: np.ndarray) -> str:
    return "[" + ", ".join(f"{v:g}" for v in np.atleast_1d(values)) + "]"


def stack_channels(
    ds: xr.Dataset, specs: list[VarSpec], config: Config, coords: CoordNames
) -> np.ndarray:
    """Pack the declared variables into a ``(time, channel, y, x)`` float32 array.

    Channel order follows :func:`varspec.channel_layout`, which is also what the output
    side uses to unpack — that shared ordering is what keeps the two halves in step.
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
    """Names of CRS/grid-mapping variables referenced by the input variables.

    Copying these through means the projection survives into the output store, which
    matters for anything downstream that reprojects or plots the result.
    """
    names: list[str] = []
    for spec in specs:
        if spec.name not in ds.variables:
            continue
        mapping = ds[spec.name].attrs.get("grid_mapping")
        if mapping and mapping in ds.variables and mapping not in names:
            names.append(str(mapping))
    return names
