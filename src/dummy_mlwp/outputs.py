"""Assembling the output dataset and writing it to zarr."""

from __future__ import annotations

import datetime as dt

import numpy as np
import xarray as xr
from loguru import logger

from .config import Config
from .errors import InputError
from .grid import CoordNames
from .inputs import find_grid_mapping_vars
from .varspec import VarSpec, channel_layout


def apply_output_mode(
    config: Config,
    predicted: np.ndarray,
    input_fields: np.ndarray,
    in_layout: list[tuple[VarSpec, int | None]],
    out_layout: list[tuple[VarSpec, int | None]],
) -> np.ndarray:
    """Decide what actually lands in the output, given the network's prediction.

    The forward pass has already happened by the time this is called — the mode chooses
    what to keep, it never skips the compute. That is deliberate: the GPU check must not
    be switchable off by configuration.
    """
    if config.output_mode == "random":
        return predicted
    if config.output_mode == "zeros":
        return np.zeros_like(predicted)
    if config.output_mode == "constant":
        return np.full_like(predicted, config.constant_value)

    # persistence: repeat the last input timestep of the matching variable/level.
    by_channel = {(spec.name, level): i for i, (spec, level) in enumerate(in_layout)}
    result = predicted.copy()
    missing: list[str] = []
    for out_channel, (spec, level) in enumerate(out_layout):
        source = by_channel.get((spec.name, level))
        if source is None:
            missing.append(str(spec))
            continue
        result[:, out_channel] = input_fields[-1, source]
    if missing:
        logger.warning(
            f"OUTPUT_MODE=persistence: {', '.join(sorted(set(missing)))} not present in "
            "the input with a matching level; using the network output for those"
        )
    return result


def build_output_dataset(
    config: Config,
    coords: CoordNames,
    input_ds: xr.Dataset,
    fields: np.ndarray,
    times: np.ndarray,
    lead_times: np.ndarray,
    reference_time: np.datetime64,
    device: str,
) -> xr.Dataset:
    """Wrap the predicted channels in a CF-flavoured dataset on the input's grid."""
    layout = channel_layout(config.output_variables, config.level_coords)
    if fields.shape[1] != len(layout):
        raise InputError(
            f"Internal channel mismatch: {fields.shape[1]} predicted channels but "
            f"{len(layout)} declared by OUTPUT_VARIABLES"
        )

    y_name, x_name = coords.y, coords.x
    out_coords: dict[str, xr.DataArray] = {
        coords.time: _coord(times, coords.time, input_ds, {"standard_name": "time"}),
        y_name: _copy_coord(input_ds[y_name]),
        x_name: _copy_coord(input_ds[x_name]),
    }

    used_levels = {s.level_coord for s in config.output_variables if s.level_coord is not None}
    for name in sorted(used_levels):
        out_coords[name] = _level_coord(name, config.level_coords[name], input_ds)

    data_vars: dict[str, xr.DataArray] = {}
    channel = 0
    for spec in config.output_variables:
        n_levels = spec.n_levels(config.level_coords)
        block = fields[:, channel : channel + n_levels]
        channel += n_levels
        dims = spec.dims(coords.time, y_name, x_name)
        values = block if spec.level_coord is not None else block[:, 0]
        data_vars[spec.name] = xr.DataArray(values, dims=dims, attrs=_var_attrs(spec))

    ds = xr.Dataset(data_vars=data_vars, coords=out_coords)

    ds = ds.assign_coords(
        forecastReferenceTime=xr.DataArray(
            reference_time,
            attrs={"standard_name": "forecast_reference_time", "long_name": "analysis time"},
        ),
        leadTime=xr.DataArray(
            lead_times,
            dims=(coords.time,),
            attrs={"standard_name": "forecast_period", "long_name": "time since analysis"},
        ),
    )

    ds = _copy_grid_mapping(ds, input_ds, config, list(data_vars))
    ds.attrs = _dataset_attrs(config, coords, device)
    return ds


def _var_attrs(spec: VarSpec) -> dict[str, str]:
    attrs = {"long_name": f"dummy prediction of {spec.name}"}
    if spec.units is not None:
        attrs["units"] = spec.units
    return attrs


def _coord(
    values: np.ndarray, name: str, input_ds: xr.Dataset, defaults: dict[str, str]
) -> xr.DataArray:
    attrs = dict(defaults)
    if name in input_ds.variables:
        attrs.update(
            {k: v for k, v in input_ds[name].attrs.items() if k not in ("units", "calendar")}
        )
    return xr.DataArray(values, dims=(name,), attrs=attrs)


def _copy_coord(da: xr.DataArray) -> xr.DataArray:
    """Reuse an input coordinate verbatim, minus its source-store encoding.

    Dropping the encoding matters: carrying the input's chunking or compressor into the
    output makes to_zarr complain when they disagree with what we ask for.
    """
    copy = da.copy(deep=True)
    copy.encoding = {}
    return copy


def _level_coord(name: str, values: np.ndarray, input_ds: xr.Dataset) -> xr.DataArray:
    if name in input_ds.variables and input_ds[name].shape == values.shape:
        return _copy_coord(input_ds[name])
    return xr.DataArray(values, dims=(name,), attrs={"long_name": name})


def _copy_grid_mapping(
    ds: xr.Dataset, input_ds: xr.Dataset, config: Config, var_names: list[str]
) -> xr.Dataset:
    """Carry the CRS variable through, so the output stays georeferenced."""
    mappings = find_grid_mapping_vars(input_ds, config.input_variables)
    if not mappings:
        return ds
    for name in mappings:
        crs = input_ds[name].copy(deep=True)
        crs.encoding = {}
        ds = ds.assign({name: crs})
    if len(mappings) == 1:
        for var in var_names:
            ds[var].attrs["grid_mapping"] = mappings[0]
    else:
        logger.warning(
            f"Input references several grid mappings ({', '.join(mappings)}); copying "
            "them through but not attaching one to the output variables"
        )
    return ds


def _dataset_attrs(config: Config, coords: CoordNames, device: str) -> dict[str, str]:
    from . import __version__

    return {
        "title": "Dummy MLWP model output",
        "institution": "DMI",
        "source": f"dummy-mlwp-model {__version__} (not a real forecast)",
        "comment": (
            "Synthetic output from a dummy model used to exercise infrastructure. "
            "These values have no meteorological meaning."
        ),
        "Conventions": "CF-1.10",
        "history": f"{dt.datetime.now(dt.UTC).isoformat(timespec='seconds')}: created by "
        f"dummy-mlwp-model {__version__}",
        "grid_type": coords.kind,
        "device": device,
        **config.provenance(),
    }


def write_output(ds: xr.Dataset, config: Config, zarr_format: int, coords: CoordNames) -> None:
    """Write the dataset to zarr, one chunk per timestep."""
    encoding = _chunk_encoding(ds, coords)
    logger.info(f"Writing {config.output_zarr} (zarr format {zarr_format})")
    ds.to_zarr(
        config.output_zarr,
        mode="w",
        consolidated=True,
        zarr_format=zarr_format,
        encoding=encoding,
    )
    logger.info(
        f"Wrote {len(ds.data_vars)} variable(s) and {ds.sizes[coords.time]} "
        f"timestep(s) to {config.output_zarr}"
    )


def _chunk_encoding(ds: xr.Dataset, coords: CoordNames) -> dict[str, dict]:
    """One timestep per chunk, full spatial extent — the shape readers usually want."""
    encoding: dict[str, dict] = {}
    for name, da in ds.data_vars.items():
        if coords.time not in da.dims:
            continue
        chunks = tuple(
            1 if dim == coords.time else size for dim, size in zip(da.dims, da.shape, strict=True)
        )
        encoding[str(name)] = {"chunks": chunks}
    return encoding
