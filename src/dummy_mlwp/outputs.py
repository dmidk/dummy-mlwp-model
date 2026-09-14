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

    The forward pass has already happened by the time this is called — the mode
    chooses what to keep, it never skips the compute. That is deliberate: the GPU
    check must not be switchable off by configuration.

    Parameters
    ----------
    config : Config
        The run configuration, for ``output_mode`` and ``constant_value``.
    predicted : numpy.ndarray
        The network's ``(time, channel, y, x)`` output.
    input_fields : numpy.ndarray
        The ``(time, channel, y, x)`` input, needed by persistence mode.
    in_layout, out_layout : list of tuple
        Channel layouts from :func:`~dummy_mlwp.varspec.channel_layout`.

    Returns
    -------
    numpy.ndarray
        An array shaped like ``predicted``. In persistence mode, output channels with
        no matching input variable and level keep the network's output, with a
        warning naming them.
    """
    if config.output_mode == "random":
        return predicted
    if config.output_mode == "zeros":
        return np.zeros_like(predicted)
    if config.output_mode == "constant":
        return np.full_like(predicted, config.constant_value)

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
    """Wrap the predicted channels in a CF-flavoured dataset on the input's grid.

    Parameters
    ----------
    config : Config
        The run configuration.
    coords : CoordNames
        Resolved coordinate names.
    input_ds : xarray.Dataset
        The input store, whose horizontal coordinates and CRS are carried through.
    fields : numpy.ndarray
        A ``(time, channel, y, x)`` array of output values.
    times, lead_times : numpy.ndarray
        Output time and lead-time coordinate values.
    reference_time : numpy.datetime64
        The analysis time.
    device : str
        Device name, recorded in the dataset attributes.

    Returns
    -------
    xarray.Dataset
        The output dataset: one variable per OUTPUT_VARIABLES entry, the input's
        horizontal coordinates verbatim, forecast time coordinates, and attributes
        recording the configuration that produced the run.

    Raises
    ------
    InputError
        If the number of predicted channels disagrees with OUTPUT_VARIABLES, which
        would mean the packing and unpacking layouts had drifted apart.
    """
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
    """Build the attributes for one output variable.

    Parameters
    ----------
    spec : VarSpec
        The output variable's declaration.

    Returns
    -------
    dict of str to str
        A ``long_name``, plus ``units`` when the spec declared them.
    """
    attrs = {"long_name": f"dummy prediction of {spec.name}"}
    if spec.units is not None:
        attrs["units"] = spec.units
    return attrs


def _coord(
    values: np.ndarray, name: str, input_ds: xr.Dataset, defaults: dict[str, str]
) -> xr.DataArray:
    """Build a coordinate with new values but the input's descriptive attributes.

    Parameters
    ----------
    values : numpy.ndarray
        The new coordinate values.
    name : str
        Coordinate name.
    input_ds : xarray.Dataset
        The input store, whose attributes are reused when it has this coordinate.
    defaults : dict of str to str
        Attributes to start from, overridden by the input's own.

    Returns
    -------
    xarray.DataArray
        The coordinate. ``units`` and ``calendar`` are dropped, since they describe
        the input's encoding rather than these values.
    """
    attrs = dict(defaults)
    if name in input_ds.variables:
        attrs.update(
            {k: v for k, v in input_ds[name].attrs.items() if k not in ("units", "calendar")}
        )
    return xr.DataArray(values, dims=(name,), attrs=attrs)


def _copy_coord(da: xr.DataArray) -> xr.DataArray:
    """Reuse an input coordinate verbatim, minus its source-store encoding.

    Parameters
    ----------
    da : xarray.DataArray
        The input coordinate.

    Returns
    -------
    xarray.DataArray
        A deep copy with empty encoding.

    Notes
    -----
    Dropping the encoding matters: carrying the input's chunking or compressor into
    the output makes ``to_zarr`` complain when they disagree with what we ask for.
    """
    copy = da.copy(deep=True)
    copy.encoding = {}
    return copy


def _level_coord(name: str, values: np.ndarray, input_ds: xr.Dataset) -> xr.DataArray:
    """Build a level coordinate, reusing the input's version when it matches.

    Parameters
    ----------
    name : str
        Level coordinate name.
    values : numpy.ndarray
        Declared level values.
    input_ds : xarray.Dataset
        The input store.

    Returns
    -------
    xarray.DataArray
        The input's coordinate when present and the same length — keeping its CF
        attributes — otherwise a fresh coordinate from the declared values.
    """
    if name in input_ds.variables and input_ds[name].shape == values.shape:
        return _copy_coord(input_ds[name])
    return xr.DataArray(values, dims=(name,), attrs={"long_name": name})


def _copy_grid_mapping(
    ds: xr.Dataset, input_ds: xr.Dataset, config: Config, var_names: list[str]
) -> xr.Dataset:
    """Carry the CRS variable through, so the output stays georeferenced.

    Parameters
    ----------
    ds : xarray.Dataset
        The output dataset being assembled.
    input_ds : xarray.Dataset
        The input store.
    config : Config
        The run configuration, for the input variable list.
    var_names : list of str
        Output variables that should reference the grid mapping.

    Returns
    -------
    xarray.Dataset
        The dataset with the CRS variable copied in and referenced. If the input uses
        several grid mappings they are all copied but none is attached, with a
        warning, since picking one would be a guess.
    """
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
    """Build the output dataset's global attributes.

    Parameters
    ----------
    config : Config
        The run configuration, summarised into the attributes.
    coords : CoordNames
        Resolved coordinate names, for the grid kind.
    device : str
        Device the run used.

    Returns
    -------
    dict of str to str
        CF-style metadata, a comment stating plainly that the values are synthetic,
        and the configuration that produced them.
    """
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
    """Write the dataset to zarr, one chunk per timestep.

    Parameters
    ----------
    ds : xarray.Dataset
        The assembled output dataset.
    config : Config
        The run configuration, for the output URI.
    zarr_format : {2, 3}
        Store format to write.
    coords : CoordNames
        Resolved coordinate names.
    """
    encoding = _chunk_encoding(ds, coords)
    logger.info(f"Writing {config.output_zarr} (zarr format {zarr_format})")
    ds.to_zarr(
        config.output_zarr,
        mode="w",
        consolidated=True,
        zarr_format=zarr_format,
        encoding=encoding,
        storage_options=config.dst_storage_options or None,
    )
    logger.info(
        f"Wrote {len(ds.data_vars)} variable(s) and {ds.sizes[coords.time]} "
        f"timestep(s) to {config.output_zarr}"
    )


def _chunk_encoding(ds: xr.Dataset, coords: CoordNames) -> dict[str, dict]:
    """Choose output chunking: one timestep per chunk, full spatial extent.

    Parameters
    ----------
    ds : xarray.Dataset
        The assembled output dataset.
    coords : CoordNames
        Resolved coordinate names.

    Returns
    -------
    dict of str to dict
        A ``to_zarr`` encoding mapping, covering every time-varying data variable.
        This is the shape readers usually want: a whole field per read.
    """
    encoding: dict[str, dict] = {}
    for name, da in ds.data_vars.items():
        if coords.time not in da.dims:
            continue
        chunks = tuple(
            1 if dim == coords.time else size for dim, size in zip(da.dims, da.shape, strict=True)
        )
        encoding[str(name)] = {"chunks": chunks}
    return encoding
