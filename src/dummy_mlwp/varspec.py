"""The variable-specification mini-grammar used by INPUT_VARIABLES / OUTPUT_VARIABLES.

Grammar, per comma-separated entry::

    name[:units][@levelCoordName]

Examples::

    t2m                      2D field, no units attribute
    t2m:K                    2D field, units="K"
    t:K@isobaricInhPa        4D field (time, isobaricInhPa, y, x)
    u:m s-1@heightAboveGround

Level coordinates are declared once, in LEVEL_COORDS::

    isobaricInhPa:850/500/250,heightAboveGround:10/100

Everything here is pure: no xarray, no I/O. That keeps the grammar cheap to test.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np

from .errors import ConfigError

#: Coordinate and variable names are camelCase-friendly but deliberately restrictive:
#: anything that would be awkward as a zarr array name is rejected up front.
_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_.-]*$")


@dataclass(frozen=True)
class VarSpec:
    """One variable, as declared in INPUT_VARIABLES or OUTPUT_VARIABLES."""

    name: str
    units: str | None = None
    level_coord: str | None = None

    def n_levels(self, level_coords: dict[str, np.ndarray]) -> int:
        """How many 2D fields this variable occupies (1 when it has no level axis)."""
        if self.level_coord is None:
            return 1
        return len(level_coords[self.level_coord])

    def dims(self, time: str, y: str, x: str) -> tuple[str, ...]:
        """Canonical dimension order for this variable."""
        if self.level_coord is None:
            return (time, y, x)
        return (time, self.level_coord, y, x)

    def __str__(self) -> str:
        text = self.name
        if self.units is not None:
            text += f":{self.units}"
        if self.level_coord is not None:
            text += f"@{self.level_coord}"
        return text


def _check_name(name: str, what: str, source: str) -> str:
    name = name.strip()
    if not name:
        raise ConfigError(f"{source}: empty {what} in entry")
    if not _NAME_RE.match(name):
        raise ConfigError(
            f"{source}: {what} {name!r} is not a valid name "
            "(must start with a letter or underscore and contain only "
            "letters, digits, '_', '.' or '-')"
        )
    return name


def parse_level_coords(text: str, source: str = "LEVEL_COORDS") -> dict[str, np.ndarray]:
    """Parse ``name:v1/v2/v3,name2:v1/v2`` into ``{name: values}``.

    Values that are all integral are kept as integers, so a pressure-level coordinate
    comes out as 850/500/250 rather than 850.0/500.0/250.0.
    """
    coords: dict[str, np.ndarray] = {}
    for entry in _split_entries(text):
        if ":" not in entry:
            raise ConfigError(
                f"{source}: entry {entry!r} has no ':' — expected 'name:v1/v2/v3', "
                "e.g. 'isobaricInhPa:850/500/250'"
            )
        raw_name, raw_values = entry.split(":", 1)
        name = _check_name(raw_name, "level coordinate name", source)
        if name in coords:
            raise ConfigError(f"{source}: level coordinate {name!r} declared more than once")

        tokens = [t.strip() for t in raw_values.split("/") if t.strip()]
        if not tokens:
            raise ConfigError(f"{source}: level coordinate {name!r} has no values")
        coords[name] = _parse_level_values(tokens, name, source)
    return coords


def _parse_level_values(tokens: list[str], name: str, source: str) -> np.ndarray:
    try:
        values = [float(t) for t in tokens]
    except ValueError as exc:
        raise ConfigError(
            f"{source}: level coordinate {name!r} has a non-numeric value: {exc}"
        ) from exc
    if len(set(values)) != len(values):
        raise ConfigError(f"{source}: level coordinate {name!r} has duplicate values")
    if all(v.is_integer() for v in values):
        return np.array([int(v) for v in values], dtype="int32")
    return np.array(values, dtype="float64")


def parse_var_specs(
    text: str,
    level_coords: dict[str, np.ndarray],
    source: str,
) -> list[VarSpec]:
    """Parse a comma-separated list of variable specs, validating level references."""
    specs: list[VarSpec] = []
    seen: set[str] = set()
    for entry in _split_entries(text):
        spec = _parse_one(entry, level_coords, source)
        if spec.name in seen:
            raise ConfigError(f"{source}: variable {spec.name!r} listed more than once")
        seen.add(spec.name)
        specs.append(spec)
    if not specs:
        raise ConfigError(f"{source}: no variables declared")
    return specs


def _parse_one(entry: str, level_coords: dict[str, np.ndarray], source: str) -> VarSpec:
    head, _, raw_level = entry.partition("@")
    level_coord: str | None = None
    if raw_level:
        level_coord = _check_name(raw_level, "level coordinate reference", source)
        if level_coord not in level_coords:
            known = ", ".join(sorted(level_coords)) or "(none declared)"
            raise ConfigError(
                f"{source}: entry {entry!r} references level coordinate "
                f"{level_coord!r}, which is not declared in LEVEL_COORDS. Declared: {known}"
            )
    elif entry.endswith("@"):
        raise ConfigError(f"{source}: entry {entry!r} has a trailing '@' with no coordinate name")

    raw_name, sep, raw_units = head.partition(":")
    name = _check_name(raw_name, "variable name", source)
    units = raw_units.strip() if sep else None
    if sep and not units:
        raise ConfigError(f"{source}: entry {entry!r} has a ':' but no units")
    return VarSpec(name=name, units=units, level_coord=level_coord)


def _split_entries(text: str) -> list[str]:
    return [entry.strip() for entry in text.split(",") if entry.strip()]


def channel_layout(
    specs: list[VarSpec], level_coords: dict[str, np.ndarray]
) -> list[tuple[VarSpec, int | None]]:
    """Flatten specs into the 2D-field ("channel") ordering the network sees.

    A variable with three levels contributes three channels, in level order. The same
    function is used for input and output, which is what keeps the tensor packing and
    unpacking consistent.
    """
    layout: list[tuple[VarSpec, int | None]] = []
    for spec in specs:
        if spec.level_coord is None:
            layout.append((spec, None))
        else:
            layout.extend((spec, i) for i in range(len(level_coords[spec.level_coord])))
    return layout
