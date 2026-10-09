"""ZARR_FORMAT: which zarr format, 2 or 3, the output store is written in."""

from __future__ import annotations

from pathlib import Path

import pytest
import xarray as xr
import zarr

from dummy_mlwp.__main__ import _resolve_zarr_format, main
from dummy_mlwp.config import Config
from dummy_mlwp.errors import ConfigError
from dummy_mlwp.inputs import detect_zarr_format

MINIMAL = {
    "INPUT_ZARR": "/in.zarr",
    "OUTPUT_ZARR": "/out.zarr",
    "INPUT_VARIABLES": "t2m",
    "OUTPUT_VARIABLES": "t2m:K",
}

# The root metadata file each format writes, and the one detect_zarr_format looks for.
MARKER = {2: ".zgroup", 3: "zarr.json"}


def invoke(monkeypatch, env):
    monkeypatch.delenv("ZARR_FORMAT", raising=False)
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    return main()


def run_env(source: Path, target: Path, zarr_format: str | None = None) -> dict[str, str]:
    env = {
        "INPUT_ZARR": str(source),
        "OUTPUT_ZARR": str(target),
        "INPUT_VARIABLES": "t2m,u10",
        "OUTPUT_VARIABLES": "t2m:K,tp:mm",
        "N_FORECAST_STEPS": "2",
        "DEVICE": "cpu",
        "MODEL_HIDDEN_CHANNELS": "8",
        "MODEL_LAYERS": "2",
    }
    if zarr_format is not None:
        env["ZARR_FORMAT"] = zarr_format
    return env


def written_format(path: Path) -> int:
    """Return the format a store was written in, after checking all of it agrees."""
    group = zarr.open_group(path, mode="r")
    formats = {group.metadata.zarr_format}
    formats |= {array.metadata.zarr_format for _, array in group.arrays()}
    assert len(formats) == 1, f"{path} mixes zarr formats {sorted(formats)}"
    (found,) = formats
    assert {m for m in MARKER.values() if (path / m).exists()} == {MARKER[found]}
    return found


# --- configuration -------------------------------------------------------------------


@pytest.mark.parametrize(
    ("value", "expected"), [("auto", "auto"), ("2", "2"), ("3", "3"), ("AUTO", "auto")]
)
def test_accepted_values(value, expected):
    assert Config.from_env(MINIMAL | {"ZARR_FORMAT": value}).zarr_format == expected


@pytest.mark.parametrize("value", ["1", "4", "v3", "zarr3", "3.0"])
def test_other_values_are_rejected(value):
    with pytest.raises(ConfigError, match="ZARR_FORMAT must be one of"):
        Config.from_env(MINIMAL | {"ZARR_FORMAT": value})


# --- detection and resolution --------------------------------------------------------


@pytest.mark.parametrize("zarr_format", [2, 3])
def test_detects_the_format_of_a_store(make_input, zarr_format):
    assert detect_zarr_format(str(make_input(zarr_format=zarr_format))) == zarr_format


def test_a_directory_that_is_not_a_store_is_undetected(tmp_path):
    assert detect_zarr_format(str(tmp_path)) is None


@pytest.mark.parametrize(("requested", "detected"), [("2", 3), ("3", 2)])
def test_an_explicit_format_overrides_the_input(monkeypatch, requested, detected):
    monkeypatch.setattr("dummy_mlwp.__main__.detect_zarr_format", lambda *_: detected)
    config = Config.from_env(MINIMAL | {"ZARR_FORMAT": requested})
    assert _resolve_zarr_format(config) == int(requested)


@pytest.mark.parametrize("detected", [2, 3])
def test_auto_follows_the_input(monkeypatch, detected):
    monkeypatch.setattr("dummy_mlwp.__main__.detect_zarr_format", lambda *_: detected)
    assert _resolve_zarr_format(Config.from_env(MINIMAL)) == detected


def test_auto_falls_back_to_format_3_when_detection_fails(monkeypatch):
    monkeypatch.setattr("dummy_mlwp.__main__.detect_zarr_format", lambda *_: None)
    assert _resolve_zarr_format(Config.from_env(MINIMAL)) == 3


# --- end to end ----------------------------------------------------------------------


@pytest.mark.parametrize("requested", [None, "auto", "2", "3"], ids=["unset", "auto", "2", "3"])
@pytest.mark.parametrize("source_format", [2, 3], ids=["input-v2", "input-v3"])
def test_output_format(monkeypatch, tmp_path, make_input, source_format, requested):
    """An explicit ZARR_FORMAT is written as given; otherwise the input's format is copied."""
    source = make_input(zarr_format=source_format)
    target = tmp_path / "out.zarr"
    assert invoke(monkeypatch, run_env(source, target, requested)) == 0

    expected = source_format if requested in (None, "auto") else int(requested)
    assert written_format(target) == expected

    out = xr.open_zarr(target, decode_timedelta=True)
    assert out.sizes["time"] == 2
    assert out.t2m.attrs["units"] == "K"
    assert out.t2m.encoding["chunks"][0] == 1


def test_the_format_does_not_change_what_a_reader_gets(monkeypatch, tmp_path, make_input):
    """Values, coordinates and attributes must survive either format identically."""
    source = make_input()
    outputs = {}
    for zarr_format in ("2", "3"):
        target = tmp_path / f"out-v{zarr_format}.zarr"
        assert invoke(monkeypatch, run_env(source, target, zarr_format)) == 0
        outputs[zarr_format] = xr.open_zarr(target, decode_timedelta=True).load()
        # The one attribute expected to differ between two runs: a creation timestamp.
        outputs[zarr_format].attrs.pop("history")

    xr.testing.assert_identical(outputs["2"], outputs["3"])
