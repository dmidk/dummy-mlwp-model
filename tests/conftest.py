"""Shared fixtures: synthetic input stores built with the same generator as the script."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Reuse scripts/make_test_input.py rather than growing a second field generator.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from make_test_input import build  # noqa: E402

LEVELS = [850.0, 500.0, 250.0]


@pytest.fixture
def make_input(tmp_path):
    """Write a synthetic input store and return its path."""

    def _make(
        name: str = "in.zarr",
        kind: str = "projected",
        nt: int = 4,
        ny: int = 12,
        nx: int = 16,
        levels: list[float] | None = None,
        freq: str = "6h",
        zarr_format: int = 3,
        seed: int = 0,
    ) -> Path:
        ds = build(kind, nt, ny, nx, levels, freq, seed)
        path = tmp_path / name
        ds.to_zarr(path, mode="w", consolidated=True, zarr_format=zarr_format)
        return path

    return _make


@pytest.fixture
def make_dataset():
    """Build an in-memory synthetic dataset, for tests that never touch disk."""

    def _make(
        kind: str = "projected",
        nt: int = 4,
        ny: int = 12,
        nx: int = 16,
        levels: list[float] | None = None,
        freq: str = "6h",
    ):
        return build(kind, nt, ny, nx, levels, freq, seed=0)

    return _make


@pytest.fixture
def base_env(tmp_path, make_input):
    """Build a minimal, valid environment pointing at a freshly written input store."""
    source = make_input(levels=LEVELS)
    return {
        "INPUT_ZARR": str(source),
        "OUTPUT_ZARR": str(tmp_path / "out.zarr"),
        "INPUT_VARIABLES": "t2m,u10,v10",
        "OUTPUT_VARIABLES": "t2m:K,tp:mm",
        "DEVICE": "cpu",
        "MODEL_HIDDEN_CHANNELS": "8",
        "MODEL_LAYERS": "2",
    }
