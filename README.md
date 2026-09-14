# dummy-mlwp-model

A containerised stand-in for a deep-learning weather-forecasting model. It reads a zarr
store, runs a real (but meteorologically meaningless) convolutional network on the GPU,
and writes a zarr store. Everything is configured through environment variables.

It exists so the infrastructure around a real MLWP model — workflow orchestration,
storage plumbing, GPU scheduling, error handling — can be built and tested without
waiting for the real model. To be useful in that role it is deliberately:

- **strict** — every expectation about the input is declared in configuration and checked,
  and each class of failure gets its own exit code;
- **honest about the GPU** — a real forward pass runs on the device for every output mode,
  so a broken CUDA setup fails here rather than three weeks later.

## Assumptions about the input

- Variables live on a **2D regular grid**, either geographic (lat/lon) or projected (x/y).
  1D horizontal coordinates, monotonic, evenly spaced.
- There is a **time axis**, strictly increasing and evenly spaced.
- Variables are `(time, y, x)`, or `(time, level, y, x)` when a level coordinate is declared.
  Dimension order in the store does not matter; it is transposed as needed.

Coordinates are discovered with [cf-xarray](https://cf-xarray.readthedocs.io/) from CF
attributes, and can be overridden when the store is not CF-compliant.

## Configuration

### Paths and variables

| Variable | Default | Meaning |
| --- | --- | --- |
| `INPUT_ZARR` | **required** | Input store URI — local path, `s3://…`, `gs://…` |
| `OUTPUT_ZARR` | **required** | Output store URI |
| `INPUT_VARIABLES` | **required** | Variables the input must contain |
| `OUTPUT_VARIABLES` | **required** | Variables to write |
| `LEVEL_COORDS` | `""` | Level coordinate declarations |

Variable spec grammar, per comma-separated entry:

```
name[:units][@levelCoordName]
```

```sh
LEVEL_COORDS=isobaricInhPa:850/500/250,heightAboveGround:10/100
INPUT_VARIABLES=t2m,u10,v10,t@isobaricInhPa
OUTPUT_VARIABLES=t2m:K,tp:mm,z:m2s-2@isobaricInhPa
```

On input the spec is an assertion: the variable must exist with exactly those dimensions,
and a referenced level coordinate must match `LEVEL_COORDS` value for value. On output it
is a construction instruction — `units` is written to the variable's attributes.

### Forecast horizon

| Variable | Default | Meaning |
| --- | --- | --- |
| `N_FORECAST_STEPS` | `-1` | `-1`: one prediction per input timestep (regression/classification framing); the output time coordinate *is* the input's. `K > 0`: `K` future steps at the input's time resolution, starting one `dt` after the last input time. |
| `FORECAST_TIMESTEP` | unset | ISO 8601 duration, e.g. `PT6H`. Only needed when `N_FORECAST_STEPS > 0` and the input has a single timestep, so `dt` cannot be inferred. |

The output always carries `forecastReferenceTime` (the last input time) and a `leadTime`
coordinate. With `N_FORECAST_STEPS=-1` lead times are zero or negative — the honest
description of a diagnostic evaluated on its own input times.

### Coordinates

| Variable | Default | Meaning |
| --- | --- | --- |
| `TIME_COORD` | auto | Override time coordinate detection |
| `X_COORD` | auto | Override x/longitude coordinate detection |
| `Y_COORD` | auto | Override y/latitude coordinate detection |

### Behaviour

| Variable | Default | Meaning |
| --- | --- | --- |
| `OUTPUT_MODE` | `random` | `random`, `persistence`, `constant`, `zeros` — see below |
| `RANDOM_SEED` | `0` | Seeds the network weights; runs are reproducible |
| `CONSTANT_VALUE` | `0.0` | Used by `constant` mode |
| `DEVICE` | `auto` | `auto`, `cuda`, `cpu`. `cuda` fails hard if no device is visible |
| `MODEL_HIDDEN_CHANNELS` | `64` | Network width — how much GPU work happens |
| `MODEL_LAYERS` | `4` | Network depth |
| `ZARR_FORMAT` | `auto` | `auto` (match the input), `2`, `3` |
| `LOG_LEVEL` | `INFO` | loguru level |

**Output modes.** The forward pass runs in *every* mode — the mode only decides what is
kept. This is deliberate: the GPU check must not be switchable off by configuration.

| Mode | Output |
| --- | --- |
| `random` | The network's output. Seeded weights over real input data, so it is deterministic, GPU-computed, and rescaled to each variable's plausible range (a predicted `t2m` lands near 273 K, not near 0). |
| `persistence` | The last input timestep of the matching variable, repeated across the output steps. Output variables with no input counterpart fall back to the network output. |
| `constant` | Every value is `CONSTANT_VALUE`. |
| `zeros` | Every value is `0`. |

With `N_FORECAST_STEPS = K > 0` the network is rolled forward autoregressively, one
forward pass per step, so wall-clock time scales with the forecast length the way a real
model does — which is the property a scheduler test actually cares about.

## Exit codes

| Code | Meaning |
| --- | --- |
| 0 | Success |
| 2 | Configuration error — a missing or malformed environment variable |
| 3 | Input error — the store does not match the configured expectations |
| 4 | Device error — a GPU was requested but is unusable |
| 1 | Anything unexpected (traceback logged) |

Input validation collects *every* problem before failing, so one run of a misconfigured
pipeline reports all of them rather than one per debugging cycle.

## Running it

### Locally

```sh
uv venv && uv pip install -e ".[dev]"
python scripts/make_test_input.py /tmp/in.zarr --kind projected --levels 850 500 250

INPUT_ZARR=/tmp/in.zarr \
OUTPUT_ZARR=/tmp/out.zarr \
LEVEL_COORDS=isobaricInhPa:850/500/250 \
INPUT_VARIABLES=t2m,u10,v10,t:K@isobaricInhPa \
OUTPUT_VARIABLES=t2m:K,tp:mm,z:m2s-2@isobaricInhPa \
N_FORECAST_STEPS=8 \
DEVICE=cpu \
python -m dummy_mlwp
```

### In a container

```sh
docker build -t dummy-mlwp:local .

docker run --rm --gpus all \
  -v /tmp:/data \
  -e INPUT_ZARR=/data/in.zarr \
  -e OUTPUT_ZARR=/data/out.zarr \
  -e INPUT_VARIABLES=t2m,u10,v10 \
  -e OUTPUT_VARIABLES=t2m:K,tp:mm \
  -e N_FORECAST_STEPS=8 \
  -e DEVICE=cuda \
  -e MODEL_HIDDEN_CHANNELS=256 \
  ghcr.io/OWNER/dummy-mlwp-model:latest
```

Prebuilt images are published to `ghcr.io/OWNER/dummy-mlwp-model` on every push to the
default branch and every `v*` tag.

To confirm the GPU is genuinely in use, look for the log line reporting the device name
and a non-zero peak GPU memory, and check that raising `MODEL_HIDDEN_CHANNELS` increases
both. Running with `DEVICE=cuda` but without `--gpus all` exits 4 rather than silently
falling back to CPU.

### Remote stores

`s3://` and `gs://` URIs work for both input and output via fsspec; install the extra
that matches your backend:

```sh
uv pip install "dummy-mlwp-model[remote]"
```

Credentials are picked up from the environment in the usual way (`AWS_*`,
`GOOGLE_APPLICATION_CREDENTIALS`).

## Versioning

The version is derived from the git tag by `hatch-vcs`. Tag a release as `v1.2.3` and
the wheel, the `dummy_mlwp.__version__` attribute, the output store's `source` attribute,
and the container tag all follow.

## Development

```sh
uv pip install -e ".[dev]"
pytest
ruff check . && ruff format --check .
```

The test suite is CPU-only and needs no container.
