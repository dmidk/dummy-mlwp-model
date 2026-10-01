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
and a referenced level coordinate must match `LEVEL_COORDS` value for value. When `units`
is given, the variable's `units` attribute must equal it exactly, compared as a plain
string with no unit parsing or normalisation: `u10:m/s` fails against a store that says
`m s-1`, and a variable with no `units` attribute fails any declared units. Leave the
units off an input entry to skip the check. On output the spec is a construction
instruction — `units` is written to the variable's attributes.

### Forecast horizon

| Variable | Default | Meaning |
| --- | --- | --- |
| `N_INPUT_TIMESTEPS` | unset (all) | How many of the input's timesteps to feed the model. Unset: all of them. `n > 0`: the **first** `n` (e.g. `2`). `n < 0`: the **last** `\|n\|` (e.g. `-2`). Asking for more than the store holds is an error, not a silent truncation. |
| `N_FORECAST_STEPS` | `-1` | `-1`: one prediction per input timestep (regression/classification framing); the output time coordinate *is* the input's. `K > 0`: `K` future steps at the input's time resolution, starting one `dt` after the last input time. |
| `FORECAST_TIMESTEP` | unset | ISO 8601 duration, e.g. `PT6H`. Only needed when `N_FORECAST_STEPS > 0` and the input has a single timestep, so `dt` cannot be inferred. |

`N_INPUT_TIMESTEPS` is applied before anything else looks at the time axis, so the
"last input time" that anchors the forecast is the last *selected* one:

```sh
# Input has 8 timesteps at 6h. Use only the last 2, then forecast 4 steps on from there.
N_INPUT_TIMESTEPS=-2 N_FORECAST_STEPS=4
```

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
  ghcr.io/dmidk/dummy-mlwp-model:latest
```

Prebuilt images are published to `ghcr.io/dmidk/dummy-mlwp-model` for version tags only
(see [Versioning](#versioning)). Pin a release, e.g. `:0.1.0` or `:0.1`; `:latest` is the
newest final release.

To confirm the GPU is genuinely in use, look for the log line reporting the device name
and a non-zero peak GPU memory, and check that raising `MODEL_HIDDEN_CHANNELS` increases
both. Running with `DEVICE=cuda` but without `--gpus all` exits 4 rather than silently
falling back to CPU.

### Remote stores

`s3://` and `gs://` URIs work for both input and output via fsspec; install the extra
that matches your backend (it is already in the container image):

```sh
uv pip install "dummy-mlwp-model[remote]"
```

**Access is anonymous unless credentials are actually given.** Reading a public bucket
is the common case for a test rig, and defaulting to signed requests turns that into a
confusing `NoCredentialsError`. Naming a profile, supplying keys, or running under an
IAM role (ECS/EKS/EC2) all count as credentials and switch signing back on. The startup
log says which mode each side ended up in.

So a public source bucket needs no configuration at all:

```sh
INPUT_ZARR=s3://public-weather-data/analysis.zarr \
OUTPUT_ZARR=/data/out.zarr \
INPUT_VARIABLES=t2m,u10,v10 OUTPUT_VARIABLES=t2m:K \
python -m dummy_mlwp
```

The input and output stores are configured **independently**, so they need not share a
host or an account. Each side reads its own variables, prefixed `SRC_` for the input
and `DST_` for the output:

| Variable | Meaning |
| --- | --- |
| `SRC_AWS_PROFILE` / `DST_AWS_PROFILE` | Named profile from `~/.aws/config` — the main knob |
| `SRC_AWS_ACCESS_KEY_ID` / `DST_AWS_ACCESS_KEY_ID` | Explicit key, when a profile is not practical |
| `SRC_AWS_SECRET_ACCESS_KEY` / `DST_…` | Matching secret |
| `SRC_AWS_SESSION_TOKEN` / `DST_…` | Session token for temporary credentials |
| `SRC_S3_ANON` / `DST_S3_ANON` | Force anonymous (`true`) or signed (`false`) access |
| `SRC_S3_ENDPOINT_URL` / `DST_S3_ENDPOINT_URL` | Endpoint override, when the AWS config is not available |
| `SRC_STORAGE_OPTIONS` / `DST_STORAGE_OPTIONS` | JSON object of any other fsspec option |

The unprefixed spellings — `AWS_PROFILE`, `AWS_ACCESS_KEY_ID`, `S3_ENDPOINT_URL` or
AWS's own `AWS_ENDPOINT_URL` — still apply to both sides, with the prefixed one winning
where both are set. `*_STORAGE_OPTIONS` is merged last, so it can override any of the
above and reach s3fs or gcsfs options that have no dedicated variable.

#### Endpoints belong in the AWS config

For an S3-compatible host that is not AWS, put the endpoint on the **profile** rather
than in the environment. Modern botocore reads `endpoint_url` straight from a profile
in `~/.aws/config`:

```ini
# ~/.aws/config
[profile dmi-minio]
region     = dk-east-1
endpoint_url = https://s3.dmi.dk
```

```ini
# ~/.aws/credentials
[dmi-minio]
aws_access_key_id     = minio-writer
aws_secret_access_key = ...
```

One profile name now carries the host, the region and the credentials together:

```sh
AWS_PROFILE=dmi-minio \
INPUT_ZARR=s3://analysis/hres.zarr \
OUTPUT_ZARR=s3://forecasts/dummy-run.zarr \
INPUT_VARIABLES=t2m,u10,v10 OUTPUT_VARIABLES=t2m:K,tp:mm \
N_FORECAST_STEPS=12 \
python -m dummy_mlwp
```

If your botocore predates per-profile `endpoint_url`, the equivalent `services` form
works too:

```ini
[services dmi]
s3 =
  endpoint_url = https://s3.dmi.dk

[profile dmi-minio]
services = dmi
region   = dk-east-1
```

#### Example 1 — a custom S3-compatible host

Both stores on the same on-premise MinIO / Ceph / DMI object store, with the endpoint
coming from the profile above. Mount `~/.aws` read-only into the container:

```sh
docker run --rm --gpus all \
  -v ~/.aws:/home/model/.aws:ro \
  -e AWS_PROFILE=dmi-minio \
  -e INPUT_ZARR=s3://analysis/hres.zarr \
  -e OUTPUT_ZARR=s3://forecasts/dummy-run.zarr \
  -e INPUT_VARIABLES=t2m,u10,v10 \
  -e OUTPUT_VARIABLES=t2m:K,tp:mm \
  -e N_FORECAST_STEPS=12 \
  -e DEVICE=cuda \
  ghcr.io/dmidk/dummy-mlwp-model:latest
```

The image runs as uid 1000 with home `/home/model`, which is why the mount goes there —
that is where botocore looks for `config` and `credentials`.

Where mounting a config file is not practical — a Kubernetes Job with only a Secret to
work with, say — give the endpoint and keys directly instead:

```sh
-e S3_ENDPOINT_URL=https://s3.dmi.dk \
-e AWS_ACCESS_KEY_ID=minio-writer \
-e AWS_SECRET_ACCESS_KEY=...
```

#### Example 2 — reading and writing across two different S3 hosts

This is what `SRC_*` and `DST_*` exist for: two profiles, each carrying its own
endpoint and credentials, so a single run reads from one host and writes to another
without the two ever sharing a credential.

```ini
# ~/.aws/config
[profile era5-reader]
region = eu-west-1
# no endpoint_url: this one is real AWS

[profile dmi-minio]
region       = dk-east-1
endpoint_url = https://s3.dmi.dk
```

```ini
# ~/.aws/credentials
[era5-reader]
aws_access_key_id     = AKIA...
aws_secret_access_key = ...

[dmi-minio]
aws_access_key_id     = minio-writer
aws_secret_access_key = ...
```

```sh
docker run --rm --gpus all \
  -v ~/.aws:/home/model/.aws:ro \
  -e SRC_AWS_PROFILE=era5-reader \
  -e INPUT_ZARR=s3://era5-source/analysis.zarr \
  -e DST_AWS_PROFILE=dmi-minio \
  -e OUTPUT_ZARR=s3://forecasts/dummy-run.zarr \
  -e INPUT_VARIABLES=t2m,u10,v10 \
  -e OUTPUT_VARIABLES=t2m:K,tp:mm \
  -e N_INPUT_TIMESTEPS=-2 \
  -e N_FORECAST_STEPS=12 \
  -e DEVICE=cuda \
  ghcr.io/dmidk/dummy-mlwp-model:latest
```

Two profile names is the whole configuration: each side resolves its own host, region
and credentials from the AWS config, exactly as the AWS CLI would.

A public source paired with an authenticated destination needs even less — anonymous is
already the default for the side with no credentials:

```sh
  -e INPUT_ZARR=s3://public-weather-data/analysis.zarr \
  -e DST_AWS_PROFILE=dmi-minio \
  -e OUTPUT_ZARR=s3://forecasts/dummy-run.zarr
```

Google Cloud Storage works the same way through `gs://` URIs, with
`GOOGLE_APPLICATION_CREDENTIALS` or `DST_STORAGE_OPTIONS='{"project":"..."}'`.

The resolved options for each side are logged at startup, with any key, secret, token
or password value masked, so you can confirm which account and host a run used.

## Versioning

The version is derived from the git tag by `hatch-vcs`. Tag a release as `v1.2.3` and
the wheel, the `dummy_mlwp.__version__` attribute, the output store's `source` attribute,
and the container tag all follow.

Pushing a version tag is also the only thing that publishes an image:

```sh
git tag v0.1.0 && git push origin v0.1.0
```

| Tag | Image tags |
| --- | --- |
| `v0.1.0` | `0.1.0`, `0.1`, `latest` |
| `v0.2.0rc1` (any PEP 440 pre-, post- or dev release) | `0.2.0rc1` only; `latest` does not move |

A pre-release tag is the way to get a test image onto a GPU host before a release.
Pull requests and pushes to `main` build and smoke-test the image, but do not push it,
and only when something that goes into the image changed. The publish job checks that
the image's version matches the tag before pushing anything.

## Development

```sh
uv pip install -e ".[dev]"
pytest
ruff check . && ruff format --check .
```

The test suite is CPU-only and needs no container.
