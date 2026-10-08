# AGENTS.md

Guidance for AI coding agents working in this repository. Humans may find it useful too.

## What this project is

A containerised stand-in for a deep-learning weather-forecasting model. It reads a zarr
store, runs a real convolutional forward pass on the GPU, and writes a zarr store.
Everything is configured through environment variables.

It is **not** a weather model. Its job is to let the infrastructure around a real MLWP
model — orchestration, storage, GPU scheduling, error handling — be built and tested
before that model exists. Two properties follow from that, and they outrank ordinary
code-quality instincts:

1. **Strictness is the feature.** Every expectation about the input is declared in
   configuration and checked, and each class of failure has its own exit code. Do not
   "helpfully" coerce, guess around, or silently tolerate an input that does not match
   what was declared. Pipelines are tested against these failures.
2. **The GPU work must be real.** `predict()` runs an actual forward pass on the
   selected device in *every* output mode. Never add a code path that skips it — that
   includes optimising away the compute for `zeros` or `constant` mode, which would
   look like a sensible improvement and would quietly destroy the point of the project.

## Layout

```
src/dummy_mlwp/
  __main__.py   entrypoint: run(), exit-code mapping, logging setup
  config.py     env -> Config dataclass; all parsing and validation
  varspec.py    the name[:units][@levelCoord] grammar (pure, no I/O)
  grid.py       cf-xarray coordinate discovery + regular-grid validation
  timeaxis.py   dt inference, forecast time construction
  storage.py    per-side (SRC_/DST_) fsspec options; storage failures -> StorageError
  inputs.py     open the store, assert it matches the config, pack channels
  model.py      DummyNet, device selection, forward pass, rollout
  outputs.py    assemble the output dataset, write zarr
  errors.py     ConfigError / InputError / DeviceError / StorageError, each with an exit code
scripts/
  make_test_input.py   synthetic input generator, reused by the test fixtures
tests/
```

## Conventions

- **Docstrings are numpy style**, on every function and method, private ones included.
  `ruff` enforces this (`D` rules, `convention = "numpy"`); tests are exempt from the
  *missing*-docstring rules only.
- **Logging is loguru, with f-strings**: `logger.info(f"...")`, not `%s` or `{}`
  placeholder arguments. Import as `from loguru import logger`.
- **Line length 100.** Run `ruff format .` rather than hand-wrapping.
- **Versioning is `hatch-vcs` from the git tag.** Never hardcode a version; never edit
  `src/dummy_mlwp/_version.py`, which is generated.
- **Store I/O maps its failures to `StorageError` (exit 5).** Wrap any new read or write
  of a store in `except storage_exceptions() as exc: raise storage_error(...) from exc`
  (both in `storage.py`). Never widen that to bare `Exception`: a programming error must
  stay exit 1 with its traceback. A missing *input* store stays an `InputError` (exit 3).
- Prefer the existing helpers over new ones: `channel_layout` is the single source of
  truth for channel ordering, and `scripts/make_test_input.py:build` is the single
  synthetic-data generator (the test fixtures import it).

## The two invariants worth stating explicitly

**Channel ordering.** `varspec.channel_layout` flattens variables into 2D fields, in
declaration order, expanding levels. `inputs.stack_channels` packs with it and
`outputs.build_output_dataset` unpacks with it. If you change one, change both — a
silent drift here produces plausible-looking output with variables swapped.

**Validation is collected, not raised eagerly.** Both layers gather every problem and
raise once, formatted with `errors.format_problems`, so a misconfigured pipeline reports
all its problems in one run:

- `inputs.validate_input` — new checks append to its `problems` list, not raise on the
  spot.
- `Config.from_env` — the parse helpers (`_get_int`, `varspec.parse_*`,
  `storage.storage_options`, ...) still raise `ConfigError`; `from_env` runs each through
  `_attempt`, which records the message and returns `None` as a placeholder. A new
  variable goes through `_attempt` too, and a cross-check appends to `problems` directly.
  A check that depends on a value that failed to parse is skipped (`if x is not None`),
  so one mistake is reported once: a broken `LEVEL_COORDS` makes `parse_var_specs` skip
  only its "is this `@` reference declared?" check, rather than flagging every reference.
  A lone problem is raised as its bare message, without the headline.

## Testing

```sh
uv pip install -e ".[dev]"
pytest                        # ~110 tests, CPU only, a few seconds
ruff check . && ruff format --check .
```

The suite never needs a GPU or a container; end-to-end tests drive `main()` with a
patched environment and check the resulting store and exit code. When adding a feature,
add both a unit test for the logic and an end-to-end test for the behaviour a pipeline
would actually see.

**The GPU path is the one thing tests cannot cover.** Changes to `model.py` need a
manual check on a GPU host — see the README's GPU section, and confirm the log reports
a device name and non-zero peak memory.

## Things that have already been decided

Do not revisit these without being asked:

- Configuration is environment variables only. No CLI, no config file.
- Level coordinates are declared once in `LEVEL_COORDS` and referenced by name.
  Coordinate names are camelCase (`isobaricInhPa`, `heightAboveGround`).
- The input and output stores are configured **independently**, via `SRC_*` and `DST_*`
  variables, so a run can read and write across two different S3 hosts or accounts.
  Unprefixed spellings fall back to both sides. Anything without a dedicated variable
  goes through the `*_STORAGE_OPTIONS` JSON escape hatch — resist adding a new env var
  per fsspec option.
- **Endpoints belong in `~/.aws/config`**, on the profile, not in the environment. One
  profile name then carries host, region and credentials together, which is what makes
  the two-host case tidy. `*_S3_ENDPOINT_URL` stays only as an override for deployments
  that cannot mount a config file; do not promote it to the primary mechanism.
- **S3 access is anonymous unless credentials are actually present** (a profile,
  explicit keys, or ambient IAM role variables). Public buckets are the common case for
  a test rig, and signing by default turns that into a confusing `NoCredentialsError`.
  `_has_credentials` in `storage.py` is the single place that decides this.
- `N_INPUT_TIMESTEPS` selects a window off either end of the input's time axis
  (positive: first n, negative: last |n|, unset: all). It is applied *after* validation
  and *before* anything reads the time axis, so the forecast anchors to the last
  **selected** timestep. Asking for more timesteps than exist is an error, never a
  silent truncation.
- Coordinates are auto-detected with cf-xarray, overridable by env var.
- The output store's zarr format matches the input's unless `ZARR_FORMAT` says otherwise.
- Output chunking is one timestep per chunk, full spatial extent, not configurable.
- The container's CUDA base is amd64-only; there is no arm64 image.
- **Images are pushed to `ghcr.io` only for version tags.** `vX.Y.Z` publishes `X.Y.Z`,
  `X.Y` and `latest`; a PEP 440 pre-, post- or dev release tag publishes only its own
  version and never moves `latest`. Pushes to `main` and PRs build and smoke-test the
  image without pushing it. The image is smoke-tested *before* it is pushed.

## Pull requests

CI runs the test suite on Python 3.11 and 3.12 with CPU-only torch, plus lint and an
end-to-end smoke test. The image workflow builds and smoke-tests the image on PRs and
pushes to `main`, but only when a path that goes into the image changed (the `paths`
list in `publish-image.yml` — extend it if you add one); it pushes to `ghcr.io` only for
version tags. Both workflows check out with `fetch-depth: 0`, because `hatch-vcs` needs
the tags — if you touch the workflows, keep that. Likewise `.dockerignore` must only
exclude paths git ignores, or the image's version gets a dirty-tree date stamp.
