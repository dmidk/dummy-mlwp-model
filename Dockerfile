# syntax=docker/dockerfile:1

# ---------------------------------------------------------------------------------
# Build stage: turn the source tree into a wheel. This is where .git is needed, so
# hatch-vcs can derive the version from the git tag — and it is also where .git
# stays, since only the wheel is copied forward into the runtime image.
# ---------------------------------------------------------------------------------
FROM python:3.11-slim AS build

RUN apt-get update \
    && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.12.22 /uv /usr/local/bin/uv

WORKDIR /src
COPY . .

# Print what the tag resolves to before building. A clone without tags yields a
# bare dev version, and seeing that in the build log is how you catch a workflow
# that forgot fetch-depth: 0.
RUN git describe --tags --dirty --always \
    && uv build --wheel --out-dir /dist \
    && ls -la /dist

# torch and its CUDA libraries, cut out of uv.lock with their hashes, for the runtime
# stage's torch layer. They all install from the PyTorch index (its URL read from
# pyproject.toml, so it is defined once), which serves byte-identical copies of the
# nvidia-* and triton wheels, so the lock's hashes verify them from there too.
RUN uv export --frozen --no-emit-project --extra cu124 --output-file /tmp/all.txt \
    && index=$(python -c "import tomllib; print(next(i['url'] for i in tomllib.load(open('pyproject.toml', 'rb'))['tool']['uv']['index'] if i['name'] == 'pytorch-cu124'))") \
    && { echo "--index-url $index"; \
         awk '/^[^ #-]/ { keep = /^(torch|triton|nvidia-[a-z0-9-]+)==/ } keep' /tmp/all.txt; } \
       > /torch.txt \
    && grep -E '^(--index-url|[a-z])' /torch.txt

# ---------------------------------------------------------------------------------
# Runtime stage: CUDA base + the locked dependencies, then the wheel.
#
# The -base image, not -runtime: the torch wheel brings its own CUDA libraries as
# nvidia-*-cu12 pip packages, so -runtime's system copies of cuBLAS, cuFFT, NCCL and
# the rest (a 1.4 GB layer) would only duplicate them. -base still carries what GPU
# access needs: the NVIDIA_* variables the container toolkit reads to mount the host
# driver, and the NVIDIA_REQUIRE_CUDA driver check. Anything added later that links
# against system CUDA libraries instead of shipping its own would need -runtime back.
# ---------------------------------------------------------------------------------
FROM nvidia/cuda:12.4.1-base-ubuntu22.04 AS runtime

# The container's Python. CI's "container" job reads it from this line, so one CI job
# always tests exactly this Python with the same locked dependencies.
ARG PYTHON_VERSION=3.11

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    VIRTUAL_ENV=/opt/venv \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_PYTHON_INSTALL_DIR=/opt/python \
    # The CUDA wheels are hundreds of MB each; uv's 30s default times out on them.
    UV_HTTP_TIMEOUT=300 \
    PATH=/opt/venv/bin:$PATH

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.12.22 /uv /usr/local/bin/uv

# uv fetches its own standalone CPython, so the image needs neither the distro's
# Python nor a third-party PPA to get a current one.
RUN uv python install ${PYTHON_VERSION} \
    && uv venv --python ${PYTHON_VERSION} ${VIRTUAL_ENV}

# torch first, as its own layer: about 3 GB, and it changes far less often than
# anything else. Its pins come from uv.lock (see the build stage), and Docker caches
# this layer by their content alone, so a lock update that leaves torch alone does not
# rebuild it — and a GPU node that already has it does not pull it again.
COPY --from=build /torch.txt /tmp/torch.txt
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --no-deps --require-hashes -r /tmp/torch.txt

# Then every other dependency, exactly as uv.lock pins it — the versions CI tests —
# with torch's CUDA 12.4 build (the cu124 extra) and the s3fs/gcsfs backends (remote).
# torch is already installed at the locked versions, so uv sync keeps it and adds the
# rest; if anything differed, it would bring it to the lock. Only pyproject.toml and
# uv.lock are mounted in, so this layer rebuilds when the lock changes, not on every
# commit. --frozen installs the lock as is; CI's --locked fails a stale lock.
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=pyproject.toml,target=/tmp/lock/pyproject.toml \
    --mount=type=bind,source=uv.lock,target=/tmp/lock/uv.lock \
    cd /tmp/lock \
    && uv sync --frozen --no-install-project --python ${PYTHON_VERSION} \
        --extra cu124 --extra remote

# Finally the application itself — a small layer that rebuilds on every commit. Its
# dependencies are all in place already; pip check confirms the wheel agrees.
COPY --from=build /dist/*.whl /tmp/
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --no-deps /tmp/*.whl \
    && rm -f /tmp/*.whl \
    && uv pip check

RUN useradd --create-home --uid 1000 model
USER model
WORKDIR /home/model

# The image exists to exercise a GPU, so it requires one: started without GPU access it
# exits 4 instead of quietly running on CPU. Pass DEVICE=cpu (or auto) to run without a
# GPU, as the CI smoke test does — its runners have none. Set here, after the installs,
# so that changing it never invalidates the dependency layer. The Python default stays
# auto.
ENV DEVICE=cuda

# Configuration is entirely environmental, so there are no CMD arguments to pass.
ENTRYPOINT ["python", "-m", "dummy_mlwp"]

LABEL org.opencontainers.image.title="dummy-mlwp-model" \
      org.opencontainers.image.description="Dummy deep-learning weather model: zarr in, zarr out" \
      org.opencontainers.image.source="https://github.com/dmidk/dummy-mlwp-model" \
      org.opencontainers.image.licenses="MIT"
