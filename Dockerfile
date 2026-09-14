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

COPY --from=ghcr.io/astral-sh/uv:0.9.16 /uv /usr/local/bin/uv

WORKDIR /src
COPY . .

# Print what the tag resolves to before building. A clone without tags yields a
# bare dev version, and seeing that in the build log is how you catch a workflow
# that forgot fetch-depth: 0.
RUN git describe --tags --dirty --always \
    && uv build --wheel --out-dir /dist \
    && ls -la /dist

# ---------------------------------------------------------------------------------
# Runtime stage: CUDA runtime + torch, then the wheel.
# ---------------------------------------------------------------------------------
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04 AS runtime

ARG PYTHON_VERSION=3.11
ARG TORCH_VERSION=2.5.1
ARG TORCH_INDEX=https://download.pytorch.org/whl/cu124

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:$PATH

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        software-properties-common ca-certificates \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        python${PYTHON_VERSION} python${PYTHON_VERSION}-venv \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.9.16 /uv /usr/local/bin/uv

RUN uv venv --python python${PYTHON_VERSION} ${VIRTUAL_ENV}

# torch first, as its own layer: it is by far the largest install and it changes
# far less often than the application source.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --index-url ${TORCH_INDEX} torch==${TORCH_VERSION}

# Then the scientific stack, still independent of the source.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install \
        "numpy>=1.26" "pandas>=2.1" "xarray>=2025.1.0" "cf-xarray>=0.9" \
        "zarr>=3.0" "fsspec>=2024.6" "loguru>=0.7" "s3fs>=2024.6" "gcsfs>=2024.6"

# Finally the application itself — a small layer that rebuilds on every commit.
COPY --from=build /dist/*.whl /tmp/
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --no-deps /tmp/*.whl \
    && rm -f /tmp/*.whl

RUN useradd --create-home --uid 1000 model
USER model
WORKDIR /home/model

# Configuration is entirely environmental, so there are no CMD arguments to pass.
ENTRYPOINT ["python", "-m", "dummy_mlwp"]

LABEL org.opencontainers.image.title="dummy-mlwp-model" \
      org.opencontainers.image.description="Dummy deep-learning weather model: zarr in, zarr out" \
      org.opencontainers.image.source="https://github.com/OWNER/dummy-mlwp-model" \
      org.opencontainers.image.licenses="MIT"
