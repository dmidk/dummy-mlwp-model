"""End-to-end runs against real S3 traffic, served by local moto servers.

``test_storage.py`` checks the options we build; these check that s3fs actually reads
and writes with them. Each server is a separate ``moto_server`` process, so two of them
are genuinely two hosts with separate state — which is what the SRC_/DST_ split is for.
Nothing here touches the network beyond localhost.
"""

from __future__ import annotations

import socket
import subprocess
import sys
import time
import urllib.request
import uuid

import pytest
import xarray as xr

from dummy_mlwp.__main__ import main
from make_test_input import build

boto3 = pytest.importorskip("boto3")
pytest.importorskip("moto.server")
pytest.importorskip("s3fs")

KEY = "testing-key"
SECRET = "testing-secret"
REGION = "us-east-1"

#: Every ambient AWS variable that could leak a developer's or CI runner's real
#: configuration into these tests, or sign requests the test means to send unsigned.
_AMBIENT_AWS_VARS = (
    "AWS_PROFILE",
    "AWS_DEFAULT_PROFILE",
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "AWS_SESSION_TOKEN",
    "AWS_ENDPOINT_URL",
    "AWS_ENDPOINT_URL_S3",
    "AWS_WEB_IDENTITY_TOKEN_FILE",
    "AWS_ROLE_ARN",
    "AWS_CONTAINER_CREDENTIALS_RELATIVE_URI",
    "AWS_CONTAINER_CREDENTIALS_FULL_URI",
    "S3_ENDPOINT_URL",
)


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _start_moto() -> tuple[subprocess.Popen, str]:
    port = _free_port()
    proc = subprocess.Popen(
        [sys.executable, "-m", "moto.server", "-H", "127.0.0.1", "-p", str(port)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    url = f"http://127.0.0.1:{port}"
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        try:
            urllib.request.urlopen(f"{url}/moto-api/", timeout=1)
            return proc, url
        except OSError:
            if proc.poll() is not None:
                break
            time.sleep(0.1)
    proc.kill()
    raise RuntimeError("moto server did not start")


@pytest.fixture(scope="module")
def s3_hosts():
    """Two independent S3 endpoints, standing in for two different object stores."""
    procs, urls = [], []
    try:
        for _ in range(2):
            proc, url = _start_moto()
            procs.append(proc)
            urls.append(url)
        yield urls
    finally:
        for proc in procs:
            proc.terminate()
            proc.wait(timeout=10)


@pytest.fixture
def aws_home(tmp_path, monkeypatch):
    """Isolate botocore from the real ~/.aws and AWS_* environment.

    Returns a function that writes profiles into a throwaway config and credentials
    file, the way a mounted ``~/.aws`` would carry them into the container.
    """
    for name in _AMBIENT_AWS_VARS:
        monkeypatch.delenv(name, raising=False)
    config = tmp_path / "aws_config"
    credentials = tmp_path / "aws_credentials"
    config.write_text("")
    credentials.write_text("")
    monkeypatch.setenv("AWS_CONFIG_FILE", str(config))
    monkeypatch.setenv("AWS_SHARED_CREDENTIALS_FILE", str(credentials))
    monkeypatch.setenv("AWS_EC2_METADATA_DISABLED", "true")

    def _add_profile(name: str, endpoint: str) -> None:
        with config.open("a") as f:
            f.write(f"[profile {name}]\nregion = {REGION}\nendpoint_url = {endpoint}\n\n")
        with credentials.open("a") as f:
            f.write(f"[{name}]\naws_access_key_id = {KEY}\naws_secret_access_key = {SECRET}\n\n")

    return _add_profile


def _client(endpoint: str):
    return boto3.client(
        "s3",
        endpoint_url=endpoint,
        region_name=REGION,
        aws_access_key_id=KEY,
        aws_secret_access_key=SECRET,
    )


def _bucket(endpoint: str) -> str:
    """Create a uniquely named bucket, so no test sees another's objects or caches."""
    name = f"b-{uuid.uuid4().hex[:12]}"
    _client(endpoint).create_bucket(Bucket=name)
    return name


def _keys(endpoint: str, bucket: str) -> set[str]:
    pages = _client(endpoint).get_paginator("list_objects_v2").paginate(Bucket=bucket)
    return {obj["Key"] for page in pages for obj in page.get("Contents", [])}


def _signed(endpoint: str) -> dict:
    return {"key": KEY, "secret": SECRET, "client_kwargs": {"endpoint_url": endpoint}}


def _put_input(endpoint: str, zarr_format: int = 3, public: bool = False) -> str:
    """Upload a synthetic input store and return its s3:// URI.

    ``public`` uploads every object as ``public-read``, as a public dataset bucket
    would serve them; moto, like S3, refuses unsigned reads of private objects.
    """
    uri = f"s3://{_bucket(endpoint)}/in.zarr"
    options = _signed(endpoint)
    if public:
        options["s3_additional_kwargs"] = {"ACL": "public-read"}
    ds = build("projected", 4, 12, 16, None, "6h", 0)
    ds.to_zarr(uri, mode="w", consolidated=True, zarr_format=zarr_format, storage_options=options)
    return uri


def _open_output(endpoint: str, uri: str) -> xr.Dataset:
    return xr.open_zarr(uri, decode_timedelta=True, storage_options=_signed(endpoint))


def invoke(monkeypatch, env: dict[str, str]) -> int:
    monkeypatch.delenv("LOG_LEVEL", raising=False)
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    return main()


@pytest.fixture
def run_env():
    """Return the model settings shared by every run; storage is added per test."""
    return {
        "INPUT_VARIABLES": "t2m,u10,v10",
        "OUTPUT_VARIABLES": "t2m:K,tp:mm",
        "N_FORECAST_STEPS": "3",
        "DEVICE": "cpu",
        "MODEL_HIDDEN_CHANNELS": "8",
        "MODEL_LAYERS": "2",
    }


def test_profile_endpoint_from_aws_config(monkeypatch, s3_hosts, aws_home, run_env):
    """The documented primary mechanism: one profile carries host and credentials.

    No endpoint variable is set at all, so if s3fs ignored the profile's
    ``endpoint_url`` the run would go to real AWS and fail.
    """
    host = s3_hosts[0]
    aws_home("rig", host)
    source = _put_input(host)
    target = f"s3://{_bucket(host)}/out.zarr"

    env = run_env | {"AWS_PROFILE": "rig", "INPUT_ZARR": source, "OUTPUT_ZARR": target}
    assert invoke(monkeypatch, env) == 0

    out = _open_output(host, target)
    assert out.sizes["time"] == 3
    assert set(out.data_vars) == {"t2m", "tp", "crs"}


def test_two_hosts_with_separate_profiles(monkeypatch, s3_hosts, aws_home, run_env):
    """SRC_ and DST_ profiles route each side to its own host."""
    src_host, dst_host = s3_hosts
    aws_home("reader", src_host)
    aws_home("writer", dst_host)
    source = _put_input(src_host)
    target_bucket = _bucket(dst_host)
    target = f"s3://{target_bucket}/out.zarr"

    env = run_env | {
        "SRC_AWS_PROFILE": "reader",
        "DST_AWS_PROFILE": "writer",
        "INPUT_ZARR": source,
        "OUTPUT_ZARR": target,
    }
    assert invoke(monkeypatch, env) == 0

    assert _open_output(dst_host, target).sizes["time"] == 3
    # The output bucket exists only on the destination host, so a write that went to
    # the source host instead would have failed rather than landing there silently.
    source_buckets = {b["Name"] for b in _client(src_host).list_buckets()["Buckets"]}
    assert target_bucket not in source_buckets


def test_endpoint_override_and_explicit_keys(monkeypatch, s3_hosts, aws_home, run_env):
    """The no-config-file fallback: endpoint and keys straight from the environment."""
    src_host, dst_host = s3_hosts
    source = _put_input(src_host)
    target = f"s3://{_bucket(dst_host)}/out.zarr"

    env = run_env | {
        "SRC_S3_ENDPOINT_URL": src_host,
        "SRC_AWS_ACCESS_KEY_ID": KEY,
        "SRC_AWS_SECRET_ACCESS_KEY": SECRET,
        "DST_S3_ENDPOINT_URL": dst_host,
        "DST_AWS_ACCESS_KEY_ID": KEY,
        "DST_AWS_SECRET_ACCESS_KEY": SECRET,
        "INPUT_ZARR": source,
        "OUTPUT_ZARR": target,
    }
    assert invoke(monkeypatch, env) == 0
    assert _open_output(dst_host, target).sizes["time"] == 3


def test_anonymous_source_with_signed_destination(monkeypatch, s3_hosts, aws_home, run_env):
    """With no credentials anywhere, the source side must go out unsigned.

    Signing here would raise ``NoCredentialsError``, so a successful read is the proof
    that anonymous access was actually used rather than merely configured.
    """
    src_host, dst_host = s3_hosts
    aws_home("writer", dst_host)
    source = _put_input(src_host, public=True)
    target = f"s3://{_bucket(dst_host)}/out.zarr"

    env = run_env | {
        "SRC_S3_ENDPOINT_URL": src_host,
        "DST_AWS_PROFILE": "writer",
        "INPUT_ZARR": source,
        "OUTPUT_ZARR": target,
    }
    assert invoke(monkeypatch, env) == 0
    assert _open_output(dst_host, target).sizes["time"] == 3


def test_output_matches_input_zarr_format_over_s3(monkeypatch, s3_hosts, aws_home, run_env):
    """Format detection has to work through s3fs, not just on a local path."""
    host = s3_hosts[0]
    aws_home("rig", host)
    source = _put_input(host, zarr_format=2)
    target_bucket = _bucket(host)
    target = f"s3://{target_bucket}/out.zarr"

    env = run_env | {"AWS_PROFILE": "rig", "INPUT_ZARR": source, "OUTPUT_ZARR": target}
    assert invoke(monkeypatch, env) == 0

    keys = _keys(host, target_bucket)
    assert "out.zarr/.zgroup" in keys
    assert "out.zarr/zarr.json" not in keys


def test_missing_input_store_on_s3_exits_3(monkeypatch, s3_hosts, aws_home, run_env):
    host = s3_hosts[0]
    aws_home("rig", host)
    bucket = _bucket(host)

    env = run_env | {
        "AWS_PROFILE": "rig",
        "INPUT_ZARR": f"s3://{bucket}/does-not-exist.zarr",
        "OUTPUT_ZARR": f"s3://{bucket}/out.zarr",
    }
    assert invoke(monkeypatch, env) == 3
    assert _keys(host, bucket) == set()
