"""Failing to reach a store: StorageError, exit code 5, and a message that says what to check.

The failures are produced three ways, none of which touches a real object store:

* ``flaky://`` — an in-memory fsspec filesystem that raises a chosen exception on a
  chosen path, so the real xarray -> zarr -> fsspec stack carries it to our code;
* local permissions — a read-only output directory, an unreadable input store;
* s3fs against ``127.0.0.1`` on a closed port, with every AWS credential source
  switched off, for the botocore errors exactly as s3fs surfaces them.
"""

from __future__ import annotations

import errno
import json
import os
import socket
import sys
import types

import pytest
import xarray as xr
from fsspec import register_implementation
from fsspec.implementations.memory import MemoryFileSystem

from dummy_mlwp.__main__ import main
from dummy_mlwp.config import Config
from dummy_mlwp.errors import InputError, StorageError
from dummy_mlwp.grid import detect_coords
from dummy_mlwp.inputs import open_input, stack_channels
from dummy_mlwp.outputs import write_output
from dummy_mlwp.storage import storage_error, storage_exceptions

try:
    from botocore import exceptions as botocore_exceptions
except ImportError:  # optional: it arrives with s3fs
    botocore_exceptions = None

needs_botocore = pytest.mark.skipif(botocore_exceptions is None, reason="needs botocore")

MINIMAL = {
    "INPUT_ZARR": "flaky://in.zarr",
    "OUTPUT_ZARR": "flaky://out.zarr",
    "INPUT_VARIABLES": "t2m,u10,v10",
    "OUTPUT_VARIABLES": "t2m:K",
}

#: One attempt only, so an unreachable endpoint fails in well under a second.
ONE_ATTEMPT = json.dumps({"config_kwargs": {"retries": {"max_attempts": 1}}})

needs_permissions = pytest.mark.skipif(
    not hasattr(os, "geteuid") or os.geteuid() == 0,
    reason="file permissions do not stop root",
)


class FlakyFileSystem(MemoryFileSystem):
    """An in-memory filesystem that raises ``error`` for every path containing ``match``."""

    protocol = "flaky"
    store: dict = {}
    pseudo_dirs = [""]
    error: BaseException | None = None
    match = ""

    def cat_file(self, path, start=None, end=None, **kwargs):
        if self.error is not None and self.match in path:
            raise self.error
        return super().cat_file(path, start=start, end=end, **kwargs)

    def pipe_file(self, path, value, mode="overwrite", **kwargs):
        if self.error is not None and self.match in path:
            raise self.error
        return super().pipe_file(path, value, mode=mode, **kwargs)


register_implementation("flaky", FlakyFileSystem, clobber=True)


@pytest.fixture
def flaky(monkeypatch):
    """Return a function that arms the flaky filesystem, starting from an empty store."""
    FlakyFileSystem.store.clear()
    FlakyFileSystem.pseudo_dirs[:] = [""]

    def _arm(error: BaseException, match: str = "") -> None:
        monkeypatch.setattr(FlakyFileSystem, "error", error)
        monkeypatch.setattr(FlakyFileSystem, "match", match)

    yield _arm
    FlakyFileSystem.store.clear()


@pytest.fixture
def flaky_input(flaky, make_input):
    """Copy a valid synthetic input store onto the flaky filesystem."""
    xr.open_zarr(make_input(name="flaky-source.zarr")).to_zarr(
        "flaky://in.zarr", mode="w", consolidated=True, zarr_format=3
    )
    return "flaky://in.zarr"


@pytest.fixture
def no_aws(monkeypatch, tmp_path):
    """Switch off every AWS credential source, so nothing here can find real ones."""
    for key in list(os.environ):
        if key.startswith(("AWS_", "SRC_", "DST_")):
            monkeypatch.delenv(key)
    monkeypatch.setenv("AWS_CONFIG_FILE", str(tmp_path / "no-aws-config"))
    monkeypatch.setenv("AWS_SHARED_CREDENTIALS_FILE", str(tmp_path / "no-aws-credentials"))
    monkeypatch.setenv("AWS_EC2_METADATA_DISABLED", "true")


def closed_port_endpoint() -> str:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    return f"http://127.0.0.1:{port}"


def invoke(monkeypatch, env: dict[str, str]) -> int:
    monkeypatch.delenv("LOG_LEVEL", raising=False)
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    return main()


def storage_failures():
    """Errors that mean the store could not be reached, as the backends raise them."""
    failures: list[BaseException] = [
        PermissionError("Access Denied"),
        OSError(errno.EIO, "Internal Error"),
        TimeoutError("timed out"),
        ConnectionRefusedError(errno.ECONNREFUSED, "Connection refused"),
    ]
    if botocore_exceptions is None:
        return failures
    return failures + [
        botocore_exceptions.NoCredentialsError(),
        botocore_exceptions.ProfileNotFound(profile="nope"),
        botocore_exceptions.EndpointConnectionError(endpoint_url="http://s3.example"),
        botocore_exceptions.ClientError(
            {"Error": {"Code": "AccessDenied", "Message": "Access Denied"}}, "GetObject"
        ),
        # A ValueError as well as a botocore error: must not be mistaken for a bad store.
        botocore_exceptions.UnknownEndpointError(service_name="s3", region_name="nowhere"),
    ]


def ids(errors):
    return [type(e).__name__ for e in errors]


# --- which exceptions count ----------------------------------------------------------


def test_oserror_always_counts_and_exception_never_does():
    found = storage_exceptions()
    assert OSError in found
    assert Exception not in found
    assert BaseException not in found


@needs_botocore
def test_botocore_errors_count_once_botocore_is_loaded():
    found = storage_exceptions()
    assert botocore_exceptions.BotoCoreError in found
    assert botocore_exceptions.ClientError in found


def test_backends_are_looked_up_not_imported(monkeypatch):
    fake = types.ModuleType("gcsfs.retry")

    class HttpError(Exception):
        pass

    fake.HttpError = HttpError
    monkeypatch.delitem(sys.modules, "gcsfs.retry", raising=False)
    assert HttpError not in storage_exceptions()
    assert "gcsfs.retry" not in sys.modules

    monkeypatch.setitem(sys.modules, "gcsfs.retry", fake)
    assert HttpError in storage_exceptions()


# --- the message ---------------------------------------------------------------------


def test_message_names_side_uri_error_and_its_cause():
    """s3fs raises PermissionError from a ClientError, which names the refused operation."""
    cause = OSError("An error occurred (AccessDenied) when calling the GetObject operation")
    exc = PermissionError("Access Denied")
    exc.__cause__ = cause

    message = str(storage_error("SRC", "s3://bucket/in.zarr", {"anon": True}, "open", exc))
    assert message.startswith("Could not open the input store 's3://bucket/in.zarr': ")
    assert "PermissionError: Access Denied" in message
    assert "when calling the GetObject operation" in message


def test_an_empty_error_message_still_names_the_type():
    message = str(storage_error("DST", "/out.zarr", {}, "write", OSError()))
    assert "Could not write the output store '/out.zarr': OSError" in message


def test_local_hints_depend_on_the_side():
    read = str(storage_error("SRC", "/data/in.zarr", {}, "open", PermissionError()))
    write = str(storage_error("DST", "file:///data/out.zarr", {}, "write", PermissionError()))
    assert "can read every file" in read
    assert "read-only" in write
    assert "AWS" not in read + write


def test_anonymous_s3_hint_names_the_side_variables():
    message = str(storage_error("SRC", "s3://b/in.zarr", {"anon": True}, "open", OSError()))
    assert "SRC is using anonymous access" in message
    assert "SRC_AWS_PROFILE" in message
    assert "SRC_S3_ENDPOINT_URL" in message


def test_signed_s3_hint_names_the_profile_and_endpoint():
    options = {
        "anon": False,
        "profile": "dmi-minio",
        "client_kwargs": {"endpoint_url": "https://s3.dmi.dk"},
    }
    message = str(storage_error("DST", "s3://b/out.zarr", options, "write", OSError()))
    assert "DST is signing requests with profile 'dmi-minio'" in message
    assert "allowed to write this bucket" in message
    assert "https://s3.dmi.dk" in message


def test_other_schemes_get_a_generic_hint():
    message = str(storage_error("SRC", "gs://b/in.zarr", {}, "open", OSError()))
    assert "gs://" in message
    assert "SRC_STORAGE_OPTIONS" in message


# --- input side ----------------------------------------------------------------------


@pytest.mark.parametrize("error", storage_failures(), ids=ids(storage_failures()))
def test_failing_to_open_the_input_is_a_storage_error(flaky, flaky_input, error):
    flaky(error)
    with pytest.raises(StorageError, match="Could not open the input store") as info:
        open_input(flaky_input)
    assert type(info.value.__cause__) is type(error)
    assert info.value.exit_code == 5


def test_a_missing_input_store_is_still_an_input_error(tmp_path):
    with pytest.raises(InputError) as info:
        open_input(str(tmp_path / "nope.zarr"))
    assert not isinstance(info.value, StorageError)


def test_a_missing_object_on_the_input_is_an_input_error(flaky, flaky_input):
    """s3fs reports a missing bucket or key as FileNotFoundError: a wrong INPUT_ZARR."""
    flaky(FileNotFoundError("NoSuchBucket"))
    with pytest.raises(InputError):
        open_input(flaky_input)


def test_a_programming_error_is_not_mistaken_for_storage(flaky, flaky_input):
    flaky(RuntimeError("a bug, not a storage problem"))
    with pytest.raises(RuntimeError, match="a bug"):
        open_input(flaky_input)


def test_failing_to_read_the_data_is_a_storage_error(flaky, flaky_input):
    """Opening reads only metadata and coordinates; the fields are fetched later."""
    ds = open_input(flaky_input)
    config = Config.from_env(MINIMAL)
    flaky(PermissionError("Access Denied"), match="t2m/c/")
    with pytest.raises(StorageError, match="Could not read the input store 'flaky://in.zarr'"):
        stack_channels(ds, config.input_variables, config, detect_coords(ds))


# --- output side ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "error",
    [*storage_failures(), FileNotFoundError("NoSuchBucket")],
    ids=[*ids(storage_failures()), "FileNotFoundError"],
)
def test_failing_to_write_the_output_is_a_storage_error(flaky, make_dataset, error):
    """Unlike the input, a missing destination is a storage failure: nothing should be there."""
    ds = make_dataset()
    flaky(error)
    with pytest.raises(StorageError, match="Could not write the output store") as info:
        write_output(ds, Config.from_env(MINIMAL), 3, detect_coords(ds))
    assert type(info.value.__cause__) is type(error)


def test_a_programming_error_while_writing_is_not_mistaken_for_storage(flaky, make_dataset):
    ds = make_dataset()
    flaky(RuntimeError("a bug, not a storage problem"))
    with pytest.raises(RuntimeError, match="a bug"):
        write_output(ds, Config.from_env(MINIMAL), 3, detect_coords(ds))


# --- end to end ----------------------------------------------------------------------


@needs_permissions
def test_read_only_output_directory_exits_5(monkeypatch, base_env, tmp_path, capsys):
    readonly = tmp_path / "readonly"
    readonly.mkdir()
    readonly.chmod(0o555)
    try:
        code = invoke(monkeypatch, base_env | {"OUTPUT_ZARR": str(readonly / "out.zarr")})
    finally:
        readonly.chmod(0o755)

    assert code == 5
    stderr = capsys.readouterr().err
    assert f"Could not write the output store '{readonly / 'out.zarr'}'" in stderr
    assert "PermissionError" in stderr
    assert "Traceback" not in stderr


@needs_permissions
def test_unreadable_input_store_exits_5(monkeypatch, base_env, capsys):
    store = base_env["INPUT_ZARR"]
    os.chmod(store, 0)
    try:
        code = invoke(monkeypatch, base_env)
    finally:
        os.chmod(store, 0o755)

    assert code == 5
    stderr = capsys.readouterr().err
    assert f"Could not open the input store '{store}'" in stderr
    assert "PermissionError" in stderr


def test_input_denied_part_way_through_exits_5(monkeypatch, base_env, flaky, flaky_input):
    flaky(PermissionError("Access Denied"), match="t2m/c/")
    assert invoke(monkeypatch, base_env | {"INPUT_ZARR": flaky_input}) == 5


def test_s3_input_without_credentials_exits_5(monkeypatch, base_env, no_aws, capsys):
    pytest.importorskip("s3fs")
    env = base_env | {
        "INPUT_ZARR": "s3://bucket/in.zarr",
        "SRC_S3_ANON": "false",
        "SRC_S3_ENDPOINT_URL": closed_port_endpoint(),
        "SRC_STORAGE_OPTIONS": ONE_ATTEMPT,
    }
    assert invoke(monkeypatch, env) == 5

    stderr = capsys.readouterr().err
    assert "Could not open the input store 's3://bucket/in.zarr'" in stderr
    assert "NoCredentialsError" in stderr
    assert "SRC_AWS_PROFILE" in stderr
    assert "Traceback" not in stderr


def test_s3_output_endpoint_unreachable_exits_5(monkeypatch, base_env, no_aws, capsys):
    pytest.importorskip("s3fs")
    endpoint = closed_port_endpoint()
    env = base_env | {
        "OUTPUT_ZARR": "s3://bucket/out.zarr",
        "DST_S3_ENDPOINT_URL": endpoint,
        "DST_STORAGE_OPTIONS": ONE_ATTEMPT,
    }
    assert invoke(monkeypatch, env) == 5

    stderr = capsys.readouterr().err
    assert "Could not write the output store 's3://bucket/out.zarr'" in stderr
    assert "EndpointConnectionError" in stderr
    assert "DST is using anonymous access" in stderr
    assert endpoint in stderr
