from __future__ import annotations

import copy
import json

import pytest

from dummy_mlwp.errors import ConfigError
from dummy_mlwp.storage import _redact, storage_options

S3 = "s3://bucket/in.zarr"
OTHER_S3 = "s3://other-bucket/out.zarr"


def test_local_paths_get_no_options():
    assert storage_options({}, "/data/in.zarr", "SRC") == {}


def test_profile_applies_to_s3():
    options = storage_options({"SRC_AWS_PROFILE": "reader"}, S3, "SRC")
    assert options["profile"] == "reader"


def test_source_and_destination_are_independent():
    env = {"SRC_AWS_PROFILE": "era5-reader", "DST_AWS_PROFILE": "dmi-minio"}
    assert storage_options(env, S3, "SRC")["profile"] == "era5-reader"
    assert storage_options(env, OTHER_S3, "DST")["profile"] == "dmi-minio"


def test_unprefixed_profile_applies_to_both_sides():
    env = {"AWS_PROFILE": "shared"}
    assert storage_options(env, S3, "SRC")["profile"] == "shared"
    assert storage_options(env, S3, "DST")["profile"] == "shared"


def test_side_specific_wins_over_unprefixed():
    env = {"AWS_PROFILE": "shared", "DST_AWS_PROFILE": "writer"}
    assert storage_options(env, S3, "SRC")["profile"] == "shared"
    assert storage_options(env, S3, "DST")["profile"] == "writer"


def test_no_endpoint_option_when_the_profile_carries_it():
    """The AWS config is the normal home for endpoint_url, so we pass nothing."""
    options = storage_options({"SRC_AWS_PROFILE": "dmi-minio"}, S3, "SRC")
    assert "client_kwargs" not in options


def test_endpoint_override_is_available():
    options = storage_options({"DST_S3_ENDPOINT_URL": "https://s3.dmi.dk"}, OTHER_S3, "DST")
    assert options["client_kwargs"] == {"endpoint_url": "https://s3.dmi.dk"}


def test_aws_endpoint_url_is_accepted_as_a_fallback():
    options = storage_options({"AWS_ENDPOINT_URL": "https://s3.dmi.dk"}, S3, "SRC")
    assert options["client_kwargs"] == {"endpoint_url": "https://s3.dmi.dk"}


# --- anonymous by default ------------------------------------------------------------


def test_anon_by_default_when_nothing_is_configured():
    assert storage_options({}, S3, "SRC")["anon"] is True


def test_a_profile_turns_signing_on():
    assert storage_options({"SRC_AWS_PROFILE": "reader"}, S3, "SRC")["anon"] is False


def test_ambient_aws_keys_turn_signing_on():
    env = {"AWS_ACCESS_KEY_ID": "AKIA...", "AWS_SECRET_ACCESS_KEY": "..."}
    assert storage_options(env, S3, "SRC")["anon"] is False


def test_side_specific_keys_turn_signing_on_and_are_passed():
    env = {"SRC_AWS_ACCESS_KEY_ID": "AKIA...", "SRC_AWS_SECRET_ACCESS_KEY": "shh"}
    options = storage_options(env, S3, "SRC")
    assert options["anon"] is False
    assert options["key"] == "AKIA..."
    assert options["secret"] == "shh"


def test_side_specific_keys_do_not_leak_to_the_other_side():
    env = {"SRC_AWS_ACCESS_KEY_ID": "AKIA...", "SRC_AWS_SECRET_ACCESS_KEY": "shh"}
    assert "key" not in storage_options(env, OTHER_S3, "DST")


@pytest.mark.parametrize(
    "var",
    [
        "AWS_WEB_IDENTITY_TOKEN_FILE",
        "AWS_ROLE_ARN",
        "AWS_CONTAINER_CREDENTIALS_RELATIVE_URI",
        "AWS_SESSION_TOKEN",
    ],
)
def test_ambient_role_credentials_turn_signing_on(var):
    """IAM roles on ECS/EKS/EC2 are credentials, even with no keys in the environment."""
    assert storage_options({var: "something"}, S3, "SRC")["anon"] is False


def test_credentials_in_the_json_escape_hatch_turn_signing_on():
    env = {"SRC_STORAGE_OPTIONS": '{"key": "AKIA...", "secret": "shh"}'}
    assert storage_options(env, S3, "SRC")["anon"] is False


def test_explicit_anon_wins_over_credentials():
    env = {"SRC_AWS_PROFILE": "reader", "SRC_S3_ANON": "true"}
    assert storage_options(env, S3, "SRC")["anon"] is True


def test_explicit_anon_false_wins_over_the_default():
    assert storage_options({"SRC_S3_ANON": "false"}, S3, "SRC")["anon"] is False


def test_json_can_override_anon():
    env = {"SRC_STORAGE_OPTIONS": '{"anon": false}'}
    assert storage_options(env, S3, "SRC")["anon"] is False


@pytest.mark.parametrize(
    ("value", "expected"), [("true", True), ("1", True), ("false", False), ("no", False)]
)
def test_anon_spellings(value, expected):
    assert storage_options({"SRC_S3_ANON": value}, S3, "SRC")["anon"] is expected


def test_bad_anon_is_rejected():
    with pytest.raises(ConfigError, match="must be a boolean"):
        storage_options({"SRC_S3_ANON": "maybe"}, S3, "SRC")


def test_anon_is_not_set_for_non_s3_schemes():
    assert "anon" not in storage_options({}, "gs://bucket/in.zarr", "SRC")
    assert storage_options({}, "/data/in.zarr", "SRC") == {}


# --- the JSON escape hatch -----------------------------------------------------------


def test_s3_options_are_ignored_for_other_schemes():
    env = {"SRC_AWS_PROFILE": "reader"}
    assert storage_options(env, "gs://bucket/in.zarr", "SRC") == {}
    assert storage_options(env, "/data/in.zarr", "SRC") == {}


def test_json_reaches_any_option():
    env = {"SRC_STORAGE_OPTIONS": '{"requester_pays": true, "config_kwargs": {"max_pool": 4}}'}
    options = storage_options(env, S3, "SRC")
    assert options["requester_pays"] is True
    assert options["config_kwargs"] == {"max_pool": 4}


def test_json_applies_to_non_s3_schemes_too():
    env = {"DST_STORAGE_OPTIONS": '{"project": "my-gcp-project"}'}
    assert storage_options(env, "gs://bucket/out.zarr", "DST") == {"project": "my-gcp-project"}


def test_json_overrides_the_dedicated_variables():
    env = {"SRC_AWS_PROFILE": "reader", "SRC_STORAGE_OPTIONS": '{"profile": "override"}'}
    assert storage_options(env, S3, "SRC")["profile"] == "override"


@pytest.mark.parametrize(
    ("value", "message"),
    [("not json", "must be valid JSON"), ('["a"]', "must be a JSON object")],
)
def test_malformed_json_is_rejected(value, message):
    with pytest.raises(ConfigError, match=message):
        storage_options({"SRC_STORAGE_OPTIONS": value}, S3, "SRC")


# --- redaction of the startup log line -----------------------------------------------


def test_nested_secrets_are_redacted():
    options = {
        "profile": "dmi-minio",
        "client_kwargs": {
            "endpoint_url": "https://s3.dmi.dk",
            "region_name": "eu-north-1",
            "aws_access_key_id": "AKIAnested",
            "aws_secret_access_key": "nested-shh",
            "aws_session_token": "nested-session",
        },
        "config_kwargs": {"s3": {"addressing_style": "path"}, "max_pool_connections": 4},
    }
    assert _redact(options) == {
        "profile": "dmi-minio",
        "client_kwargs": {
            "endpoint_url": "https://s3.dmi.dk",
            "region_name": "eu-north-1",
            "aws_access_key_id": "***",
            "aws_secret_access_key": "***",
            "aws_session_token": "***",
        },
        "config_kwargs": {"s3": {"addressing_style": "path"}, "max_pool_connections": 4},
    }


def test_secret_name_masks_the_whole_value_even_when_it_is_an_object():
    """A gcsfs service-account dict under ``token`` goes as one, not field by field."""
    token = {"type": "service_account", "client_email": "sa@p.iam", "private_key": "-----"}
    assert _redact({"project": "my-gcp-project", "token": token}) == {
        "project": "my-gcp-project",
        "token": "***",
    }


def test_service_account_fields_are_masked_under_a_non_secret_name():
    info = {
        "type": "service_account",
        "project_id": "my-gcp-project",
        "private_key_id": "abc123",
        "private_key": "-----BEGIN PRIVATE KEY-----",
        "client_email": "runner@my-gcp-project.iam.gserviceaccount.com",
        "client_secret": "oauth-shh",
        "refresh_token": "refresh-shh",
    }
    redacted = _redact({"session_kwargs": {"info": info}})["session_kwargs"]["info"]
    assert redacted["private_key_id"] == "***"
    assert redacted["private_key"] == "***"
    assert redacted["client_secret"] == "***"
    assert redacted["refresh_token"] == "***"
    # Identifies the account rather than authenticating as it, so it stays readable.
    assert redacted["client_email"] == info["client_email"]
    assert redacted["project_id"] == "my-gcp-project"


def test_lists_of_dicts_are_redacted():
    options = {
        "candidates": [
            {"profile": "first", "secret": "list-shh"},
            ("tuple-item", {"password": "tuple-shh"}),
            "plain",
            3,
        ]
    }
    assert _redact(options) == {
        "candidates": [
            {"profile": "first", "secret": "***"},
            ("tuple-item", {"password": "***"}),
            "plain",
            3,
        ]
    }


@pytest.mark.parametrize(
    "name",
    [
        "key",
        "secret",
        "token",
        "password",
        "passwd",
        "passphrase",
        "credential",
        "credentials",
        "auth",
        "Authorization",
        "SSECustomerKey",
        "sas_token",
        "account_key",
        "connection_string",
    ],
)
def test_secret_names_are_masked(name):
    assert _redact({"headers": {name: "shh"}}) == {"headers": {name: "***"}}


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("profile", "dmi-minio"),
        ("region_name", "eu-north-1"),
        ("endpoint_url", "https://s3.dmi.dk"),
        ("project", "my-gcp-project"),
        ("client_email", "runner@my-gcp-project.iam.gserviceaccount.com"),
        ("anon", True),
        ("requester_pays", True),
        ("version_aware", False),
        ("default_block_size", 5242880),
        ("signature_version", "s3v4"),
        ("tcp_keepalive", True),
        ("client_kwargs", {}),
        ("s3_additional_kwargs", {"ACL": "private"}),
    ],
)
def test_realistic_non_secret_options_stay_visible(name, value):
    assert _redact({name: value}) == {name: value}


def test_url_passwords_are_masked_but_user_and_host_kept():
    options = {
        "client_kwargs": {"endpoint_url": "https://minio-user:url-shh@s3.dmi.dk:9000/"},
        "config_kwargs": {"proxies": {"https": "http://proxy-user:proxy-shh@proxy.dmi.dk:3128"}},
    }
    redacted = _redact(options)
    assert redacted["client_kwargs"]["endpoint_url"] == "https://minio-user:***@s3.dmi.dk:9000/"
    assert (
        redacted["config_kwargs"]["proxies"]["https"] == "http://proxy-user:***@proxy.dmi.dk:3128"
    )


@pytest.mark.parametrize(
    "value",
    [
        "https://s3.dmi.dk",
        "https://only-a-user@s3.dmi.dk",
        "eu-north-1",
        "not a url: at all",
        "http://[::1",  # urlsplit raises on this; redaction must not
    ],
)
def test_strings_without_a_url_password_are_unchanged(value):
    assert _redact({"endpoint_url": value}) == {"endpoint_url": value}


def test_redaction_does_not_modify_the_options_passed_to_fsspec():
    options = {
        "secret": "top-shh",
        "client_kwargs": {"aws_secret_access_key": "nested-shh", "endpoint_url": "https://s3"},
        "items": [{"token": "list-shh"}],
    }
    snapshot = copy.deepcopy(options)

    redacted = _redact(options)

    assert options == snapshot
    assert redacted["client_kwargs"] is not options["client_kwargs"]
    assert redacted["items"][0] is not options["items"][0]


def test_storage_options_returns_the_real_secrets():
    """Only the log is masked; s3fs still gets the credentials it was given."""
    env = {"SRC_STORAGE_OPTIONS": '{"client_kwargs": {"aws_secret_access_key": "nested-shh"}}'}
    options = storage_options(env, S3, "SRC")
    assert options["client_kwargs"]["aws_secret_access_key"] == "nested-shh"


def test_nested_destination_secret_stays_out_of_the_startup_log(monkeypatch, base_env, capsys):
    """Drive the whole application, and read the log a pipeline would actually see."""
    from dummy_mlwp.__main__ import main

    storage = {
        "client_kwargs": {"region_name": "eu-north-1", "aws_secret_access_key": "nested-shh"},
        "config_kwargs": {"proxies": {"https": "http://proxy-user:proxy-shh@proxy.dmi.dk"}},
        "session_kwargs": [{"private_key": "pem-shh", "client_email": "sa@p.iam"}],
    }
    env = base_env | {
        # memory:// stands in for s3:// and accepts (and ignores) these options.
        "OUTPUT_ZARR": "memory://redacted-startup-log.zarr",
        "DST_STORAGE_OPTIONS": json.dumps(storage),
        "N_FORECAST_STEPS": "1",
    }
    monkeypatch.delenv("LOG_LEVEL", raising=False)
    for key, value in env.items():
        monkeypatch.setenv(key, value)

    assert main() == 0

    stderr = capsys.readouterr().err
    line = next(line for line in stderr.splitlines() if "DST storage options" in line)
    for secret in ("nested-shh", "proxy-shh", "pem-shh"):
        assert secret not in stderr
    assert "eu-north-1" in line
    assert "proxy-user:***@proxy.dmi.dk" in line
    assert "sa@p.iam" in line


# --- the options have to be ones s3fs actually understands --------------------------

s3fs = pytest.importorskip("s3fs")


@pytest.mark.parametrize(
    "env",
    [
        {},
        {"SRC_AWS_PROFILE": "some-profile"},
        {"SRC_S3_ENDPOINT_URL": "https://s3.dmi.dk"},
        {"SRC_AWS_ACCESS_KEY_ID": "AKIA...", "SRC_AWS_SECRET_ACCESS_KEY": "shh"},
        {"SRC_S3_ANON": "true"},
        {"SRC_STORAGE_OPTIONS": '{"requester_pays": true}'},
    ],
)
def test_options_construct_a_real_s3_filesystem(env):
    """Guards against inventing an option name s3fs would silently ignore or reject.

    No network: constructing the filesystem only resolves configuration.
    """
    fs = s3fs.S3FileSystem(**storage_options(env, S3, "SRC"), skip_instance_cache=True)
    assert "s3" in fs.protocol


def test_endpoint_override_reaches_the_filesystem():
    options = storage_options({"SRC_S3_ENDPOINT_URL": "https://s3.dmi.dk"}, S3, "SRC")
    fs = s3fs.S3FileSystem(**options, skip_instance_cache=True)
    assert fs.client_kwargs["endpoint_url"] == "https://s3.dmi.dk"


def test_anon_default_reaches_the_filesystem():
    fs = s3fs.S3FileSystem(**storage_options({}, S3, "SRC"), skip_instance_cache=True)
    assert fs.anon is True


def test_credentials_are_redacted_from_the_log():
    from loguru import logger

    messages: list[str] = []
    sink = logger.add(lambda m: messages.append(str(m)), level="INFO")
    try:
        env = {"SRC_STORAGE_OPTIONS": '{"key": "AKIAsecret", "secret": "shhh"}'}
        storage_options(env, S3, "SRC")
    finally:
        logger.remove(sink)

    combined = "".join(messages)
    assert "AKIAsecret" not in combined
    assert "shhh" not in combined
    assert "***" in combined
