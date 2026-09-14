from __future__ import annotations

import pytest

from dummy_mlwp.errors import ConfigError
from dummy_mlwp.storage import storage_options

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
