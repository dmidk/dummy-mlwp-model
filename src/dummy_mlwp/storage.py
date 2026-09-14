"""Per-side storage options, so input and output can live on different object stores.

The input and output stores are configured independently: a run can read from one S3
account and write to another, or read from a public bucket and write to an on-premise
S3-compatible host. Each side reads its own ``SRC_`` / ``DST_`` variables, falling back
to the unprefixed spellings when both sides share one configuration.

Two deliberate choices:

* **Endpoints belong in the AWS config.** A profile in ``~/.aws/config`` can carry its
  own ``endpoint_url``, which is how one run reaches two different S3-compatible hosts
  with nothing but two profile names. ``<side>_S3_ENDPOINT_URL`` stays available as an
  override for deployments that cannot mount a config file.
* **Access is anonymous unless credentials are actually given.** Reading a public
  bucket is the common case for a test rig, and defaulting to signed requests turns
  that into a confusing ``NoCredentialsError``. Naming a profile, supplying keys, or
  running under an IAM role all count as credentials and switch signing back on.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any
from urllib.parse import urlsplit

from loguru import logger

from .errors import ConfigError

#: Schemes for which the S3-specific options below are meaningful.
S3_SCHEMES = ("s3", "s3a")

#: Environment variables that mean "this process already has AWS credentials", beyond a
#: profile or explicit keys: the ambient role credentials used on ECS, EKS and EC2.
_AMBIENT_CREDENTIAL_VARS = (
    "AWS_ACCESS_KEY_ID",
    "AWS_SESSION_TOKEN",
    "AWS_WEB_IDENTITY_TOKEN_FILE",
    "AWS_ROLE_ARN",
    "AWS_CONTAINER_CREDENTIALS_RELATIVE_URI",
    "AWS_CONTAINER_CREDENTIALS_FULL_URI",
)

#: fsspec option names that carry a credential.
_CREDENTIAL_OPTIONS = ("profile", "key", "secret", "token")


def storage_options(env: Mapping[str, str], uri: str, side: str) -> dict[str, Any]:
    """Build the fsspec storage options for one side of the run.

    Parameters
    ----------
    env : mapping of str to str
        Environment to read.
    uri : str
        The store URI, whose scheme decides which options apply.
    side : {'SRC', 'DST'}
        Which side is being configured; also the environment variable prefix.

    Returns
    -------
    dict
        Keyword arguments for the fsspec filesystem. Empty for a local path with no
        overrides, which leaves the existing behaviour untouched. For an S3 URI,
        always includes ``anon``, resolved as described in the module docstring.

    Raises
    ------
    ConfigError
        If ``<side>_STORAGE_OPTIONS`` is not a JSON object, or ``<side>_S3_ANON`` is
        not a boolean.
    """
    scheme = urlsplit(uri).scheme.lower()
    options: dict[str, Any] = {}

    if scheme in S3_SCHEMES:
        options.update(_s3_options(env, side))
    elif _s3_variables_set(env, side):
        logger.warning(
            f"{side}_AWS_PROFILE / {side}_S3_ENDPOINT_URL / {side}_S3_ANON are set but "
            f"{uri} is not an S3 URI; ignoring them"
        )

    # Merged last, so it can override anything above and reach options that have no
    # dedicated variable of their own.
    options.update(_extra(env, side))

    if scheme in S3_SCHEMES and "anon" not in options:
        options["anon"] = not _has_credentials(env, side, options)

    if options:
        logger.info(f"{side} storage options: {_redact(options)}")
        if options.get("anon"):
            logger.info(
                f"{side} is using anonymous access; set {side}_AWS_PROFILE (or supply "
                f"credentials) if the bucket is not public"
            )
    return options


def _s3_options(env: Mapping[str, str], side: str) -> dict[str, Any]:
    """Collect the S3-specific options for one side.

    Parameters
    ----------
    env : mapping of str to str
        Environment to read.
    side : str
        Variable prefix, ``'SRC'`` or ``'DST'``.

    Returns
    -------
    dict
        Any of ``profile``, ``client_kwargs``, ``key``, ``secret``, ``token`` and
        ``anon`` that the environment specifies. ``anon`` appears only when set
        explicitly; otherwise it is derived later, once the JSON options are known.

    Raises
    ------
    ConfigError
        If ``<side>_S3_ANON`` is not a boolean.
    """
    options: dict[str, Any] = {}

    profile = _get(env, side, "AWS_PROFILE")
    if profile:
        options["profile"] = profile

    # Usually unset: the endpoint is better carried by the profile in ~/.aws/config.
    endpoint = _get(env, side, "S3_ENDPOINT_URL", "AWS_ENDPOINT_URL")
    if endpoint:
        options["client_kwargs"] = {"endpoint_url": endpoint}

    for option, name in (
        ("key", "AWS_ACCESS_KEY_ID"),
        ("secret", "AWS_SECRET_ACCESS_KEY"),
        ("token", "AWS_SESSION_TOKEN"),
    ):
        # Only the prefixed spellings: unprefixed AWS_* keys are already ambient, and
        # boto picks them up itself.
        value = env.get(f"{side}_{name}", "").strip()
        if value:
            options[option] = value

    anon = _get(env, side, "S3_ANON")
    if anon is not None:
        options["anon"] = _parse_bool(anon, f"{side}_S3_ANON")

    return options


def _s3_variables_set(env: Mapping[str, str], side: str) -> bool:
    """Report whether any S3-only variable is set for this side.

    Parameters
    ----------
    env : mapping of str to str
        Environment to read.
    side : str
        Variable prefix, ``'SRC'`` or ``'DST'``.

    Returns
    -------
    bool
        True when at least one prefixed S3 variable is set, which is worth warning
        about if the URI turns out not to be an S3 one.
    """
    names = ("AWS_PROFILE", "S3_ENDPOINT_URL", "S3_ANON", "AWS_ACCESS_KEY_ID")
    return any(env.get(f"{side}_{name}", "").strip() for name in names)


def _has_credentials(env: Mapping[str, str], side: str, options: Mapping[str, Any]) -> bool:
    """Decide whether this side has credentials, and so should sign its requests.

    Parameters
    ----------
    env : mapping of str to str
        Environment to read.
    side : str
        Variable prefix, ``'SRC'`` or ``'DST'``.
    options : mapping
        The storage options resolved so far, including the JSON escape hatch.

    Returns
    -------
    bool
        True when a profile, explicit keys, or ambient role credentials are present.
        False means the request should go out unsigned.
    """
    if any(options.get(name) for name in _CREDENTIAL_OPTIONS):
        return True
    if _get(env, side, "AWS_PROFILE"):
        return True
    return any(env.get(name, "").strip() for name in _AMBIENT_CREDENTIAL_VARS)


def _get(env: Mapping[str, str], side: str, *names: str) -> str | None:
    """Read one side-specific variable, falling back to the unprefixed spellings.

    Parameters
    ----------
    env : mapping of str to str
        Environment to read.
    side : str
        Variable prefix, ``'SRC'`` or ``'DST'``.
    *names : str
        Variable names without the prefix, in order of preference.

    Returns
    -------
    str or None
        The first of ``<side>_<name>`` (all names, in order) that is set, then the
        first bare ``<name>`` that is set — so a plain ``AWS_PROFILE`` still applies to
        both sides — otherwise ``None``.
    """
    for key in [f"{side}_{name}" for name in names] + list(names):
        value = env.get(key, "").strip()
        if value:
            return value
    return None


def _extra(env: Mapping[str, str], side: str) -> dict[str, Any]:
    """Parse the JSON escape hatch for options with no dedicated variable.

    Parameters
    ----------
    env : mapping of str to str
        Environment to read.
    side : str
        Variable prefix, ``'SRC'`` or ``'DST'``.

    Returns
    -------
    dict
        The decoded object, or empty when unset.

    Raises
    ------
    ConfigError
        If the value is not valid JSON, or is JSON but not an object.
    """
    key = f"{side}_STORAGE_OPTIONS"
    raw = env.get(key, "").strip()
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ConfigError(f"{key} must be valid JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise ConfigError(f"{key} must be a JSON object, got {type(parsed).__name__}")
    return parsed


def _parse_bool(value: str, key: str) -> bool:
    """Parse a boolean environment variable.

    Parameters
    ----------
    value : str
        The raw value.
    key : str
        Variable name, used in the error message.

    Returns
    -------
    bool
        The parsed value.

    Raises
    ------
    ConfigError
        If the value is not a recognised boolean spelling.
    """
    lowered = value.strip().lower()
    if lowered in ("1", "true", "yes", "on"):
        return True
    if lowered in ("0", "false", "no", "off"):
        return False
    raise ConfigError(f"{key} must be a boolean such as 'true' or 'false', got {value!r}")


def _redact(options: Mapping[str, Any]) -> dict[str, Any]:
    """Mask credential-bearing values so options can be logged safely.

    Parameters
    ----------
    options : mapping
        The storage options about to be logged.

    Returns
    -------
    dict
        A copy with secret-looking values replaced by ``'***'``. Profile and endpoint
        names are kept, since seeing which account and host a run used is the whole
        point of logging this.
    """
    secret = ("key", "secret", "token", "password")
    return {k: ("***" if any(s in k.lower() for s in secret) else v) for k, v in options.items()}
