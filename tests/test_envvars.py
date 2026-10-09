"""The environment-variable registry and the misspelling warning."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest
from loguru import logger

from dummy_mlwp.config import Config
from dummy_mlwp.envvars import REGISTRY, warn_unknown_env_vars

PACKAGE = Path(__file__).resolve().parent.parent / "src" / "dummy_mlwp"


@pytest.fixture
def logged():
    """Collect (level, message) for everything logged at WARNING or above."""
    records: list[tuple[str, str]] = []
    handler = logger.add(
        lambda message: records.append((message.record["level"].name, message.record["message"])),
        level="WARNING",
    )
    yield records
    logger.remove(handler)


# --- misspellings --------------------------------------------------------------------


@pytest.mark.parametrize(
    ("typo", "meant"),
    [
        ("N_FORECAST_STEP", "N_FORECAST_STEPS"),
        ("N_FORCAST_STEPS", "N_FORECAST_STEPS"),
        ("N_INPUT_TIMESTEP", "N_INPUT_TIMESTEPS"),
        ("INPUT_VARIBLES", "INPUT_VARIABLES"),
        ("OUTPUTMODE", "OUTPUT_MODE"),
        ("MODEL_LAYER", "MODEL_LAYERS"),
        ("LOGLEVEL", "LOG_LEVEL"),
        ("DEVCIE", "DEVICE"),
        ("AWS_PROFLE", "AWS_PROFILE"),
        ("SRC_AWS_PROFIL", "SRC_AWS_PROFILE"),
        ("DST_S3_ENDPOINT", "DST_S3_ENDPOINT_URL"),
    ],
)
def test_a_misspelling_warns_with_a_suggestion(typo, meant):
    warnings = warn_unknown_env_vars({typo: "8"})
    assert list(warnings) == [typo]
    assert f"{typo} has no effect" in warnings[typo]
    assert f"Did you mean {meant}?" in warnings[typo]


def test_the_warning_is_logged_at_warning_level(logged):
    warn_unknown_env_vars({"N_FORECAST_STEP": "8"})
    assert len(logged) == 1
    level, message = logged[0]
    assert level == "WARNING"
    assert "Did you mean N_FORECAST_STEPS?" in message


def test_each_offending_name_warns_once(logged):
    warnings = warn_unknown_env_vars({"N_FORECAST_STEP": "8", "INPUT_VARIBLES": "t2m"})
    assert list(warnings) == ["INPUT_VARIBLES", "N_FORECAST_STEP"]
    assert len(logged) == 2


def test_wrong_case_is_reported():
    warnings = warn_unknown_env_vars({"n_forecast_steps": "8"})
    assert "Did you mean N_FORECAST_STEPS?" in warnings["n_forecast_steps"]
    assert "case-sensitive" in warnings["n_forecast_steps"]


def test_equally_close_names_are_all_suggested():
    """There is no unprefixed STORAGE_OPTIONS; either side's is an equally good guess."""
    warnings = warn_unknown_env_vars({"STORAGE_OPTIONS": "{}"})
    assert "Did you mean DST_STORAGE_OPTIONS or SRC_STORAGE_OPTIONS?" in warnings["STORAGE_OPTIONS"]


def test_values_are_never_logged(logged):
    """A misspelled variable may hold a secret, so only its name may appear."""
    warnings = warn_unknown_env_vars({"SRC_AWS_SECRET_ACESS_KEY": "hunter2"})
    assert "Did you mean SRC_AWS_SECRET_ACCESS_KEY?" in warnings["SRC_AWS_SECRET_ACESS_KEY"]
    assert not any("hunter2" in message for _, message in logged)


def test_reads_the_process_environment_by_default(monkeypatch):
    monkeypatch.setenv("N_FORECAST_STEP", "8")
    assert "N_FORECAST_STEP" in warn_unknown_env_vars()


# --- the store prefixes --------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "side", "store"), [("SRC_REGION", "SRC", "input"), ("DST_BUCKET", "DST", "output")]
)
def test_unknown_store_prefixed_names_warn(name, side, store):
    warnings = warn_unknown_env_vars({name: "x"})
    assert f"{side}_ prefix is reserved for the {store} store" in warnings[name]
    assert f"{side}_STORAGE_OPTIONS" in warnings[name]


# --- what must stay quiet ------------------------------------------------------------


def test_registered_variables_do_not_warn():
    assert warn_unknown_env_vars({name: "x" for name in REGISTRY}) == {}


def test_kubernetes_service_links_do_not_warn():
    """Services named input, src, dst or input-zarr inject these into every pod."""
    env = {
        "INPUT_SERVICE_HOST": "10.0.0.11",
        "INPUT_SERVICE_PORT": "8080",
        "INPUT_SERVICE_PORT_HTTP": "8080",
        "INPUT_PORT": "tcp://10.0.0.11:8080",
        "INPUT_PORT_8080_TCP": "tcp://10.0.0.11:8080",
        "INPUT_PORT_8080_TCP_ADDR": "10.0.0.11",
        "INPUT_PORT_8080_TCP_PORT": "8080",
        "INPUT_PORT_8080_TCP_PROTO": "tcp",
        "SRC_SERVICE_HOST": "10.0.0.12",
        "SRC_PORT": "tcp://10.0.0.12:9000",
        "SRC_PORT_9000_TCP_ADDR": "10.0.0.12",
        "DST_SERVICE_PORT": "9000",
        "DST_PORT_53_UDP": "udp://10.0.0.13:53",
        "DST_PORT_9000_SCTP_PROTO": "sctp",
        # Close enough to INPUT_ZARR to count as a misspelling, were it not a link.
        "INPUT_ZARR_SERVICE_HOST": "10.0.0.14",
        "INPUT_ZARR_PORT": "tcp://10.0.0.14:80",
        "KUBERNETES_SERVICE_HOST": "10.96.0.1",
        "KUBERNETES_PORT_443_TCP_PROTO": "tcp",
    }
    assert warn_unknown_env_vars(env) == {}


@pytest.mark.parametrize(
    "name",
    [
        "HOSTNAME",
        "PATH",
        "HOME",
        "LANG",
        "TERM",
        "TZ",
        "http_proxy",
        "LD_LIBRARY_PATH",
        "PYTHONUNBUFFERED",
        "NVIDIA_VISIBLE_DEVICES",
        "NVIDIA_DRIVER_CAPABILITIES",
        "CUDA_VERSION",
        "CUDA_VISIBLE_DEVICES",
        "PYTORCH_CUDA_ALLOC_CONF",
        "OMP_NUM_THREADS",
        "AWS_REGION",
        "AWS_DEFAULT_REGION",
        "AWS_CONFIG_FILE",
        "AWS_SHARED_CREDENTIALS_FILE",
        "AWS_ROLE_SESSION_NAME",
        "AWS_SECURITY_TOKEN",
        # Read by boto, and close to a registered name: allowed explicitly.
        "AWS_SECRET_ACCESS_KEY",
        "AWS_ENDPOINT_URL_S3",
        "AWS_ENDPOINT_URL_STS",
        "GOOGLE_APPLICATION_CREDENTIALS",
        "POD_NAME",
        "JOB_COMPLETION_INDEX",
        "MODEL_NAME",
        "OUTPUT_DIR",
    ],
)
def test_unrelated_variables_do_not_warn(name):
    assert warn_unknown_env_vars({name: "x"}) == {}


@pytest.mark.parametrize(
    ("near_miss", "meant"),
    [
        ("INPUT_VARS", "INPUT_VARIABLES"),
        ("HIDDEN_CHANNELS", "MODEL_HIDDEN_CHANNELS"),
        ("DST_PROFILE", "DST_AWS_PROFILE"),
        ("S3_ENDPOINT", "S3_ENDPOINT_URL"),
    ],
)
def test_the_cutoff_also_catches_near_misses(near_miss, meant):
    """Pin the cutoff from below; test_unrelated_variables_do_not_warn pins it from above."""
    assert meant in warn_unknown_env_vars({near_miss: "x"})[near_miss]


# --- the registry must match what the code reads -------------------------------------


class RecordingEnv(dict):
    """A dict that remembers every key looked up in it."""

    def __init__(self, values: dict[str, str]):
        super().__init__(values)
        self.read: set[str] = set()

    def get(self, key, default=None):
        self.read.add(key)
        return super().get(key, default)

    def __getitem__(self, key):
        """Record the key, then look it up."""
        self.read.add(key)
        return super().__getitem__(key)

    def __contains__(self, key):
        """Record the key, then test for it."""
        self.read.add(key)
        return super().__contains__(key)


# S3 URIs exercise the S3 options and the credential probe; local paths exercise the
# "S3 variables set for a non-S3 URI" check instead. Between them, every read happens.
STORE_URIS = {
    "s3": ("s3://src-bucket/in.zarr", "s3://dst-bucket/out.zarr"),
    "local": ("/data/in.zarr", "/data/out.zarr"),
}


def variables_read_by_config(input_zarr: str, output_zarr: str) -> set[str]:
    env = RecordingEnv(
        {
            "INPUT_ZARR": input_zarr,
            "OUTPUT_ZARR": output_zarr,
            "INPUT_VARIABLES": "t2m",
            "OUTPUT_VARIABLES": "t2m:K",
        }
    )
    Config.from_env(env)
    return env.read


def _is_named(node: ast.AST, name: str) -> bool:
    """Match `name` and `<anything>.name`, so `import os as _os` cannot hide a read."""
    if isinstance(node, ast.Name):
        return node.id == name
    return isinstance(node, ast.Attribute) and node.attr == name


def _is_environ(node: ast.AST) -> bool:
    return _is_named(node, "environ")


def _is_getenv(node: ast.AST) -> bool:
    return _is_named(node, "getenv")


def direct_environment_reads(source: str) -> list[tuple[int, str | None]]:
    """Find reads of os.environ / os.getenv that bypass the env mapping.

    Returns (line, name) per read, with name None when the key is not a string literal.
    """
    reads = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Call):
            func = node.func
            on_environ = isinstance(func, ast.Attribute) and _is_environ(func.value)
            if not (_is_getenv(func) or (on_environ and func.attr in ("get", "pop"))):
                continue
            key = node.args[0] if node.args else None
        elif isinstance(node, ast.Subscript) and _is_environ(node.value):
            key = node.slice
        elif isinstance(node, ast.Compare) and any(map(_is_environ, node.comparators)):
            key = node.left
        else:
            continue
        literal = isinstance(key, ast.Constant) and isinstance(key.value, str)
        reads.append((node.lineno, key.value if literal else None))
    return reads


def test_the_source_scanner_sees_every_kind_of_direct_read():
    source = "\n".join(
        [
            "import os",
            "import os as _os",
            "from os import environ, getenv",
            "os.environ.get('A')",
            "os.environ['B']",
            "'C' in os.environ",
            "os.getenv('D')",
            "environ.get('E')",
            "getenv('F')",
            "os.environ.pop('G')",
            "_os.environ.get('H')",
            "os.environ.get(f'{side}_I')",
            "env = os.environ",
        ]
    )
    names = [name for _, name in direct_environment_reads(source)]
    assert sorted(name for name in names if name is not None) == list("ABCDEFGH")
    assert names.count(None) == 1  # the f-string; `env = os.environ` is not a read


@pytest.mark.parametrize("scheme", sorted(STORE_URIS))
def test_every_variable_the_config_reads_is_registered(scheme):
    unregistered = variables_read_by_config(*STORE_URIS[scheme]) - REGISTRY
    assert not unregistered, f"add these to envvars.REGISTRY: {sorted(unregistered)}"


def test_direct_environment_reads_are_registered():
    """Reads that bypass the env mapping must name a literal, registered variable."""
    problems = []
    for path in sorted(PACKAGE.glob("*.py")):
        for line, name in direct_environment_reads(path.read_text()):
            if name is None:
                problems.append(f"{path.name}:{line} reads a computed name")
            elif name not in REGISTRY:
                problems.append(f"{path.name}:{line} reads {name}, which is not registered")
    assert not problems, "\n".join(problems)


def test_the_registry_has_no_stale_entries():
    read = set().union(*(variables_read_by_config(*uris) for uris in STORE_URIS.values()))
    for path in PACKAGE.glob("*.py"):
        read |= {name for _, name in direct_environment_reads(path.read_text()) if name}
    stale = REGISTRY - read
    assert not stale, f"registered but never read, remove from envvars.REGISTRY: {sorted(stale)}"
