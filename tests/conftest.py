from pathlib import Path

import caikit
import pytest

from .fixtures.docker import *
from .fixtures.grpc import *
from .fixtures.http import *
from .fixtures.mocked_results import *
from .fixtures.tls import *
from .fixtures.utils import ConnectionType


def pytest_configure(config):
    pass


@pytest.fixture(scope="session")
def monkeysession():
    with pytest.MonkeyPatch.context() as mp:
        yield mp


@pytest.fixture
def model_name(request: pytest.FixtureRequest):
    """name of the model utilized by the tests. Has to be in `tests/tiny_models`"""
    # Note that this can be overridden in tests via indirect parametrization
    if "caikit_tgis_service" in request.fixturenames:
        return "flan-t5-small-caikit"

    available_models = [
        "T5ForConditionalGeneration-caikit",
        "BertForSequenceClassification-caikit",
        "BloomForCausalLM-caikit",
    ]
    return available_models[0]


@pytest.fixture(
    scope="session",
    params=[ConnectionType.INSECURE, ConnectionType.TLS, ConnectionType.MTLS],
)
def connection_type(request):
    yield request.param


@pytest.fixture(scope="session")
def caikit_nlp_runtime(
    grpc_server_thread_port,  # noqa: F811
    http_server_thread_port,  # noqa: F811
    connection_type,
    server_key_file,
    server_cert_file,
    ca_cert_file,
):
    """configures caikit for local testing"""
    models_directory = str(Path(__file__).parent / "tiny_models")

    tgis_backend_config = [
        {
            "type": "TGIS",
            "config": {"connection": {"hostname": "localhost:8033"}},
        }
    ]
    config = {
        "merge_strategy": "merge",
        "runtime": {
            "metrics": {"enabled": False},
            "local_models_dir": models_directory,
            "library": "caikit_nlp",
            "lazy_load_local_models": True,
            "grpc": {"enabled": True, "port": grpc_server_thread_port},
            "http": {
                "enabled": True,
                "port": http_server_thread_port,
                "server_shutdown_grace_period_seconds": 2,
            },
        },
        "model_management": {
            "initializers": {
                "default": {
                    "type": "LOCAL",
                    "config": {"backend_priority": tgis_backend_config},
                }
            }
        },
        "log": {"formatter": "pretty"},
    }
    if connection_type is ConnectionType.TLS:
        config["runtime"]["tls"] = {
            "server": {
                "key": server_key_file,
                "cert": server_cert_file,
            }
        }

    if connection_type is ConnectionType.MTLS:
        config["runtime"]["tls"] = {
            "server": {
                "key": server_key_file,
                "cert": server_cert_file,
            },
            "client": {
                "cert": ca_cert_file,
            },
        }

    caikit.config.configure(config_dict=config)
