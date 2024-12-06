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
def model_name(request: pytest.FixtureRequest, using_tgis_backend):
    """name of the model utilized by the tests. Has to be in `tests/tiny_models`"""
    # Note that this can be overridden in tests via indirect parametrization
    if using_tgis_backend:
        return "flan-t5-small-caikit"

    available_models = [
        "T5ForConditionalGeneration-caikit",
        "BertForSequenceClassification-caikit",
        "BloomForCausalLM-caikit",
    ]
    return available_models[0]


@pytest.fixture
def embedding_model_name(model_name: str):
    return f"{model_name}-embedding"


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
    using_tgis_backend,
):
    """configures caikit for local testing"""
    models_directory = str(Path(__file__).parent / "tiny_models")

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
        "log": {"formatter": "pretty"},
    }

    if using_tgis_backend:
        config["model_management"] = (
            {
                "initializers": {
                    "default": {
                        "type": "LOCAL",
                        "config": {
                            "backend_priority": {
                                "type": "TGIS",
                                "config": {
                                    "connection": {"hostname": "localhost:8033"}
                                },
                            }
                        },
                    }
                }
            },
        )

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
