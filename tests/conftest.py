from pathlib import Path

import caikit
import pytest

from .fixtures.grpc import (  # noqa: F401
    grpc_client,
    grpc_server,
    grpc_server_port,
)
from .fixtures.http import (  # noqa: F401
    http_config,
    http_server,
    http_server_port,
)
from .fixtures.mocked_results import *  # noqa: F403
from .fixtures.tls import *  # noqa: F403
from .fixtures.utils import ConnectionType


@pytest.fixture(scope="session")
def monkeysession():
    with pytest.MonkeyPatch.context() as mp:
        yield mp


@pytest.fixture
def model_name():
    """name of the model utilized by the tests. Has to be in `tests/tiny_models`"""
    # Note that this can be overridden in tests via indirect parametrization
    available_models = [
        "BertForSequenceClassification-caikit",
        "BloomForCausalLM-caikit",
        "T5ForConditionalGeneration-caikit",
    ]
    return available_models[0]


@pytest.fixture(
    autouse=True,
    scope="session",
    params=[ConnectionType.INSECURE, ConnectionType.TLS, ConnectionType.MTLS],
)
def connection_type(request):
    yield request.param


@pytest.fixture(scope="session")
def caikit_nlp_runtime(
    grpc_server_port,  # noqa: F811
    http_server_port,  # noqa: F811
    connection_type,
    server_key_file,
    server_cert_file,
    ca_cert_file,
):
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
            "grpc": {"enabled": True, "port": grpc_server_port},
            "http": {
                "enabled": True,
                "port": http_server_port,
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
