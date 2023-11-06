import time
from pathlib import Path
from typing import Callable, TypeVar

import pytest
import requests
from grpc_health.v1 import health_pb2, health_pb2_grpc

from src.caikit_nlp_client.grpc_channel import GrpcChannelConfig, make_channel
from src.caikit_nlp_client.http_config import HTTPConfig

_T = TypeVar("_T")


def wait_until(pred: Callable[..., _T], timeout: float, pause: float = 0.1) -> _T:
    start = time.perf_counter()
    exc = None
    while (time.perf_counter() - start) < timeout:
        try:
            value = pred()
        except Exception as e:  # pylint: disable=broad-except
            exc = e
        else:
            return value
        time.sleep(pause)

    raise TimeoutError("timed out waiting") from exc


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


@pytest.fixture
def caikit_nlp_runtime(monkeypatch):
    """configures caikit by setting environment variables"""
    monkeypatch.setenv("RUNTIME_LIBRARY", "caikit_nlp")
    # make caikit runtime logs readable: it logs in json by default
    monkeypatch.setenv("LOG_FORMATTER", "pretty")
    monkeypatch.setenv(
        "RUNTIME_LOCAL_MODELS_DIR",
        str(Path(__file__).parent / "tiny_models"),
    )
    monkeypatch.setenv("RUNTIME_HTTP_PORT", "8080")
    # TODO: add port configuration here using `RUNTIME_GRPC_PORT`


@pytest.fixture()
def grpc_server_port():
    """default port for caikit grpc runtime"""
    return 8085


def channel_factory(host: str, port: int):
    config = GrpcChannelConfig(
        host=host,
        port=port,
        insecure=True,  # TODO: handle cases with secure=True with self signed certs
    )
    return make_channel(config)


@pytest.fixture
def channel(grpc_server_port, grpc_server):
    """Returns returns a grpc client connected to a locally running server"""
    return channel_factory("localhost", grpc_server_port)


# TODO: make this session (or module) scoped, this will mean having to fix
# the configuration so that it can be session-scoped (monkeypatch does not like this)
@pytest.fixture
def grpc_server(caikit_nlp_runtime, grpc_server_port):
    from caikit.runtime.grpc_server import RuntimeGRPCServer

    grpc_server = RuntimeGRPCServer()
    grpc_server.start(blocking=False)

    def health_check():
        channel = channel_factory("localhost", grpc_server_port)
        stub = health_pb2_grpc.HealthStub(channel)
        health_check_request = health_pb2.HealthCheckRequest()
        stub.Check(health_check_request)

    wait_until(health_check, timeout=30, pause=0.5)

    yield grpc_server

    grpc_server.stop()


@pytest.fixture
def http_config(caikit_nlp_runtime):
    from caikit.config import get_config

    return HTTPConfig(
        host="localhost",
        port=get_config().runtime.http.port,
        tls=False,
    )


@pytest.fixture
def http_server(caikit_nlp_runtime, http_config):
    from caikit.runtime.http_server import RuntimeHTTPServer

    http_server = RuntimeHTTPServer()
    http_server.start(blocking=False)

    def health_check():
        response = requests.get(
            f"http://{http_config.host}:{http_config.port}/health",
        )
        assert response.status_code == 200
        assert response.text == "OK"

    wait_until(health_check, timeout=30, pause=0.5)

    yield http_server

    http_server.stop()
