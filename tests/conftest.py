import socket
import time
from collections.abc import Iterable
from enum import Enum
from pathlib import Path
from typing import Callable, Optional, TypeVar

import caikit
import pytest
import requests
from caikit_nlp_client.grpc_client import GrpcConfig, make_channel
from caikit_nlp_client.http_client import HttpConfig
from grpc_health.v1 import health_pb2, health_pb2_grpc

from .tls_fixtures import *  # noqa: F403

_T = TypeVar("_T")


WAIT_TIME_OUT = 10


@pytest.fixture(scope="session")
def monkeysession():
    with pytest.MonkeyPatch.context() as mp:
        yield mp


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


class ConnectionType(Enum):
    INSECURE = 1
    TLS = 2
    MTLS = 3


@pytest.fixture(
    autouse=True,
    scope="session",
    params=[ConnectionType.INSECURE, ConnectionType.TLS, ConnectionType.MTLS],
)
def connection_type(request):
    yield request.param


@pytest.fixture(scope="session")
def caikit_nlp_runtime(
    grpc_server_port,
    http_server_port,
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


def get_random_port():
    from contextlib import closing

    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(("", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        port = s.getsockname()[1]
        return port


@pytest.fixture(scope="session")
def grpc_server_port():
    """default port for caikit grpc runtime"""
    return get_random_port()


@pytest.fixture(scope="session")
def http_server_port():
    """default port for caikit grpc runtime"""
    return get_random_port()


def channel_factory(
    host: str,
    port: int,
    connection_type: ConnectionType,
    ca_cert: Optional[bytes] = None,
    client_key: Optional[bytes] = None,
    client_cert: Optional[bytes] = None,
    server_cert: Optional[bytes] = None,
):
    if connection_type is ConnectionType.INSECURE:
        config = GrpcConfig(host=host, port=port)
    elif connection_type is ConnectionType.MTLS:
        config = GrpcConfig(
            host=host,
            port=port,
            mtls=True,
            client_key=client_key,
            client_cert=client_cert,
            server_cert=server_cert,
        )
    else:
        config = GrpcConfig(
            host=host,
            port=port,
            tls=True,
            ca_cert=ca_cert,
            client_key=client_key,
            client_cert=client_cert,
        )

    return make_channel(config)


@pytest.fixture(scope="session")
def grpc_config(
    grpc_server_port, connection_type, ca_cert, client_key, client_cert, server_cert
):
    if connection_type is ConnectionType.INSECURE:
        return GrpcConfig(host="localhost", port=grpc_server_port)

    if connection_type is ConnectionType.TLS:
        return GrpcConfig(
            host="localhost", port=grpc_server_port, tls=True, ca_cert=ca_cert
        )

    if connection_type is ConnectionType.MTLS:
        return GrpcConfig(
            host="localhost",
            port=grpc_server_port,
            mtls=True,
            client_cert=client_cert,
            client_key=client_key,
            server_cert=server_cert,
            ca_cert=ca_cert,
        )

    raise ValueError(f"invalid {connection_type=}")


@pytest.fixture(scope="session")
def channel(
    grpc_server_port,
    grpc_server,
    connection_type,
    ca_cert,
    client_key,
    client_cert,
    server_cert,
):
    """Returns returns a grpc client connected to a locally running server"""
    return channel_factory(
        "localhost",
        grpc_server_port,
        connection_type,
        ca_cert,
        client_key,
        client_cert,
        server_cert,
    )


@pytest.fixture(scope="session")
def grpc_server(
    caikit_nlp_runtime,
    grpc_server_port,
    mock_text_generation,
    connection_type,
    ca_cert,
    client_key,
    client_cert,
    server_cert,
):
    from caikit.runtime.grpc_server import RuntimeGRPCServer

    grpc_server = RuntimeGRPCServer()
    grpc_server.start(blocking=False)

    def health_check():
        if connection_type is ConnectionType.INSECURE:
            kwargs = {}
        elif connection_type is ConnectionType.TLS:
            kwargs = {
                "ca_cert": ca_cert,
            }
        elif connection_type is ConnectionType.MTLS:
            kwargs = {
                "server_cert": server_cert,
                "client_cert": client_cert,
                "client_key": client_key,
                "ca_cert": ca_cert,
            }
        else:
            raise ValueError(f"Unknown {connection_type=}")

        channel = channel_factory(
            "localhost", grpc_server_port, connection_type, **kwargs
        )
        stub = health_pb2_grpc.HealthStub(channel)
        health_check_request = health_pb2.HealthCheckRequest()
        stub.Check(health_check_request)

    wait_until(health_check, timeout=WAIT_TIME_OUT, pause=0.5)

    yield grpc_server

    grpc_server.stop()


@pytest.fixture(scope="session")
def http_config(
    caikit_nlp_runtime,
    connection_type,
    client_cert_file,
    client_key_file,
    ca_cert_file,
    server_cert_file,
):
    http_config = HttpConfig(
        host="localhost",
        port=caikit.config.get_config().runtime.http.port,
    )
    if connection_type is ConnectionType.INSECURE:
        return http_config

    if connection_type is ConnectionType.TLS:
        http_config.tls = True
        http_config.ca_cert_file = ca_cert_file
        return http_config

    if connection_type is ConnectionType.MTLS:
        http_config.mtls = True
        http_config.client_crt_path = client_cert_file
        http_config.client_key_path = client_key_file
        http_config.ca_crt_path = ca_cert_file

        return http_config

    raise ValueError(f"invalid {connection_type=}")


@pytest.fixture(scope="session")
def caikit_test_producer():
    from caikit.interfaces.nlp.data_model.text_generation import ProducerId

    yield ProducerId(
        name="Testing Producer",
        version="0.0.1",
    )


@pytest.fixture(scope="session")
def generated_text():
    yield "mocked generated text result"


@pytest.fixture(scope="session")
def generated_text_result(caikit_test_producer, generated_text):
    from caikit.interfaces.nlp.data_model.text_generation import (
        FinishReason,
        GeneratedTextResult,
    )

    yield GeneratedTextResult(
        generated_text="mocked generated text result",
        generated_tokens=42,
        finish_reason=FinishReason.EOS_TOKEN,
        producer_id=caikit_test_producer,
        input_token_count=10,
        seed=None,  # Optional[np.uint64]
    )


# FIXME: Validate text stream mocking. There's a lot of logic here.
#        Can this be simplified?
@pytest.fixture(scope="session")
def generated_text_stream_result(caikit_test_producer, generated_text):
    from caikit.interfaces.nlp.data_model.text_generation import (
        FinishReason,
        GeneratedTextStreamResult,
        GeneratedToken,
        TokenStreamDetails,
    )

    split_text = generated_text.split(" ")

    # TODO: validate token_list
    token_list = [GeneratedToken(text="dummy generated token value", logprob=0.42)]
    input_token_count = len(split_text)

    result = []
    for text in split_text:
        details = TokenStreamDetails(
            finish_reason=FinishReason.NOT_FINISHED,
            generated_tokens=42,  # FIXME: is this correct?
            seed=None,
            input_token_count=input_token_count,
        )

        stream_result = GeneratedTextStreamResult(
            generated_text=text,
            tokens=token_list,
            details=details,
        )

        result.append(stream_result)

    result[-1].details.finish_reason = FinishReason.EOS_TOKEN
    return result


@pytest.fixture(scope="session")
def mock_text_generation(
    generated_text_result, generated_text_stream_result, monkeysession
):
    # import caikit_nlp.modules.text_generation.text_generation_local
    import caikit_nlp.modules.text_generation.text_generation_tgis
    from caikit.interfaces.nlp.data_model.text_generation import (
        GeneratedTextResult,
        GeneratedTextStreamResult,
    )

    # NOTE: config uses tgis, so this is not really required,
    #       unless we want to test the local text generation module
    # monkeypatch.setattr(
    #     caikit_nlp.modules.text_generation.text_generation_local,
    #     "generate_text_func",
    #     lambda: generated_text_result,
    # )

    class StubTGISGenerationClient:
        def __init__(self, *args, **kwargs):
            pass

        def unary_generate(self, *args, **kwargs) -> GeneratedTextResult:
            return generated_text_result

        def stream_generate(
            self, *args, **kwargs
        ) -> Iterable[GeneratedTextStreamResult]:
            yield from generated_text_stream_result

    monkeysession.setattr(
        caikit_nlp.modules.text_generation.text_generation_tgis,
        "TGISGenerationClient",
        StubTGISGenerationClient,
    )

    yield


@pytest.fixture(scope="session")
def http_server(
    caikit_nlp_runtime,
    http_config,
    mock_text_generation,
    connection_type,
    ca_cert_file,
    client_cert_file,
    client_key_file,
):
    from caikit.runtime.http_server import RuntimeHTTPServer

    http_server = RuntimeHTTPServer()
    http_server.start(blocking=False)

    def health_check():
        if connection_type is ConnectionType.INSECURE:
            scheme = "http"
            kwargs = {}
        elif connection_type is ConnectionType.TLS:
            scheme = "https"
            kwargs = {"verify": ca_cert_file}
        elif connection_type is ConnectionType.MTLS:
            scheme = "https"
            kwargs = {
                "verify": ca_cert_file,
                "cert": (client_cert_file, client_key_file),
            }
        else:
            raise ValueError(f"Invalid {connection_type=}")
        response = requests.get(
            f"{scheme}://{http_config.host}:{http_config.port}/health", **kwargs
        )

        assert response.status_code == 200
        assert response.text == "OK"

    wait_until(health_check, timeout=WAIT_TIME_OUT, pause=0.5)

    yield http_server

    http_server.stop()
