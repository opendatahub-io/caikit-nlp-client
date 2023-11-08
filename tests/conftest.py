import socket
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Callable, TypeVar

import caikit
import pytest
import requests
from caikit_nlp_client.grpc_channel import GrpcChannelConfig, make_channel
from caikit_nlp_client.http_client import HTTPConfig
from grpc_health.v1 import health_pb2, health_pb2_grpc

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
def caikit_nlp_runtime(grpc_server_port, http_server_port):
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
            "local_models_dir": models_directory,
            "library": "caikit_nlp",
            "lazy_load_local_models": True,
            "grpc": {"enabled": True, "port": grpc_server_port},
            "http": {"enabled": True, "port": http_server_port},
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

    caikit.config.configure(config_dict=config)


def get_random_port():
    sock = socket.socket()
    sock.bind(("", 0))
    return sock.getsockname()[1]


@pytest.fixture
def grpc_server_port():
    """port for caikit grpc runtime"""
    return get_random_port()


@pytest.fixture()
def http_server_port():
    """port for caikit grpc runtime"""
    return get_random_port()


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


@pytest.fixture
def grpc_server(caikit_nlp_runtime, grpc_server_port, mock_text_generation):
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
def caikit_test_producer():
    from caikit.interfaces.nlp.data_model.text_generation import ProducerId

    yield ProducerId(
        name="Testing Producer",
        version="0.0.1",
    )


@pytest.fixture
def generated_text():
    yield "mocked generated text result"


@pytest.fixture
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
@pytest.fixture
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


@pytest.fixture
def mock_text_generation(
    generated_text_result, generated_text_stream_result, monkeypatch
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

    monkeypatch.setattr(
        caikit_nlp.modules.text_generation.text_generation_tgis,
        "TGISGenerationClient",
        StubTGISGenerationClient,
    )

    yield


@pytest.fixture
def http_server(caikit_nlp_runtime, http_config, mock_text_generation):
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
