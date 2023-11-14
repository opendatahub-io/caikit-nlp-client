from collections.abc import Iterable
from pathlib import Path

import caikit
import pytest

from .fixtures.grpc import grpc_client, grpc_server, grpc_server_port  # noqa: F401
from .fixtures.http import http_config, http_server, http_server_port  # noqa: F401
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
