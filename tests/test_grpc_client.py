from types import GeneratorType

import pytest
from caikit_nlp_client.grpc_client import GrpcClient

from .fixtures.utils import ConnectionType


def test_generate_text(model_name, grpc_client, prompt):
    generated_text = grpc_client.generate_text(model_name, prompt)

    assert isinstance(generated_text, str)
    assert generated_text


def test_generate_text_with_optional_args(grpc_client, model_name, prompt):
    generated_text = grpc_client.generate_text(
        model_name,
        prompt,
        preserve_input_text=False,
        max_new_tokens=20,
        min_new_tokens=4,
    )

    assert isinstance(generated_text, str)
    assert generated_text
    # TODO: also validate passing of parameters


@pytest.mark.parametrize("connection_type", [ConnectionType.INSECURE], scope="session")
def test_context_manager(mocker, grpc_server, connection_type, request, model_name):
    with GrpcClient(*grpc_server, insecure=True) as client:
        close_spy = mocker.spy(client, "_close")
        assert client.generate_text(model_name, "dummy text")

        channel = client._channel
        channel_close_spy = mocker.spy(channel, "close")

    close_spy.assert_called()
    channel_close_spy.assert_called()


def test_generate_text_with_no_model_id(grpc_client):
    with pytest.raises(ValueError, match="request must have a model id"):
        grpc_client.generate_text("", "What does foobar mean?")


def test_generate_text_stream(
    model_name, grpc_client, generated_text_stream_result, prompt
):
    response = grpc_client.generate_text_stream(
        model_name,
        prompt,
    )

    assert isinstance(response, GeneratorType)
    response_list = list(response)
    assert response_list
    assert all(isinstance(text, str) for text in response_list)


def test_generate_text_stream_with_optional_args(
    model_name, grpc_client, generated_text_stream_result, prompt
):
    response = grpc_client.generate_text_stream(
        model_name,
        prompt,
        preserve_input_text=False,
        max_new_tokens=20,
        min_new_tokens=4,
    )

    assert isinstance(response, GeneratorType)
    response_list = list(response)
    assert response_list
    assert all(isinstance(text, str) for text in response_list)


def test_request_invalid_kwarg(model_name, grpc_client):
    with pytest.raises(ValueError, match="Unsupported kwarg key='invalid_kwarg'"):
        grpc_client.generate_text(model_name, "dummy", invalid_kwarg=42)
