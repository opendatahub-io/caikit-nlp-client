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

    with pytest.raises(ValueError, match="Unsupported kwarg key='invalid_kwarg'"):
        grpc_client.generate_text(model_name, "dummy", invalid_kwarg=42)


def test_request_exception_handling(
    using_real_caikit,
    grpc_client,
    mock_text_generation,
    model_name,
):
    """force generation of an exception at text generation time to make
    sure the client returns useful information,"""
    exc_prefix = (
        "Exception raised during inference. This may be a problem with your input:"
    )
    stream_exc_prefix = "Exception iterating responses:"
    if using_real_caikit:
        prompt = "dummy"
        detail = "Value out of range: -1"
        match = f"{exc_prefix} {detail}"
        match_stream = f"{stream_exc_prefix} {detail}"
        kwargs = {
            # provide invalid kwargs
            "min_new_tokens": -1,
        }
    else:
        # mock_text_generation raises an exception when [[raise exception]] is in
        # the input text
        detail = "user requested an exception"
        prompt = "[[raise exception]] dummy"
        match = f"{exc_prefix} {detail}"
        match_stream = f"{stream_exc_prefix} {detail}"
        kwargs = {}

    with pytest.raises(
        RuntimeError,
        match=match,
    ):
        grpc_client.generate_text(
            model_name,
            prompt,
            **kwargs,
        )

    streaming_response = grpc_client.generate_text_stream(model_name, prompt, **kwargs)
    with pytest.raises(
        RuntimeError,
        match=match_stream,
    ):
        list(streaming_response)


def test_get_text_generation_parameters(grpc_client):
    assert grpc_client.get_text_generation_parameters() == {
        "text": "string",
        "max_new_tokens": "int64",
        "min_new_tokens": "int64",
        "truncate_input_tokens": "int64",
        "decoding_method": "string",
        "top_k": "int64",
        "top_p": "double",
        "typical_p": "double",
        "temperature": "double",
        "seed": "uint64",
        "repetition_penalty": "double",
        "max_time": "double",
        "exponential_decay_length_penalty": {
            "start_index": "int64",
            "decay_factor": "double",
        },
        "stop_sequences": "string",
        "preserve_input_text": "bool",
    }


def test_invalid_init_options(grpc_server):
    with pytest.raises(ValueError, match="insecure cannot be used with verify"):
        GrpcClient(*grpc_server, insecure=True, verify=True)

    for kwargs in (
        {"ca_cert": "dummy"},
        {"client_key": "dummy"},
        {"client_cert": "dummy"},
    ):
        with pytest.raises(
            ValueError, match="cannot use insecure with TLS/mTLS certificates"
        ):
            GrpcClient(*grpc_server, insecure=True, **kwargs)


def test_verify(model_name, grpc_server, connection_type, client_key):
    """test verify kwarg for TLS connections"""
    if connection_type != ConnectionType.TLS:
        return

    with pytest.raises(
        RuntimeError,
        match="Could not connect to localhost:.*:failed to connect to all addresses;.*",
    ):
        grpc_client = GrpcClient(*grpc_server)

    grpc_client = GrpcClient(*grpc_server, verify=False)

    assert grpc_client.generate_text(model_name, "dummy text")


def test_grpc_client_with_bogus_certificate_files(grpc_server):
    for kwargs in (
        {"ca_cert": "/some/random/path/cert.pem"},
        {"client_key": "/some/random/path/cert.pem"},
        {"client_cert": "/some/random/path/cert.pem"},
    ):
        with pytest.raises(
            FileNotFoundError,
            match=".*No such file or directory.*",
        ):
            GrpcClient(
                *grpc_server,
                insecure=False,
                **kwargs,
            )


def test_grpc_client_load_certificate(ca_cert, ca_cert_file):
    assert ca_cert == GrpcClient._try_load_certificate(ca_cert)
    assert ca_cert == GrpcClient._try_load_certificate(ca_cert_file)
    assert GrpcClient._try_load_certificate(None) is None

    with pytest.raises(
        FileNotFoundError,
        match=".*No such file or directory.*",
    ):
        GrpcClient._try_load_certificate("/some/random/path/cert.pem")

    with pytest.raises(
        ValueError, match=".*should be a path to a certificate files or bytes"
    ):
        GrpcClient._try_load_certificate(("Pinky", "Brain"))
