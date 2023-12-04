from types import GeneratorType

import pytest
import requests
from caikit_nlp_client import HttpClient
from requests.exceptions import SSLError

from .conftest import ConnectionType


def test_client_invalid_args():
    with pytest.raises(
        TypeError, match=".* missing 1 required positional argument: 'base_url'"
    ):
        HttpClient()

    with pytest.raises(ValueError, match="Cannot use verify=False with ca_cert_path"):
        HttpClient("dummy_base_url", verify=False, ca_cert_path="dummy")


def test_generate_text(
    http_client, model_name, prompt, mocker, accept_self_signed_certs
):
    generated_text = http_client.generate_text(model_name, prompt)
    assert isinstance(generated_text, str)
    assert generated_text


def test_create_json_request():
    client = HttpClient("dummyurl")
    assert client._create_json_request("dummymodel", "dummytext") == {
        "model_id": "dummymodel",
        "inputs": "dummytext",
    }

    assert client._create_json_request(
        "dummymodel", "dummytext", max_new_tokens=42, min_new_tokens=1
    ) == {
        "model_id": "dummymodel",
        "inputs": "dummytext",
        "parameters": {
            "max_new_tokens": 42,
            "min_new_tokens": 1,
        },
    }
    assert client._create_json_request(
        "dummymodel", "dummytext", example_parameter="value"
    ) == {
        "model_id": "dummymodel",
        "inputs": "dummytext",
        "parameters": {
            "example_parameter": "value",
        },
    }


def test_generate_text_with_optional_args(
    http_client,
    model_name,
    generated_text_result,
    prompt,
    mocker,
    accept_self_signed_certs,
):
    mock = mocker.spy(requests, "post")

    generated_text = http_client.generate_text(
        model_name, prompt, max_new_tokens=20, min_new_tokens=4
    )

    assert isinstance(generated_text, str)
    assert generated_text
    assert mock.call_args_list[-1].kwargs["json"]["parameters"]["max_new_tokens"] == 20
    assert mock.call_args_list[-1].kwargs["json"]["parameters"]["min_new_tokens"] == 4


def test_timeout_kwarg(
    http_client, model_name, prompt, mocker, accept_self_signed_certs
):
    mock = mocker.spy(requests, "post")

    http_client.generate_text(model_name, prompt)
    assert mock.call_args_list[-1].kwargs["timeout"] == 60.0

    http_client.generate_text(model_name, prompt, timeout=42.0)
    assert mock.call_args_list[-1].kwargs["timeout"] == 42.0


def test_generate_text_with_no_model_id(http_client):
    with pytest.raises(ValueError, match="request must have a model id"):
        http_client.generate_text("", "dummy")


def test_generate_text_stream(
    pytestconfig,
    http_client,
    model_name,
    generated_text_stream_result,
    prompt,
    accept_self_signed_certs,
):
    if not pytestconfig.option.real_caikit:
        pytest.skip(
            reason="stream mocking is broken, see https://github.com/opendatahub-io/caikit-nlp-client/issues/46"
        )

    response = http_client.generate_text_stream(
        model_name,
        prompt,
    )

    assert isinstance(response, GeneratorType)
    response_list = list(response)
    assert response_list
    assert all(isinstance(text, str) for text in response_list)


def test_generate_text_stream_with_optional_args(
    pytestconfig,
    http_client,
    model_name,
    generated_text_stream_result,
    prompt,
    accept_self_signed_certs,
    mocker,
):
    if not pytestconfig.option.real_caikit:
        pytest.skip(
            reason="stream mocking is broken, see https://github.com/opendatahub-io/caikit-nlp-client/issues/46"
        )

    mock = mocker.spy(requests, "post")
    response = http_client.generate_text_stream(
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
    assert mock.call_args_list[-1].kwargs["json"]["parameters"]["max_new_tokens"] == 20
    assert mock.call_args_list[-1].kwargs["json"]["parameters"]["min_new_tokens"] == 4


def test_request_exception_handling(
    using_real_caikit,
    http_client,
    mock_text_generation,
    model_name,
    mocker,
    accept_self_signed_certs,
    pytestconfig,
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
        match_stream = (
            f"{stream_exc_prefix} Unhandled exception: Value out of range: -1"
        )
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
        http_client.generate_text(
            model_name,
            prompt,
            **kwargs,
        )
    if pytestconfig.option.real_caikit:
        streaming_response = http_client.generate_text_stream(
            model_name, prompt, **kwargs
        )
        with pytest.raises(
            RuntimeError,
            match=match_stream,
        ):
            list(streaming_response)


def test_get_text_generation_parameters(
    http_client, monkeysession, accept_self_signed_certs
):
    assert http_client.get_text_generation_parameters() == {
        "max_new_tokens": "integer",
        "min_new_tokens": "integer",
        "truncate_input_tokens": "integer",
        "decoding_method": "string",
        "top_k": "integer",
        "top_p": "number",
        "typical_p": "number",
        "temperature": "number",
        "repetition_penalty": "number",
        "max_time": "number",
        "exponential_decay_length_penalty": {
            "start_index": "integer",
            "decay_factor": "number",
        },
        "stop_sequences": "array",
        "seed": "integer",
        "preserve_input_text": "boolean",
    }


@pytest.mark.parametrize("connection_type", [ConnectionType.TLS], indirect=True)
def test_tls_enabled(
    model_name,
    http_server,
    monkeysession,
    ca_cert_file,
    connection_type,
):
    assert connection_type is ConnectionType.TLS, "TLS should be enabled for this test"

    http_client = HttpClient("https://{}:{}".format(*http_server))

    with pytest.raises(SSLError, match=".*CERTIFICATE_VERIFY_FAILED.*"):
        assert http_client.generate_text(model_name, "dummy text")

    # a valid certificate autority should validate the response with no extra args
    with monkeysession.context() as monkeypatch:
        monkeypatch.setenv("REQUESTS_CA_BUNDLE", ca_cert_file)

        assert http_client.generate_text(model_name, "dummy text")

    # setting verify=False should make the request go through
    http_client = HttpClient("https://{}:{}".format(*http_server), verify=False)
    assert http_client.generate_text(model_name, "dummy text")


@pytest.mark.parametrize("connection_type", [ConnectionType.MTLS], indirect=True)
def test_client_instantiation(
    ca_cert_file,
    client_key_file,
    client_cert_file,
    connection_type,
):
    """mTLS tests"""

    # should not raise when providing a CA bundle
    HttpClient(
        "https://localhost:8080",
        ca_cert_path=ca_cert_file,
    )
    with pytest.raises(
        ValueError,
        match="Must provide both client_cert_path and client_key_path for mTLS",
    ):
        HttpClient(
            "https://localhost:8080",
            ca_cert_path=ca_cert_file,
            client_cert_path=client_cert_file,
        )
        HttpClient(
            "https://localhost:8080",
            ca_cert_path=ca_cert_file,
            client_key_path=client_key_file,
        )
        HttpClient(
            "https://localhost:8080",
            client_cert_path=client_cert_file,
            client_key_path=client_key_file,
        )
