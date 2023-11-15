from types import GeneratorType

import pytest
from caikit_nlp_client import HttpClient
from requests.exceptions import SSLError

from .conftest import ConnectionType


def test_generate_text(http_client, model_name, prompt, mocker):
    import requests

    mock = mocker.spy(requests, "post")

    generated_text = http_client.generate_text(model_name, prompt)

    assert isinstance(generated_text, str)
    assert generated_text
    assert "timeout" in mock.call_args_list[0].kwargs


def test_generate_text_with_optional_args(
    http_client, model_name, generated_text_result, prompt, mocker
):
    import requests

    mock = mocker.spy(requests, "post")

    generated_text = http_client.generate_text(
        model_name, prompt, timeout=42.0, max_new_tokens=20, min_new_tokens=4
    )

    assert isinstance(generated_text, str)
    assert generated_text
    assert mock.call_args_list[-1].kwargs["timeout"] == 42.0
    # TODO: also validate passing of parameters using mocker.spy


def test_generate_text_with_no_model_id(http_client):
    with pytest.raises(ValueError, match="request must have a model id"):
        http_client.generate_text("", "dummy")


def test_generate_text_stream(
    pytestconfig, http_client, model_name, generated_text_stream_result, prompt
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
    pytestconfig, http_client, model_name, generated_text_stream_result, prompt
):
    if not pytestconfig.option.real_caikit:
        pytest.skip(
            reason="stream mocking is broken, see https://github.com/opendatahub-io/caikit-nlp-client/issues/46"
        )

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
    # TODO: verify passing of parameters using mocker.spy


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
