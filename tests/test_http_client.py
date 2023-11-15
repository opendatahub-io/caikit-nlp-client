from types import GeneratorType

import pytest
from caikit_nlp_client import HttpClient
from requests.exceptions import SSLError

from .conftest import ConnectionType


def test_generate_text(
    http_client,
    model_name,
    generated_text_result,
):
    response = http_client.generate_text(model_name, "What does foobar mean?")

    assert response == generated_text_result.generated_text


def test_generate_text_with_optional_args(
    http_client,
    model_name,
    generated_text_result,
):
    response = http_client.generate_text(
        model_name,
        "What does foobar mean?",
        max_new_tokens=20,
        min_new_tokens=4,
    )
    assert response == generated_text_result.generated_text
    # TODO: also validate passing of parameters using mocker.spy


def test_generate_text_with_no_model_id(http_client):
    with pytest.raises(ValueError, match="request must have a model id"):
        http_client.generate_text("", "What does foobar mean?")


def test_generate_text_stream(
    pytestconfig, http_client, model_name, generated_text_stream_result
):
    if not pytestconfig.option.real_caikit:
        pytest.skip(
            reason="stream mocking is broken, see https://github.com/opendatahub-io/caikit-nlp-client/issues/46"
        )

    response = http_client.generate_text_stream(
        model_name, "What is the meaning of life?"
    )

    assert isinstance(response, GeneratorType)
    assert list(response) == [
        stream_part.generated_text for stream_part in generated_text_stream_result
    ]


def test_generate_text_stream_with_optional_args(
    pytestconfig, http_client, model_name, generated_text_stream_result
):
    if not pytestconfig.option.real_caikit:
        pytest.skip(
            reason="stream mocking is broken, see https://github.com/opendatahub-io/caikit-nlp-client/issues/46"
        )

    response = http_client.generate_text_stream(
        model_name,
        "What is the meaning of life?",
        preserve_input_text=False,
        max_new_tokens=20,
        min_new_tokens=4,
    )

    assert isinstance(response, GeneratorType)
    assert list(response) == [
        stream_part.generated_text for stream_part in generated_text_stream_result
    ]
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
