import pytest
from requests.exceptions import SSLError

from .conftest import ConnectionType


def test_generate_text(
    http_client,
    model_name,
    generated_text_result,
    monkeysession,
    ca_cert_file,
    connection_type,
):
    if connection_type is ConnectionType.TLS:
        # a valid certificate autority should validate the response with no extra args
        with monkeysession.context() as monkeypatch:
            monkeypatch.setenv("REQUESTS_CA_BUNDLE", ca_cert_file)
            response = http_client.generate_text(model_name, "What does foobar mean?")
    else:
        response = http_client.generate_text(model_name, "What does foobar mean?")

    assert response


def test_generate_text_with_optional_args(
    http_client,
    model_name,
    generated_text_result,
    monkeysession,
    ca_cert_file,
    connection_type,
):
    if connection_type is ConnectionType.TLS:
        # a valid certificate autority should validate the response with no extra args
        with monkeysession.context() as monkeypatch:
            monkeypatch.setenv("REQUESTS_CA_BUNDLE", ca_cert_file)
            response = http_client.generate_text(
                model_name,
                "What does foobar mean?",
                max_new_tokens=20,
                min_new_tokens=4,
            )

    else:
        response = http_client.generate_text(
            model_name,
            "What does foobar mean?",
            max_new_tokens=20,
            min_new_tokens=4,
        )
    assert response
    # TODO: also validate passing of parameters


def test_generate_text_with_no_model_id(http_client):
    with pytest.raises(ValueError, match="request must have a model id"):
        http_client.generate_text("", "What does foobar mean?")


@pytest.mark.skip(reason="stream is broken")
def test_generate_text_stream(http_client, model_name, generated_text_stream_result):
    result = http_client.generate_text_stream(
        model_name, "What is the meaning of life?"
    )
    assert result == [
        stream_part.generated_text for stream_part in generated_text_stream_result
    ]


@pytest.mark.skip(reason="stream is broken")
def test_generate_text_stream_with_optional_args(
    http_client, model_name, generated_text_stream_result
):
    result = http_client.generate_text_stream(
        model_name,
        "What is the meaning of life?",
        preserve_input_text=False,
        max_new_tokens=20,
        min_new_tokens=4,
    )
    assert result == [
        stream_part.generated_text for stream_part in generated_text_stream_result
    ]


@pytest.mark.parametrize("connection_type", [ConnectionType.TLS], indirect=True)
def test_tls_enabled(
    http_client,
    model_name,
    http_server,
    monkeysession,
    ca_cert_file,
    connection_type,
):
    assert connection_type is ConnectionType.TLS, "TLS should be enabled for this test"

    with pytest.raises(SSLError, match=".*CERTIFICATE_VERIFY_FAILED.*"):
        assert http_client.generate_text(model_name, "dummy text")

    # a valid certificate autority should validate the response with no extra args
    with monkeysession.context() as monkeypatch:
        monkeypatch.setenv("REQUESTS_CA_BUNDLE", ca_cert_file)

        assert http_client.generate_text(model_name, "dummy text")
