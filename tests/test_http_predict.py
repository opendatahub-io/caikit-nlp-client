import pytest

from src.caikit_nlp_client.http_client import HTTPCaikitNlpClient


@pytest.fixture
def http_client(http_config, http_server) -> HTTPCaikitNlpClient:
    """Returns a grpc client connected to a locally running server"""
    return HTTPCaikitNlpClient(http_config)


def test_generate_text(http_client, model_name):
    response = http_client.generate_text(model_name, "What does foobar mean?")
    assert response


def test_generate_text_with_optional_args(http_client, model_name):
    response = http_client.generate_text(
        model_name,
        "What does foobar mean?",
        max_new_tokens=20,
        min_new_tokens=4,
    )
    assert response


def test_generate_text_with_no_model_id(http_client):
    with pytest.raises(ValueError, match="request must have a model id"):
        http_client.generate_text("", "What does foobar mean?")


@pytest.mark.skip(reason="HTTP stream is not working as expected")
def test_generate_text_stream(http_client, model_name):
    results = http_client.generate_text_stream(
        model_name, "What is the meaning of life?"
    )
    assert len(results) == 9


@pytest.mark.skip(reason="HTTP stream is not working as expected")
def test_generate_text_stream_with_optional_args(http_client, model_name):
    results = http_client.generate_text_stream(
        model_name,
        "What is the meaning of life?",
        preserve_input_text=False,
        max_new_tokens=20,
        min_new_tokens=4,
    )
    assert len(results) == 9
