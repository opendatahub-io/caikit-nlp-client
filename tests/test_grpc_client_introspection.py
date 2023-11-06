import pytest

from src.caikit_nlp_client.grpc_client_introspection import (
    GrpcCaikitNlpClientIntrospection,
)


@pytest.fixture
def connected_client(channel):
    """Returns returns a grpc client connected to a locally running server"""
    return GrpcCaikitNlpClientIntrospection(channel)


def test_generate_text(model_name, connected_client):
    generated_text = connected_client.generate_text(
        model_name, "What does foobar mean?"
    )

    assert generated_text


def test_generate_text_with_optional_args(connected_client, model_name):
    generated_text = connected_client.generate_text(
        model_name,
        "What does foobar mean?",
        preserve_input_text=False,
        max_new_tokens=20,
        min_new_tokens=4,
    )
    assert generated_text


def test_generate_text_with_no_model_id(connected_client):
    with pytest.raises(ValueError, match="request must have a model id"):
        connected_client.generate_text("", "What does foobar mean?")


@pytest.mark.xfail(
    reason="BertForSequenceClassification-caikit does not support streaming"
)
def test_generate_text_stream(model_name, connected_client):
    results = connected_client.generate_text_stream(
        model_name, "What is the meaning of life?"
    )
    assert results


@pytest.mark.xfail(
    reason="BertForSequenceClassification-caikit does not support streaming"
)
def test_generate_text_stream_with_optional_args(model_name, connected_client):
    results = connected_client.generate_text_stream(
        model_name,
        "What is the meaning of life?",
        preserve_input_text=False,
        max_new_tokens=20,
        min_new_tokens=4,
    )
    assert results
