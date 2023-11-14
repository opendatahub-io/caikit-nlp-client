import pytest


def test_generate_text(model_name, grpc_client, generated_text_result):
    generated_text = grpc_client.generate_text(model_name, "What does foobar mean?")

    assert generated_text == generated_text_result.generated_text


def test_generate_text_with_optional_args(
    grpc_client, model_name, generated_text_result
):
    generated_text = grpc_client.generate_text(
        model_name,
        "What does foobar mean?",
        preserve_input_text=False,
        max_new_tokens=20,
        min_new_tokens=4,
    )
    assert generated_text == generated_text_result.generated_text
    # TODO: also validate passing of parameters


def test_generate_text_with_no_model_id(grpc_client):
    with pytest.raises(ValueError, match="request must have a model id"):
        grpc_client.generate_text("", "What does foobar mean?")


def test_generate_text_stream(model_name, grpc_client, generated_text_stream_result):
    result = grpc_client.generate_text_stream(
        model_name, "What is the meaning of life?"
    )

    assert result == [
        stream_part.generated_text for stream_part in generated_text_stream_result
    ]


def test_generate_text_stream_with_optional_args(
    model_name, grpc_client, generated_text_stream_result
):
    result = grpc_client.generate_text_stream(
        model_name,
        "What is the meaning of life?",
        preserve_input_text=False,
        max_new_tokens=20,
        min_new_tokens=4,
    )
    assert result == [
        stream_part.generated_text for stream_part in generated_text_stream_result
    ]
    # TODO: also validate passing of parameters


def test_request_invalid_kwarg(model_name, grpc_client):
    with pytest.raises(ValueError, match="Unsupported kwarg key='invalid_kwarg'"):
        grpc_client.generate_text(model_name, "dummy", invalid_kwarg=42)
