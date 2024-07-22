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

    with pytest.raises(ValueError, match="Unsupported kwarg: invalid_kwarg=42"):
        grpc_client.generate_text(model_name, "dummy", invalid_kwarg=42)


def test_request_exception_handling(
    using_real_caikit,
    grpc_client,
    mock_text_generation,
    model_name,
):
    """force generation of an exception at text generation time to make
    sure the client returns useful information,"""
    stream_exc_prefix = "Exception iterating responses:"
    if using_real_caikit:
        prompt = "dummy"
        detail = "Value out of range: -1"
        match = f"{detail}"
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
        match = f"{detail}"
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


def test_get_text_generation_parameters(grpc_client: GrpcClient):
    params = grpc_client.get_text_generation_parameters()
    expected_params = {
        "text": "string",
        "max_new_tokens": "int64",
        "min_new_tokens": "int64",
        "truncate_input_tokens": "int64",
        "decoding_method": "string",
        "top_k": "int64",
        "top_p": "double",
        "typical_p": "double",
        "temperature": "double",
        "repetition_penalty": "double",
        "max_time": "double",
        "exponential_decay_length_penalty": {
            "start_index": "int64",
            "decay_factor": "double",
        },
        "stop_sequences": "string",
        "seed": "uint64",
        "preserve_input_text": "bool",
        "input_tokens": "bool",
        "generated_tokens": "bool",
        "token_logprobs": "bool",
        "token_ranks": "bool",
    }
    assert params == expected_params


def test_models_info(grpc_client: GrpcClient, using_real_caikit):
    models_info = grpc_client.models_info()
    expected_models_number = 1 if using_real_caikit else 4

    assert len(models_info) == expected_models_number
    required_fields = (
        "loaded",
        "metadata",
        "model_path",
        "module_id",
        "module_metadata",
        "name",
        "size",
    )
    assert all(field in model for field in required_fields for model in models_info)


def test_embedding(
    grpc_client: GrpcClient, embedding_model_name, prompt, using_real_caikit
):
    if using_real_caikit:
        pytest.skip(reason="embeddings endpoint does not work with caikit+tgis")

    with pytest.raises(ValueError, match="request must have a model id"):
        grpc_client.embedding(model_id=None, text=prompt)

    response = grpc_client.embedding(model_id=embedding_model_name, text=prompt)
    assert "result" in response
    assert "data" in response["result"]
    assert "producer_id" in response
    assert response["producer_id"]["name"] == "EmbeddingModule"


def test_embeddings(
    grpc_client: GrpcClient, embedding_model_name, prompt, using_real_caikit
):
    if using_real_caikit:
        pytest.skip(reason="embeddings endpoint does not work with caikit+tgis")

    response = grpc_client.embeddings(
        model_id=embedding_model_name,
        texts=[prompt, prompt + prompt],
    )

    assert response["results"]
    assert "producer_id" in response

    vectors = response["results"]["vectors"]
    assert len(vectors) == 2
    assert all("data" in vec for vec in vectors)
    assert "input_token_count" in response


def test_sentence_similarity(
    grpc_client: GrpcClient, embedding_model_name, prompt, using_real_caikit
):
    if using_real_caikit:
        pytest.skip(reason="embeddings endpoint does not work with caikit+tgis")

    response = grpc_client.sentence_similarity(
        model_id=embedding_model_name,
        source_sentence="source text",
        sentences=["source sent", "source tex"],
    )
    assert "result" in response
    assert "scores" in response["result"]
    assert len(response["result"]["scores"]) == 2


def test_sentence_similarity_tasks(
    grpc_client: GrpcClient, embedding_model_name, prompt, using_real_caikit
):
    if using_real_caikit:
        pytest.skip(reason="embeddings endpoint does not work with caikit+tgis")

    response = grpc_client.sentence_similarity_tasks(
        embedding_model_name,
        ["source text", "text 2"],
        ["source sent", "source tex"],
    )
    assert "results" in response
    assert len(response["results"]) == 2
    assert all("scores" in el for el in response["results"])
    assert len(response["results"][0]["scores"]) == 2
    assert len(response["results"][1]["scores"]) == 2


def test_rerank(
    grpc_client: GrpcClient, embedding_model_name, prompt, using_real_caikit
):
    if using_real_caikit:
        pytest.skip(reason="embeddings endpoint does not work with caikit+tgis")

    documents = [
        {"doc1": 1},
        {"doc2": 2},
    ]

    query = "what's this"

    response = grpc_client.rerank(
        model_id=embedding_model_name,
        documents=documents,
        query=query,
    )

    assert "result" in response
    assert "scores" in response["result"]
    assert "producer_id" in response
    assert response["producer_id"]["name"] == "EmbeddingModule"

    assert len(response["result"]["scores"]) == 2


def test_rerank_tasks(
    grpc_client: GrpcClient, embedding_model_name, prompt, using_real_caikit
):
    if using_real_caikit:
        pytest.skip(reason="embeddings endpoint does not work with caikit+tgis")

    documents = [
        {"doc1": 1},
        {"doc2": 2},
    ]
    queries = [
        "query1",
        "query2",
    ]
    response = grpc_client.rerank_tasks(
        model_id=embedding_model_name,
        documents=documents,
        queries=queries,
    )

    assert "result" in response
    assert "scores" in response["result"]
    assert "producer_id" in response
    assert response["producer_id"]["name"] == "EmbeddingModule"

    assert len(response["result"]["scores"]) == 2


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
