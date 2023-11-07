from dataclasses import dataclass

import pytest
from caikit_nlp_client.grpc_client_introspection import GrpcCaikitNlpClientIntrospection


@pytest.fixture
def connected_client(channel):
    """Returns returns a grpc client connected to a locally running server"""
    return GrpcCaikitNlpClientIntrospection(channel)


@dataclass
class GenerateTextTestData:
    model_id: str
    input_text: str
    expected_text: str
    preserve_input_text: bool
    max_new_tokens: int
    min_new_tokens: int


TEST_CASES: list[GenerateTextTestData] = [
    GenerateTextTestData(
        "BertForSequenceClassification-caikit",
        "Hello, my dog is cute",
        "hello, my dog is cuteρ 白ᅩ xʒ ハ ܬ 河 水ʒ セタ x x セ v セ 河 興 セ",
        True,
        20,
        5,
    ),
    GenerateTextTestData(
        "BloomForCausalLM-caikit",
        "Hello stub",
        "Hello stububububububububububububububububububububub",
        True,
        20,
        5,
    ),
    GenerateTextTestData(
        "T5ForConditionalGeneration-caikit",
        "Hello, Mr. Bond!",
        "",
        True,
        20,
        5,
    ),
]

STREAMING_TEST_CASES: list[GenerateTextTestData] = []


@pytest.mark.parametrize("test_case", TEST_CASES)
def test_generate_text(connected_client, test_case):
    generated_text = connected_client.generate_text(
        test_case.model_id,
        test_case.input_text,
    )

    assert generated_text == test_case.expected_text


@pytest.mark.parametrize("test_case", TEST_CASES)
def test_generate_text_with_optional_args(connected_client, test_case):
    generated_text = connected_client.generate_text(
        test_case.model_id,
        test_case.input_text,
        preserve_input_text=test_case.preserve_input_text,
        max_new_tokens=test_case.max_new_tokens,
        min_new_tokens=test_case.min_new_tokens,
    )
    assert generated_text == test_case.expected_text


def test_generate_text_with_no_model_id(connected_client):
    with pytest.raises(ValueError, match="request must have a model id"):
        connected_client.generate_text("", "What does foobar mean?")


# @pytest.mark.parametrize("test_case", STREAMING_TEST_CASES)
@pytest.mark.xfail(reason="None of the models defined support streaming")
def test_generate_text_stream(connected_client, test_case):
    results = connected_client.generate_text_stream(
        test_case.model_id,
        test_case.input_text,
    )
    assert results


# @pytest.mark.parametrize("test_case", STREAMING_TEST_CASES)
@pytest.mark.xfail(reason="None of the models defined support streaming")
def test_generate_text_stream_with_optional_args(connected_client, test_case):
    results = connected_client.generate_text_stream(
        test_case.model_id,
        test_case.input_text,
        preserve_input_text=test_case.preserve_input_text,
        max_new_tokens=test_case.max_new_tokens,
        min_new_tokens=test_case.min_new_tokens,
    )
    assert results
