import json
import logging
from collections.abc import Iterable
from typing import Any

import requests

log = logging.getLogger(__name__)


class HttpClient:
    """HTTP client for a caikit nlp runtime server

    Args:
        http_config (HttpConfig): Configurations to make HTTP call.
    """

    def __init__(self, base_url: str, **kwargs):
        """Client class for a Caikit NLP HTTP server
        >>> # For unsecured connections use (the port is optional)
        >>> client = HttpClient("http://localhost:8080")
        >>> # For TLS (https) connections use (the port is optional):
        >>> client = HttpClient("https://localhost:8080")
        >>> # For mTLS connections use (the port is optional):
        >>> client = HttpClient("https://localhost:8080",
            ca_cert_path='path to ca pem file', \
                client_cert_path='path to client pem file', \
                    client_key_path='path to client private key file')
        >>> generated_text = client.generate_text_stream(
                "flan-t5-small-caikit",
                "What is the boiling point of Nitrogen?"
            )
        """
        text_generation_endpoint = "/api/v1/task/text-generation"
        text_generation_stream_endpoint = (
            "/api/v1/task/server-streaming-text-generation"
        )

        self._api_url = f"{base_url}{text_generation_endpoint}"
        self._stream_api_url = f"{base_url}{text_generation_stream_endpoint}"
        self._mtls = False
        if (
            "client_crt_path" in kwargs
            and "client_key_path" in kwargs
            and "ca_crt_path" in kwargs
        ):
            self._mtls = True
            self._client_crt_path = kwargs.get("client_crt_path")
            self._client_key_path = kwargs.get("client_key_path")
            self._ca_crt_path = kwargs.get("ca_crt_path")

    def generate_text(self, model_id: str, text: str, **kwargs) -> str:
        """Queries the `text-generation` endpoint for the given model_id

        Args:
            model_id: the model identifier
            text: the text to generate

        Keyword Args:
            preserve_input_text (Bool): preserve the input text (default to False)
            max_new_tokens (int): maximum number of new tokens
            min_new_tokens (int): minimum number of new tokens

        Raises:
            ValueError: thrown if an empty model id is passed
            exc: thrown if any exceptions are caught while creating and sending
            the text generation request

        Returns:
            the generated text
        """
        if model_id == "":
            raise ValueError("request must have a model id")
        log.info(f"Calling generate_text for '{model_id}'")
        json_input = self._create_json_request(model_id, text, **kwargs)

        kwargs = {}
        if self._mtls:
            kwargs["verify"] = self._ca_crt_path
            kwargs["cert"] = (
                self._client_crt_path,
                self._client_key_path,
            )
        response = requests.post(self._api_url, json=json_input, timeout=10.0, **kwargs)
        log.debug(f"Response: {response}")
        result: str = response.text
        log.info("Calling generate_text was successful")
        return result

    def generate_text_stream(
        self, model_id: str, text: str, **kwargs
    ) -> Iterable[dict[str, Any]]:
        """Queries the `text-generation` stream endpoint for the given model_id

        Args:
            model_id: the model identifier
            text: the text to generate

        Keyword Args:
            preserve_input_text (Bool): preserve the input text (default to False)
            max_new_tokens (int): maximum number of new tokens
            min_new_tokens (int): minimum number of new tokens

        Raises:
            ValueError: thrown if an empty model id is passed
            exc: thrown if any exceptions are caught while creating and sending
            the text generation request

        Returns:
            a list of messages generated from the server

        Example:

        >>> text = "What is 2+2?"
        >>> chunks = []
        >>> for message in http_client.generate_text_stream(
        >>>     "flan-t5-small-caikit",
        >>>     text,
        >>> ):
        >>>     chunk = message.generated_text
        >>>     if message["details"]["finish_reason"] == "NOT_FINISHED":
        >>>         print("Got chunk")
        >>>         chunks.append(chunk)
        >>>     print(f"final result: {''.join(chunks)}")
        >>>     print(f"finish_reason: {message['details']['finish_reason']}")
        """
        if model_id == "":
            raise ValueError("request must have a model id")
        log.info(f"Calling generate_text_stream for '{model_id}'")
        json_input = self._create_json_request(model_id, text, **kwargs)

        kwargs = {}
        if self._mtls:
            kwargs["verify"] = self._ca_crt_path
            kwargs["cert"] = (
                self._client_crt_path,
                self._client_key_path,
            )

        response = requests.post(
            self._stream_api_url, json=json_input, timeout=10.0, **kwargs
        )

        buffer: list[bytes] = []
        for line in response.iter_lines():
            if line:
                # each line will be in the format <message type>: <data>
                # we don't care about the message type
                buffer.append(line.split(b":", maxsplit=1)[1])
                continue

            try:
                message = json.loads(b"".join(buffer))
            except json.JSONDecodeError:
                # message not over yet
                continue

            buffer.clear()
            yield message

        if buffer:
            final_message = json.loads(b"".join(buffer))
            yield final_message

    def _create_json_request(self, model_id, text, **kwargs) -> dict[str, Any]:
        json_input = {
            "model_id": model_id,
            "inputs": text,
            "parameters": {"max_new_tokens": 200, "min_new_tokens": 10},
        }
        if parameters := json_input.get("parameters"):
            parameters.update(kwargs)
        return json_input
