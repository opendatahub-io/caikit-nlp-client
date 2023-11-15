import json
import logging
from collections.abc import Iterable
from typing import Any, Optional, Union

import requests

log = logging.getLogger(__name__)


class HttpClient:
    """HTTP client for a caikit nlp runtime server

    Args:
        http_config (HttpConfig): Configurations to make HTTP call.
    """

    def __init__(
        self,
        base_url: str,
        ca_cert_path: Optional[str] = None,
        client_cert_path: Optional[str] = None,
        client_key_path: Optional[str] = None,
    ):
        """Client class for a Caikit NLP HTTP server

        For unsecured connections:

        >>> client = HttpClient("http://localhost:8080")

        For TLS (https) connections:

        >>> client = HttpClient("https://localhost:8080")

        For mTLS connections, both client_cert_path and client_key_path have to be
        provided:

        >>> client = HttpClient("https://localhost:8080",
        >>>     ca_cert_path="path/to/ca.pem",
        >>>     client_cert_path='path/to/client_cert.pem',
        >>>     client_key_path='path/to/client_key.pem'
        >>> )

        """
        text_generation_endpoint = "/api/v1/task/text-generation"
        text_generation_stream_endpoint = (
            "/api/v1/task/server-streaming-text-generation"
        )

        self._api_url = f"{base_url}{text_generation_endpoint}"
        self._stream_api_url = f"{base_url}{text_generation_stream_endpoint}"

        self._client_cert_path = client_cert_path
        self._client_key_path = client_key_path
        self._ca_cert_path = ca_cert_path
        if (
            any(
                (
                    self._client_key_path,
                    self._client_cert_path,
                )
            )
            and not self._mtls_configured
        ):
            raise ValueError(
                "Must provide both client_cert_path and client_key_path for mTLS"
            )

    @property
    def _mtls_configured(self):
        return all((self._client_key_path, self._client_cert_path))

    def _get_tls_configuration(self) -> dict[str, Union[str, tuple[str, str]]]:
        req_kwargs: dict[str, Union[str, tuple[str, str]]] = {}
        if self._mtls_configured:
            assert self._client_key_path
            assert self._client_cert_path

            req_kwargs["cert"] = (
                self._client_cert_path,
                self._client_key_path,
            )
        if self._ca_cert_path:
            req_kwargs["verify"] = self._ca_cert_path

        return req_kwargs

    def generate_text(
        self, model_id: str, text: str, timeout: float = 60.0, **kwargs
    ) -> str:
        """Queries the `text-generation` endpoint for the given model_id

        Args:
            model_id: the model identifier
            text: the text to generate

        Keyword Args:
            preserve_input_text (Bool): preserve the input text (default to False)
            max_new_tokens (int): maximum number of new tokens
            min_new_tokens (int): minimum number of new tokens
            timeout (int): HTTP request timeout value in seconds

        Raises:
            ValueError: thrown if an empty model id is passed
            exc: thrown if any exceptions are caught while creating and sending
            the text generation request

        Returns:
            the generated text

        Example:

        >>> generated_text = client.generate_text(
                "flan-t5-small-caikit",
                "What is the boiling point of Nitrogen?"
            )
        """
        if not model_id:
            raise ValueError("request must have a model id")

        log.info(f"Calling generate_text for '{model_id}'")
        json_input = self._create_json_request(
            model_id,
            text,
            **kwargs,
        )

        req_kwargs = self._get_tls_configuration()

        response = requests.post(
            self._api_url,
            json=json_input,
            timeout=timeout,
            **req_kwargs,  # type: ignore
        )

        response.raise_for_status()  # TODO

        return response.json()["generated_text"]

    def generate_text_stream(
        self, model_id: str, text: str, timeout: float = 60.0, **kwargs
    ) -> Iterable[str]:
        """Queries the `text-generation` stream endpoint for the given model_id

        Args:
            model_id: the model identifier
            text: the text to generate

        Keyword Args:
            preserve_input_text (Bool): preserve the input text (default to False)
            max_new_tokens (int): maximum number of new tokens
            min_new_tokens (int): minimum number of new tokens
            timeout (int): HTTP request timeout value in seconds

        Raises:
            ValueError: thrown if an empty model id is passed
            exc: thrown if any exceptions are caught while creating and sending
            the text generation request

        Returns:
            a list of messages generated from the server

        Example:

        >>> text = "What is 2+2?"
        >>> chunks = []
        >>> for chunk in http_client.generate_text_stream(
        >>>     "flan-t5-small-caikit",
        >>>     text,
        >>> ):
        >>>     print(f"Got {chunk=}")
        >>>     chunks.append(chunk)
        >>> print(f"final result: {''.join(chunks)}")

        """
        if not model_id:
            raise ValueError("request must have a model id")

        log.info(f"Calling generate_text_stream for '{model_id}'")

        payload: dict[str, Any] = {
            "model_id": model_id,
            "inputs": text,
        }
        if kwargs:
            payload["parameters"] = kwargs

        req_kwargs = self._get_tls_configuration()

        response = requests.post(
            self._stream_api_url,
            json=payload,
            timeout=timeout,
            **req_kwargs,  # type: ignore
        )

        buffer: list[bytes] = []
        for line in response.iter_lines():
            if line:
                # the first 6 bytes contain "data: ", we can skip those
                buffer.append(line[6:])
                continue

            try:
                message = json.loads(b"".join(buffer))
            except json.JSONDecodeError:
                # message not over yet
                continue

            buffer.clear()
            yield message["generated_text"]

        if buffer:
            final_message = json.loads(b"".join(buffer))
            yield final_message["generated_text"]

    def _create_json_request(self, model_id, text, **kwargs) -> dict[str, Any]:
        json_input = {
            "model_id": model_id,
            "inputs": text,
            "parameters": {"max_new_tokens": 200, "min_new_tokens": 10},
        }
        if parameters := json_input.get("parameters"):
            parameters.update(kwargs)
        return json_input
