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
        verify: Optional[bool] = None,
        ca_cert_path: Optional[str] = None,
        client_cert_path: Optional[str] = None,
        client_key_path: Optional[str] = None,
    ):
        """Client class for a Caikit NLP HTTP server

        For unsecured connections:

        >>> client = HttpClient("http://localhost:8080")

        For TLS (https) connections:

        >>> client = HttpClient("https://localhost:8080")

        to skip verification of TLS certs:

        >>> client = HttpClient("https://localhost:8080", verify=False)

        to use a custom CA:

        >>> client = HttpClient("https://localhost:8080", ca_cert_path=path)

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

        self._api_base = base_url
        self._api_url = f"{base_url}{text_generation_endpoint}"
        self._stream_api_url = f"{base_url}{text_generation_stream_endpoint}"
        self._models_info_url = f"{base_url}/info/models"

        if verify is False and ca_cert_path:
            raise ValueError("Cannot use verify=False with ca_cert_path")

        self._verify = verify

        self._client_cert_path = client_cert_path
        self._client_key_path = client_key_path
        self._ca_cert_path = ca_cert_path
        if (
            any((self._client_key_path, self._client_cert_path))
            and not self._mtls_configured
        ):
            raise ValueError(
                "Must provide both client_cert_path and client_key_path for mTLS"
            )

    @property
    def _mtls_configured(self):
        return all((self._client_key_path, self._client_cert_path, self._ca_cert_path))

    def _get_tls_configuration(self) -> dict[str, Union[str, tuple[str, str]]]:
        req_kwargs: dict = {}
        if self._mtls_configured:
            assert self._client_key_path
            assert self._client_cert_path

            req_kwargs["cert"] = (
                self._client_cert_path,
                self._client_key_path,
            )

        if self._ca_cert_path:
            req_kwargs["verify"] = self._ca_cert_path
        elif self._verify is not None:
            req_kwargs["verify"] = self._verify

        return req_kwargs

    def get_text_generation_parameters(
        self,
        timeout: float = 60.0,
    ) -> dict:
        """returns a dict with available fields and their type"""
        req_kwargs = self._get_tls_configuration()

        openapi_spec = requests.get(
            f"{self._api_base}/openapi.json",
            timeout=timeout,
            **req_kwargs,  # type: ignore
        ).json()

        request_schema = openapi_spec["paths"]["/api/v1/task/text-generation"]["post"][
            "requestBody"
        ]["content"]["application/json"]["schema"]
        parameters = request_schema["properties"]["parameters"]

        def simplify_parameter_schema(parameters: dict) -> dict:
            """recursively flattens openapi's spec into a human-friendly dict"""

            if "$ref" in parameters:
                value = parameters["$ref"]
                prefix, name = value.rsplit("/", maxsplit=1)
                assert prefix == "#/components/schemas"
                params = openapi_spec["components"]["schemas"][name]["properties"]
            else:
                params = parameters["allOf"][0]["properties"]

            flattened = {}
            for param, description in params.items():
                if "$ref" in description:
                    flattened[param] = simplify_parameter_schema(description)
                else:
                    flattened[param] = description["type"]

            return flattened

        return simplify_parameter_schema(parameters)

    def generate_text(
        self,
        model_id: str,
        text: str,
        timeout: float = 60.0,
        **kwargs,
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
        log.debug(f"Response: {response}")
        if response.status_code == 200:
            return response.json()["generated_text"]

        try:
            details = response.json()["details"]
        except json.JSONDecodeError:
            details = f"{response.reason} {response.text=}"

        raise RuntimeError(f"{response.status_code=} {details}")

    def generate_text_stream(
        self,
        model_id: str,
        text: str,
        timeout: float = 60.0,
        **kwargs,
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
            stream=True,
            **req_kwargs,  # type: ignore
        )
        log.debug(f"Response: {response}")
        buffer: list[bytes] = []
        for line in response.iter_lines():
            if line:
                if line.startswith(b"data: "):
                    buffer.append(line[6:])
                    continue
                if line.startswith(b"event: message"):
                    assert not line[14:]
                    continue
                if line.startswith(b"event: error"):
                    assert not line[12:]
                    continue

            try:
                message = json.loads(b"".join(buffer))
            except json.JSONDecodeError:
                # message not over yet
                continue

            buffer.clear()
            if "details" in message and "code" in message:
                raise RuntimeError(
                    "Exception iterating responses: {}".format(message["details"])
                )
            yield message["generated_text"]
        if buffer:
            final_message = json.loads(b"".join(buffer))
            if "details" in final_message and "code" in final_message:
                raise RuntimeError(
                    "Exception iterating responses: {}".format(final_message["details"])
                )
            try:
                yield final_message["generated_text"]
            except KeyError as exc:
                raise RuntimeError(
                    "Unexpected response from the server: generated text is missing"
                ) from exc

    def _create_json_request(self, model_id, text, **kwargs) -> dict[str, Any]:
        payload = {
            "model_id": model_id,
            "inputs": text,
        }
        if kwargs:
            payload.update(parameters=kwargs)
        return payload

    def models_info(
        self,
        timeout: float = 60.0,
    ) -> dict[str, list[dict[str, Any]]]:
        req_kwargs = self._get_tls_configuration()

        response = requests.get(
            self._models_info_url,
            timeout=timeout,
            **req_kwargs,  # type: ignore
        )
        response.raise_for_status()

        return response.json()["models"]

    def embedding(
        self,
        model_id: str,
        text: str,
        timeout: float = 60.0,
        parameters: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        if not model_id:
            raise ValueError("request must have a model id")

        log.info(f"Calling embedding for '{model_id}'")
        json_input: dict[str, Any] = {
            "inputs": text,
            "model_id": model_id,
        }
        if parameters:
            json_input.update(parameters=parameters)

        req_kwargs = self._get_tls_configuration()

        response = requests.post(
            f"{self._api_base}/api/v1/task/embedding",
            json=json_input,
            timeout=timeout,
            **req_kwargs,  # type: ignore
        )
        log.debug(f"Response: {response}")
        return self._unpack_or_raise_details(response)

    def embedding_tasks(
        self,
        model_id: str,
        texts: list[str],
        timeout: float = 60.0,
        parameters: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        if not model_id:
            raise ValueError("request must have a model id")

        log.info(f"Calling embedding_tasks for '{model_id}'")
        json_input: dict[str, Any] = {
            "inputs": texts,
            "model_id": model_id,
        }
        if parameters:
            json_input.update(parameters=parameters)

        req_kwargs = self._get_tls_configuration()

        response = requests.post(
            f"{self._api_base}/api/v1/task/embedding-tasks",
            json=json_input,
            timeout=timeout,
            **req_kwargs,  # type: ignore
        )
        log.debug(f"Response: {response}")
        return self._unpack_or_raise_details(response)

    def sentence_similarity(
        self,
        model_id: str,
        source_sentence: str,
        sentences: list[str],
        timeout: float = 60.0,
        parameters: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        if not model_id:
            raise ValueError("request must have a model id")

        log.info(f"Calling sentence_similarity for '{model_id}'")
        json_input = {
            "inputs": {
                "source_sentence": source_sentence,
                "sentences": sentences,
            },
            "model_id": model_id,
        }
        if parameters:
            json_input.update(parameters=parameters)

        req_kwargs = self._get_tls_configuration()

        response = requests.post(
            f"{self._api_base}/api/v1/task/sentence-similarity",
            json=json_input,
            timeout=timeout,
            **req_kwargs,  # type: ignore
        )
        log.debug(f"Response: {response}")
        return self._unpack_or_raise_details(response)

    def sentence_similarity_tasks(
        self,
        model_id: str,
        source_sentences: list[str],
        sentences: list[str],
        timeout: float = 60.0,
        parameters: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        if not model_id:
            raise ValueError("request must have a model id")

        log.info(f"Calling sentence_similarity_tasks for '{model_id}'")
        json_input = {
            "inputs": {
                "source_sentences": source_sentences,
                "sentences": sentences,
            },
            "model_id": model_id,
        }
        if parameters:
            json_input.update(parameters=parameters)

        req_kwargs = self._get_tls_configuration()

        response = requests.post(
            f"{self._api_base}/api/v1/task/sentence-similarity-tasks",
            json=json_input,
            timeout=timeout,
            **req_kwargs,  # type: ignore
        )
        log.debug(f"Response: {response}")
        return self._unpack_or_raise_details(response)

    def rerank(
        self,
        model_id: str,
        documents: list[dict[str, Any]],
        query: str,
        timeout: float = 60.0,
        parameters: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        if not model_id:
            raise ValueError("request must have a model id")

        log.info(f"Calling rerank for '{model_id}'")
        json_input = {
            "inputs": {
                "documents": documents,
                "query": query,
            },
            "model_id": model_id,
        }
        if parameters:
            json_input.update(parameters=parameters)

        req_kwargs = self._get_tls_configuration()

        response = requests.post(
            f"{self._api_base}/api/v1/task/rerank",
            json=json_input,
            timeout=timeout,
            **req_kwargs,  # type: ignore
        )
        log.debug(f"Response: {response}")
        return self._unpack_or_raise_details(response)

    def rerank_tasks(
        self,
        model_id: str,
        documents: list[dict[str, Any]],
        queries: list[str],
        timeout: float = 60.0,
        parameters: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        if not model_id:
            raise ValueError("request must have a model id")

        log.info(f"Calling rerank_tasks for '{model_id}'")
        json_input = {
            "inputs": {
                "documents": documents,
                "queries": queries,
            },
            "model_id": model_id,
        }
        if parameters:
            json_input.update(parameters=parameters)

        req_kwargs = self._get_tls_configuration()

        response = requests.post(
            f"{self._api_base}/api/v1/task/rerank-tasks",
            json=json_input,
            timeout=timeout,
            **req_kwargs,  # type: ignore
        )
        log.debug(f"Response: {response}")
        return self._unpack_or_raise_details(response)

    def _unpack_or_raise_details(
        self, response: requests.Response, details_field=None
    ) -> dict[str, Any]:
        """
        The details_field is present because there seems to be an inconsistent
        naming of this field between different endpoints.
        Some have "details" while others have "detail".

        See the following issue: https://github.com/caikit/caikit/issues/750

        When details_field == None, try to detect the right name of the details field.
        """
        if response.status_code == 200:
            return response.json()

        try:
            if details_field is None:
                details_field = "details" if "details" in response.json() else "detail"
            details = response.json()[details_field]
        except json.JSONDecodeError:
            details = f"{response.reason} {response.text=}"

        raise RuntimeError(f"{response.status_code=} {details}")
