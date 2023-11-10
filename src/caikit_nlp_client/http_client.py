import logging
from dataclasses import dataclass
from typing import Any, Optional

import requests

log = logging.getLogger(__name__)


@dataclass
class HTTPConfig:
    host: str
    port: int
    tls: bool = False
    mtls: bool = False
    client_key_path: Optional[str] = None
    client_crt_path: Optional[str] = None
    ca_crt_path: Optional[str] = None

    def __post_init__(self):
        if self.mtls and not self.client_key_path:
            raise ValueError("must provide a client_key_path with mTLS")

        if self.mtls and self.tls:
            raise ValueError("mTLS and TLS are mutually exclusive")


class HTTPCaikitNlpClient:
    """HTTP client for a caikit nlp runtime server

    Args:
        http_config (HTTPConfig): Configurations to make HTTP call.
    """

    def __init__(self, http_config: HTTPConfig):
        protocol = "https" if (http_config.mtls or http_config.tls) else "http"
        base_url = f"{protocol}://{http_config.host}:{http_config.port}"
        text_generation_endpoint = "/api/v1/task/text-generation"
        text_generation_stream_endpoint = (
            "/api/v1/task/server-streaming-text-generation"
        )

        self.api_url = f"{base_url}{text_generation_endpoint}"
        self.stream_api_url = f"{base_url}{text_generation_stream_endpoint}"
        self.mtls = http_config.mtls
        self.tls = http_config.tls
        if self.tls or self.mtls:
            if http_config.ca_crt_path:
                self.ca_crt_path = http_config.ca_crt_path
            else:
                raise ValueError(
                    "The CA cert is required for TLS and mTlS configuration"
                )
        if self.mtls:
            if http_config.client_crt_path and http_config.client_key_path:
                self.client_crt_path = http_config.client_crt_path
                self.client_key_path = http_config.client_key_path
            else:
                raise ValueError("Client key and certificates are required for mTLS")

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
        try:
            log.info(f"Calling generate_text for '{model_id}'")
            json_input = create_json_request(model_id, text, **kwargs)

            kwargs = {}
            if self.tls or self.mtls:
                kwargs["verify"] = self.ca_crt_path
            if self.mtls:
                kwargs["cert"] = (
                    self.client_crt_path,
                    self.client_key_path,
                )
            response = requests.post(
                self.api_url, json=json_input, timeout=10.0, **kwargs
            )
            log.debug(f"Response: {response}")
            result: str = response.text
            log.info("Calling generate_text was successful")
            return result
        except Exception as exc:
            log.exception(f"Caught exception {exc}, re-throwing")
            raise exc

    def generate_text_stream(self, model_id: str, text: str, **kwargs) -> list[str]:
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
            a list of generated text (token)
        """
        if model_id == "":
            raise ValueError("request must have a model id")
        try:
            log.info(f"Calling generate_text_stream for '{model_id}'")
            json_input = create_json_request(model_id, text, **kwargs)

            kwargs = {}
            if self.tls or self.mtls:
                kwargs["verify"] = self.ca_crt_path
            if self.mtls:
                kwargs["cert"] = (
                    self.client_crt_path,
                    self.client_key_path,
                )

            response = requests.post(
                self.api_url, json=json_input, timeout=10.0, **kwargs
            )
            log.debug(f"Response: {response}")
            result = [response.text]
            log.info("Calling generate_text_stream was successful")
            return result
        except Exception as exc:
            log.exception(f"Caught exception {exc}, re-throwing")
            raise exc


def create_json_request(model_id, text, **kwargs) -> dict[str, Any]:
    json_input = {
        "model_id": model_id,
        "inputs": text,
        "parameters": {"max_new_tokens": 200, "min_new_tokens": 10},
    }
    if parameters := json_input.get("parameters"):
        parameters.update(kwargs)
    return json_input
