import logging
from typing import Any

import requests

from src.caikit_nlp_client.http_config import HTTPConfig

log = logging.getLogger(__name__)


class HTTPCaikitNlpClient:
    """HTTP client for a caikit nlp runtime server

    Args:
        http_config (HTTPConfig): Configurations to make HTTP call.
    """

    def __init__(self, http_config: HTTPConfig):
        protocol = "https" if (http_config.mtls or http_config.tls) else "http"
        base_url = f"{protocol}://{http_config.host}:{http_config.port}"
        text_generation_endpoint = "/api/v1/task/text-generation"
        text_generation_stream_endpoint = "/api/v1/task/server-streaming-text"

        self.api_url = f"{base_url}{text_generation_endpoint}"
        self.stream_api_url = f"{base_url}{text_generation_stream_endpoint}"

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
            response = requests.post(self.api_url, json=json_input, timeout=10)
            log.debug(f"Response: {response}")
            result = response.text
            log.info("Calling generate_text was successful")
            return result
        except Exception as exc:
            log.exception(f"Caught exception {exc}, re-throwing")
            raise exc

    def generate_text_stream(self, model_id: str, text: str, **kwargs) -> [str]:
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
            response = requests.post(self.stream_api_url, json=json_input, timeout=10)
            log.debug(f"Response: {response}")
            result = response.text
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
    json_input.get("parameters").update(kwargs)
    return json_input
