import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from google._upb._message import Message

import grpc
from google.protobuf.descriptor_pool import DescriptorPool
from google.protobuf.message_factory import GetMessageClass
from grpc_reflection.v1alpha.proto_reflection_descriptor_database import (
    ProtoReflectionDescriptorDatabase,
)

log = logging.getLogger(__name__)


class GrpcClient:
    """GRPC client for a caikit nlp runtime server

    Args:
        channel (grpc.Channel): a connected GRPC channel for use of making the calls.
    """

    def __init__(self, host: str, port: int, **kwarg) -> None:
        """Client class for a Caikit NLP grpc server

        >>> # To connect via an insecure port
        >>> client = GrpcClient(host="localhost", port="8085")
        >>> generated_text = client.generate_text_stream(
                "flan-t5-small-caikit",
                "What is the boiling point of Nitrogen?"
            )
        """

        self._channel = self.__make_channel(host, port, **kwarg)
        self._reflection_db = ProtoReflectionDescriptorDatabase(self._channel)
        self._desc_pool = DescriptorPool(self._reflection_db)
        self._text_generation_task_request = GetMessageClass(
            self._desc_pool.FindMessageTypeByName(
                "caikit.runtime.Nlp.TextGenerationTaskRequest"
            )
        )
        self._task_text_generation_request = GetMessageClass(
            self._desc_pool.FindMessageTypeByName(
                "caikit.runtime.Nlp.ServerStreamingTextGenerationTaskRequest"
            )
        )
        self._generated_text_result = GetMessageClass(
            self._desc_pool.FindMessageTypeByName(
                "caikit_data_model.nlp.GeneratedTextResult"
            )
        )
        self._task_predict = self._channel.unary_unary(
            "/caikit.runtime.Nlp.NlpService/TextGenerationTaskPredict",
            request_serializer=self._text_generation_task_request.SerializeToString,
            response_deserializer=self._generated_text_result.FromString,
        )
        self._streaming_task_predict = self._channel.unary_stream(
            "/caikit.runtime.Nlp.NlpService/ServerStreamingTextGenerationTaskPredict",
            request_serializer=self._task_text_generation_request.SerializeToString,
            response_deserializer=self._generated_text_result.FromString,
        )

    def generate_text(self, model_id: str, text: str, **kwargs) -> str:
        """Sends a generate text request to the server for the given model id

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
        metadata = [("mm-model-id", model_id)]

        request = self._text_generation_task_request()
        self._populate_request(request, text, **kwargs)
        response = self._task_predict(request=request, metadata=metadata)
        log.debug(f"Response: {response}")
        result = response.generated_text
        log.info("Calling generate_text was successful")
        return result

    def generate_text_stream(self, model_id: str, text: str, **kwargs) -> list[str]:
        """Sends a generate text stream request to the server for the given model id

        Args:
            model_id: the model identifier
            text: the text to generate

        Keyword Args:
            preserve_input_text (Bool): preserve the input text (default to False)
            max_new_tokens (int): maximum number of new tokens
            min_new_tokens (int): minimum number of new tokens

        Raises:
            ValueError: thrown if an empty model id is passed
            exc: thrown if any exceptions are caught while creating and sending the
                text generation request

        Returns:
            a list of generated text (token)
        """
        if model_id == "":
            raise ValueError("request must have a model id")

        log.info(f"Calling generate_text_stream for '{model_id}'")

        metadata = [("mm-model-id", model_id)]

        request = self._task_text_generation_request()
        self._populate_request(request, text, **kwargs)
        result = []
        for item in self._streaming_task_predict(metadata=metadata, request=request):
            result.append(item.generated_text)
        log.info(
            f"Calling generate_text_stream was successful, '{len(result)}'"
            " items in result"
        )
        return result

    def _populate_request(self, request: "Message", text: str, **kwargs):
        """dynamically converts kwargs to request attributes."""
        request.text = text
        for key, value in kwargs.items():
            try:
                setattr(request, key, value)
            except AttributeError as exc:
                raise ValueError(f"Unsupported kwarg {key=}") from exc

    def __del__(self):
        if hasattr(self, "_channel") and self._channel:
            self._channel.close()

    def __make_channel(self, host: str, port: int, **kwargs) -> grpc.Channel:
        log.debug(f"Making a channel for {host}:{port} with these kwargs={kwargs}")
        if host.strip() == "":
            raise ValueError("A non empty host name is required")
        if port <= 0:
            raise ValueError("A non zero port is required")

        connection = f"{host}:{port}"
        ca_cert = kwargs.get("ca_cert")
        if ca_cert:
            log.debug("A CA certificate has been detected, creating a TLS channel")
            return grpc.secure_channel(
                connection,
                grpc.ssl_channel_credentials(ca_cert),
            )
        client_key = kwargs.get("client_key")
        client_cert = kwargs.get("client_cert")
        server_cert = kwargs.get("server_cert")
        if client_key and client_cert and server_cert:
            log.debug(
                "A client key, client and server certificates have been detected, \
                creating a mTLS channel"
            )
            return grpc.secure_channel(
                connection,
                grpc.ssl_channel_credentials(server_cert, client_key, client_cert),
            )
        log.debug("No certificates detected creating an insecure connection")
        return grpc.insecure_channel(connection)
