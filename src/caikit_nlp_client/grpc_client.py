import logging
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from google._upb._message import Descriptor, Message

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

    def __init__(
        self,
        host: str,
        port: int,
        *,
        insecure: bool = False,
        ca_cert: Optional[bytes] = None,
        client_key: Optional[bytes] = None,
        client_cert: Optional[bytes] = None,
    ) -> None:
        """Client class for a Caikit NLP grpc server

        >>> # To connect via an insecure port
        >>> client = GrpcClient("localhost", port=8085)
        >>> generated_text = client.generate_text_stream(
        >>>     "flan-t5-small-caikit",
        >>>     "What is the boiling point of Nitrogen?",
        >>> )
        """

        try:
            self._channel = self._make_channel(
                host,
                port,
                insecure=insecure,
                client_key=client_key,
                client_cert=client_cert,
                ca_cert=ca_cert,
            )
        except grpc._channel._MultiThreadedRendezvous as exc:
            log.error("Could not connect to the server: %s", exc.details)
            raise RuntimeError(f"Could not connect to {host}:{port}") from None

        self._reflection_db = ProtoReflectionDescriptorDatabase(self._channel)
        self._desc_pool = DescriptorPool(self._reflection_db)
        try:
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
        except KeyError as exc:
            log.error("The grpc server does not have the type: %s", exc)
            raise ValueError(str(exc)) from exc

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
        try:
            response = self._task_predict(request=request, metadata=metadata)
        except grpc._channel._InactiveRpcError as exc:
            raise RuntimeError(exc.details()) from None

        log.debug(f"Response: {response}")
        result = response.generated_text
        log.info("Calling generate_text was successful")
        return result

    def get_text_generation_parameters(self) -> dict[str, Any]:
        """returns a dict with available fields and their type"""
        descriptor: Descriptor = self._task_text_generation_request.DESCRIPTOR

        # hack: used map the grpc type (int) to a human-readable string
        grpc_type_to_str = {
            getattr(descriptor.fields[0], t): t.split("_")[1].lower()
            for t in dir(descriptor.fields[0])
            if t.startswith("TYPE_")
        }

        def simplify_descriptor(
            descriptor: "Descriptor",
        ) -> dict:
            """recursively flattens grpc descriptor into a human-friendly dict"""

            flattened: dict = {}
            for field in descriptor.fields:
                string_type = grpc_type_to_str[field.type]

                if field.message_type:
                    flattened[field.name] = simplify_descriptor(field.message_type)
                else:
                    flattened[field.name] = string_type

            return flattened

        return simplify_descriptor(descriptor)

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        self._close()
        return False

    def generate_text_stream(self, model_id: str, text: str, **kwargs) -> Iterable[str]:
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

        Returns:
            a list of generated text (tokens)

        Example:

        >>> text = "What is 2+2?"
        >>> chunks = []
        >>> for chunk in grpc_client.generate_text_stream(
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

        metadata = [("mm-model-id", model_id)]

        request = self._task_text_generation_request()
        self._populate_request(request, text, **kwargs)

        try:
            yield from (
                message.generated_text
                for message in self._streaming_task_predict(
                    metadata=metadata, request=request
                )
            )
        except grpc._channel._MultiThreadedRendezvous as exc:
            raise RuntimeError(exc.details()) from None

    def _populate_request(self, request: "Message", text: str, **kwargs):
        """dynamically converts kwargs to request attributes."""
        request.text = text
        for key, value in kwargs.items():
            try:
                setattr(request, key, value)
            except AttributeError as exc:
                raise ValueError(f"Unsupported kwarg {key=}") from exc

    def _close(self):
        try:
            if hasattr(self, "_channel") and self._channel:
                self._channel.close()
        except Exception:
            log.exception("Unexpected exception while closing client")

    def __del__(self):
        self._close()

    def _make_channel(
        self,
        host: str,
        port: int,
        *,
        insecure: bool = False,
        ca_cert: Optional[bytes] = None,
        client_key: Optional[bytes] = None,
        client_cert: Optional[bytes] = None,
    ) -> grpc.Channel:
        if not host.strip():
            raise ValueError("A non empty host name is required")
        if int(port) <= 0:
            raise ValueError("A non zero port number is required")

        connection = f"{host}:{port}"
        if insecure:
            log.warning("Connecting over an insecure grpc channel")
            return grpc.insecure_channel(connection)

        credentials_kwargs: dict[str, bytes] = {}
        if ca_cert and not (any((client_cert, client_key))):
            log.info("Connecting using provided CA certificate for secure channel")
            credentials_kwargs.update(root_certificates=ca_cert)
        elif client_cert and client_key and ca_cert:
            log.info("Connecting using mTLS for secure channel")
            credentials_kwargs.update(
                root_certificates=ca_cert,
                private_key=client_key,
                certificate_chain=client_cert,
            )
        else:
            raise ValueError("mTLS requires client_cert, client_key and ca_cert")

        return grpc.secure_channel(
            connection,
            grpc.ssl_channel_credentials(**credentials_kwargs),
        )
