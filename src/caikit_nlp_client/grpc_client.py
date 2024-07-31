import logging
import re
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any, Optional, Union

import grpc
from grpc_reflection.v1alpha.proto_reflection_descriptor_database import (
    ProtoReflectionDescriptorDatabase,
)

from google.protobuf.descriptor_pool import DescriptorPool
from google.protobuf.json_format import MessageToDict
from google.protobuf.message_factory import GetMessageClass

from .utils import get_server_certificate

if TYPE_CHECKING:
    from google._upb._message import (
        Descriptor,
        Message,
        MethodDescriptor,
        ServiceDescriptor,
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
        verify: Optional[bool] = None,
        ca_cert: Union[None, bytes, str] = None,
        client_cert: Union[None, bytes, str] = None,
        client_key: Union[None, bytes, str] = None,
    ) -> None:
        """Client class for a Caikit NLP grpc server

        Default connection mode uses a grpc secure channel (TLS)

        >>> client = GrpcClient("localhost", port=8085, plaintext=True)
        >>> generated_text = client.generate_text_stream(
        >>>     "flan-t5-small-caikit",
        >>>     "What is the boiling point of Nitrogen?",
        >>> )

        To connect using TLS:

        >>> client = GrpcClient("localhost", port=443)

        To skip certificate verification:

        >>> client = GrpcClient(remote_host, port=443, insecure=True)

        To provide a custom certificate:

        >>> with open("cert.pem", "rb") as fh:
        >>>     cert = fh.read()
        >>> client = GrpcClient(remote_host, port=443, ca_cert=cert)

        To skip certificate(s) verification:

        >>> client = GrpcClient(remote_host, port=443, verify=False)
        """
        self._channel = self._make_channel(
            host,
            port,
            insecure=insecure,
            verify=verify,
            client_key=client_key,
            client_cert=client_cert,
            ca_cert=ca_cert,
        )

        self._reflection_db = ProtoReflectionDescriptorDatabase(self._channel)
        self._desc_pool = DescriptorPool(self._reflection_db)
        try:
            # call the models info endpoint to make sure that the connection works
            models = self.models_info()
            log.info(f"Available models: {models}")
        except grpc._channel._MultiThreadedRendezvous as exc:
            log.error("Could not connect to the server: %s", exc.details())
            raise RuntimeError(
                f"Could not connect to {host}:{port}:" f"{exc.details()}"
            ) from None
        except KeyError as exc:
            log.error("The grpc server does not have the type: %s", exc)
            raise ValueError(str(exc)) from exc

    def generate_text(
        self,
        model_id: str,
        text: str,
        **kwargs,
    ) -> str:
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

        try:
            result = self._make_request(
                method_name="TextGenerationTaskPredict",
                model_id=model_id,
                text=text,
                **kwargs,
            )
        except grpc._channel._InactiveRpcError as exc:
            raise RuntimeError(exc.details()) from None

        log.info("Calling generate_text was successful")
        return result["generatedText"]

    def get_text_generation_parameters(self) -> dict[str, Any]:
        """returns a dict with available fields and their type"""
        service: ServiceDescriptor = self._desc_pool.FindServiceByName(
            "caikit.runtime.Nlp.NlpService"
        )

        method: MethodDescriptor = service.methods_by_name["TextGenerationTaskPredict"]
        descriptor: Descriptor = method.input_type

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

    def generate_text_stream(
        self,
        model_id: str,
        text: str,
        **kwargs,
    ) -> Iterable[str]:
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

        service_name = "caikit.runtime.Nlp.NlpService"
        service: ServiceDescriptor = self._desc_pool.FindServiceByName(service_name)
        method: MethodDescriptor = service.methods_by_name[
            "ServerStreamingTextGenerationTaskPredict"
        ]

        Request: Message = GetMessageClass(method.input_type)
        Response: Message = GetMessageClass(method.output_type)

        endpoint = f"/{service.full_name}/{method.name}"
        make_request = self._channel.unary_stream(
            endpoint,
            request_serializer=Request.SerializeToString,
            response_deserializer=Response.FromString,
        )
        try:
            request = Request(text=text, **kwargs)
        except ValueError as exc:
            if match := re.match('Protocol message .* has no "(.*)" field.', str(exc)):
                key = match.group(1)
                error = f"Unsupported kwarg: {key}={kwargs.get(key)}"
                raise ValueError(error) from None

            raise

        try:
            yield from (
                message.generated_text
                for message in make_request(
                    request=request,
                    metadata=metadata,
                )
            )
        except grpc._channel._MultiThreadedRendezvous as exc:
            raise RuntimeError(exc.details()) from None

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
        verify: Optional[bool] = None,
        ca_cert: Union[None, bytes, str] = None,
        client_key: Union[None, bytes, str] = None,
        client_cert: Union[None, bytes, str] = None,
    ) -> grpc.Channel:
        """Creates a grpc channel

        Args:
        - host: str
        - port: str
        - (optional) insecure: use a plaintext connection (default=False)
        - (optional) verify: set to False to disable remote host certificate(s)
                     verification. Cannot be used with `plaintext` or with MTLS
        - (optional) ca_cert: certificate authority to use
        - (optional) client_key: client key for mTLS mode
        - (optional) client_cert: client cert for mTLS mode

        """
        if not host.strip():
            raise ValueError("A non empty host name is required")
        if int(port) <= 0:
            raise ValueError("A non zero port number is required")
        if insecure and any(
            (val is not None) for val in (ca_cert, client_key, client_cert)
        ):
            raise ValueError("cannot use insecure with TLS/mTLS certificates")
        if insecure and verify:
            raise ValueError("insecure cannot be used with verify")

        client_key_bytes = self._try_load_certificate(client_key)
        client_cert_bytes = self._try_load_certificate(client_cert)
        ca_cert_bytes = self._try_load_certificate(ca_cert)

        connection = f"{host}:{port}"
        if insecure:
            log.warning("Connecting over an insecure plaintext grpc channel")
            return grpc.insecure_channel(connection)

        credentials_kwargs: dict[str, bytes] = {}
        if ca_cert_bytes and not (any((client_cert_bytes, client_key_bytes))):
            log.info("Connecting using provided CA certificate for secure channel")
            credentials_kwargs.update(root_certificates=ca_cert_bytes)
        elif client_cert_bytes and client_key_bytes and ca_cert_bytes:
            log.info("Connecting using mTLS for secure channel")
            credentials_kwargs.update(
                root_certificates=ca_cert_bytes,
                private_key=client_key_bytes,
                certificate_chain=client_cert_bytes,
            )
        elif verify is False:
            log.warning(
                "insecure mode: trusting remote certificate from %s:%d",
                host,
                port,
            )

            cert = get_server_certificate(host, port).encode()
            credentials_kwargs.update(root_certificates=cert)

        return grpc.secure_channel(
            connection, grpc.ssl_channel_credentials(**credentials_kwargs)
        )

    @staticmethod
    def _try_load_certificate(certificate: Union[None, bytes, str]) -> Optional[bytes]:
        """If the certificate points to a file, return the contents (plaintext reads).
        Else return the bytes"""
        if not certificate:
            return None

        if isinstance(certificate, bytes):
            return certificate

        if isinstance(certificate, str):
            with open(certificate, "rb") as secret_file:
                return secret_file.read()
        raise ValueError(
            f"{certificate=} should be a path to a certificate files or bytes"
        )

    def models_info(self) -> list[dict[str, Any]]:
        response = self._make_request(
            method_name="GetModelsInfo",
            service_name="caikit.runtime.info.InfoService",
        )

        models_dict = response["models"]

        # make sure that the output format matches the http client's
        for model in models_dict:
            model["model_path"] = model.pop("modelPath")
            model["module_id"] = model.pop("moduleId")
            model["module_metadata"] = model.pop("moduleMetadata")

        return models_dict

    def _make_request(
        self,
        method_name: str,
        service_name: str = "caikit.runtime.Nlp.NlpService",
        **kwargs,
    ):
        service: ServiceDescriptor = self._desc_pool.FindServiceByName(service_name)

        method: MethodDescriptor = service.methods_by_name[method_name]
        Request: Message = GetMessageClass(method.input_type)
        Response: Message = GetMessageClass(method.output_type)

        endpoint = f"/{service.full_name}/{method.name}"
        make_request = self._channel.unary_unary(
            endpoint,
            request_serializer=Request.SerializeToString,
            response_deserializer=Response.FromString,
        )

        if "model_id" in kwargs:
            model_id = kwargs.pop("model_id")
            metadata = [("mm-model-id", model_id)]
        else:
            metadata = None

        log.debug(f"making request to {endpoint=}, {metadata=}")
        try:
            request = Request(**kwargs)
        except ValueError as exc:
            if match := re.match('Protocol message .* has no "(.*)" field.', str(exc)):
                key = match.group(1)
                error = f"Unsupported kwarg: {key}={kwargs.get(key)}"
                raise ValueError(error) from None

            raise

        response = make_request(
            request,
            metadata=metadata,
        )
        log.debug(f"Response: {response}")

        return MessageToDict(response)

    def embedding(
        self,
        model_id: str,
        text: str,
        truncate_input_tokens: Optional[bool] = None,
    ) -> dict[str, Any]:
        if not model_id:
            raise ValueError("request must have a model id")

        response = self._make_request(
            method_name="EmbeddingTaskPredict",
            model_id=model_id,
            text=text,
            truncate_input_tokens=truncate_input_tokens,
        )

        response["producer_id"] = response.pop("producerId")
        response["input_token_count"] = response.pop("inputTokenCount")
        response["result"]["data"] = response["result"].pop("dataNpfloat32sequence")
        return response

    def embeddings(
        self,
        model_id: str,
        texts: list[str],
        truncate_input_tokens: Optional[int] = None,
    ) -> dict[str, Any]:
        if not model_id:
            raise ValueError("request must have a model id")

        response = self._make_request(
            method_name="EmbeddingTasksPredict",
            model_id=model_id,
            texts=texts,
            truncate_input_tokens=truncate_input_tokens,
        )
        response["producer_id"] = response.pop("producerId")
        response["input_token_count"] = response.pop("inputTokenCount")

        for vec_dict in response["results"]["vectors"]:
            vec_dict["data"] = vec_dict.pop("dataNpfloat32sequence")

        return response

    def sentence_similarity(
        self,
        model_id: str,
        source_sentence: str,
        sentences: list[str],
        truncate_input_tokens: Optional[int] = None,
    ) -> dict[str, Any]:
        if not model_id:
            raise ValueError("request must have a model id")

        return self._make_request(
            method_name="SentenceSimilarityTaskPredict",
            model_id=model_id,
            source_sentence=source_sentence,
            sentences=sentences,
            truncate_input_tokens=truncate_input_tokens,
        )

    def sentence_similarity_tasks(
        self,
        model_id: str,
        source_sentences: list[str],
        sentences: list[str],
        truncate_input_tokens: Optional[int] = None,
    ) -> dict[str, Any]:
        if not model_id:
            raise ValueError("request must have a model id")

        return self._make_request(
            method_name="SentenceSimilarityTasksPredict",
            model_id=model_id,
            source_sentences=source_sentences,
            sentences=sentences,
            truncate_input_tokens=truncate_input_tokens,
        )

    def rerank(
        self,
        model_id: str,
        documents: list[dict[str, Any]],
        query: str,
        top_n: Optional[int] = None,
        truncate_input_tokens: Optional[int] = None,
        return_documents: Optional[bool] = False,
        return_query: Optional[bool] = False,
        return_text: Optional[bool] = False,
    ) -> dict[str, Any]:
        if not model_id:
            raise ValueError("request must have a model id")

        if not all(isinstance(doc, dict) for doc in documents):
            raise ValueError('documents should be a list of dicts {"text": <text>}')

        Struct: Message = GetMessageClass(
            self._desc_pool.FindMessageTypeByName("google.protobuf.Struct")
        )

        docs = []
        for doc in documents:
            d = Struct()
            d.update(doc)
            docs.append(d)

        response = self._make_request(
            method_name="RerankTaskPredict",
            model_id=model_id,
            documents=docs,
            top_n=top_n,
            truncate_input_tokens=truncate_input_tokens,
            return_documents=return_documents,
            return_query=return_query,
            return_text=return_text,
        )
        response["producer_id"] = response.pop("producerId")
        response["input_token_count"] = response.pop("inputTokenCount")

        return response

    def rerank_tasks(
        self,
        model_id: str,
        documents: list[dict[str, Any]],
        queries: list[str],
        top_n: Optional[int] = None,
        truncate_input_tokens: Optional[int] = None,
        return_documents: Optional[bool] = False,
        return_query: Optional[bool] = False,
        return_text: Optional[bool] = False,
    ) -> dict[str, Any]:
        if not model_id:
            raise ValueError("request must have a model id")

        if not all(isinstance(doc, dict) for doc in documents):
            raise ValueError("documents should be a list of dicts")

        Struct: Message = GetMessageClass(
            self._desc_pool.FindMessageTypeByName("google.protobuf.Struct")
        )

        docs = []
        for doc in documents:
            d = Struct()
            d.update(doc)
            docs.append(d)

        response = self._make_request(
            method_name="RerankTaskPredict",
            model_id=model_id,
            documents=docs,
            top_n=top_n,
            truncate_input_tokens=truncate_input_tokens,
            return_documents=return_documents,
            return_query=return_query,
            return_text=return_text,
        )
        response["producer_id"] = response.pop("producerId")
        response["input_token_count"] = response.pop("inputTokenCount")

        return response
