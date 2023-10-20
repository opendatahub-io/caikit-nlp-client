import logging

import grpc
from google.protobuf.descriptor_pool import DescriptorPool
from google.protobuf.message_factory import GetMessageClass
from grpc_reflection.v1alpha.proto_reflection_descriptor_database import (
    ProtoReflectionDescriptorDatabase,
)

log = logging.getLogger(__name__)


class GrpcCaikitNlpClientIntrospection:
    """GRPC client for a caikit nlp runtime server

    Args:
        channel (grpc.Channel): a connected GRPC channel for use of making the calls.
    """

    def __init__(self, channel: grpc.Channel) -> None:
        try:
            self.reflection_db = ProtoReflectionDescriptorDatabase(channel)
            self.desc_pool = DescriptorPool(self.reflection_db)
            self.text_generation_task_request = GetMessageClass(
                self.desc_pool.FindMessageTypeByName(
                    "caikit.runtime.Nlp.TextGenerationTaskRequest"
                )
            )
            self.server_streaming_text_generation_task_request = GetMessageClass(
                self.desc_pool.FindMessageTypeByName(
                    "caikit.runtime.Nlp.ServerStreamingTextGenerationTaskRequest"
                )
            )
            self.generated_text_result = GetMessageClass(
                self.desc_pool.FindMessageTypeByName(
                    "caikit_data_model.nlp.GeneratedTextResult"
                )
            )
            self.text_generation_task_predict = channel.unary_unary(
                "/caikit.runtime.Nlp.NlpService/TextGenerationTaskPredict",
                request_serializer=self.text_generation_task_request.SerializeToString,
                response_deserializer=self.generated_text_result.FromString,
            )
            self.server_streaming_text_generation_task_predict = channel.unary_stream(
                "/caikit.runtime.Nlp.NlpService/ServerStreamingTextGenerationTaskPredict",
                request_serializer=self.server_streaming_text_generation_task_request.SerializeToString,
                response_deserializer=self.generated_text_result.FromString,
            )
        except Exception as exc:
            log.error(f"Caught exception {exc}, re-throwing")
            raise exc

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
        try:
            log.info(f"Calling generate_text for '{model_id}'")
            metadata = [("mm-model-id", model_id)]

            request = self.text_generation_task_request()
            self.__populate_request(request, text, **kwargs)
            response = self.text_generation_task_predict(
                request=request, metadata=metadata
            )
            log.debug(f"Response: {response}")
            result = response.generated_text
            log.info("Calling generate_text was successful")
            return result
        except Exception as exc:
            log.error(f"Caught exception {exc}, re-throwing")
            raise exc

    def generate_text_stream(self, model_id: str, text: str, **kwargs) -> [str]:
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
        try:
            log.info(f"Calling generate_text_stream for '{model_id}'")

            metadata = [("mm-model-id", model_id)]

            request = self.server_streaming_text_generation_task_request()
            self.__populate_request(request, text, **kwargs)
            result = []
            for item in self.server_streaming_text_generation_task_predict(
                metadata=metadata, request=request
            ):
                result.append(item.generated_text)
            log.info(
                f"Calling generate_text_stream was successful, '{len(result)}'"
                " items in result"
            )
            return result
        except Exception as exc:
            log.error(f"Caught exception {exc}, re-throwing")
            raise exc

    def __populate_request(self, request, text, **kwargs):
        request.text = text
        if "preserve_input_text" in kwargs:
            request.preserve_input_text = kwargs.get("preserve_input_text")
        if "max_new_tokens" in kwargs:
            request.max_new_tokens = kwargs.get("max_new_tokens")
        if "min_new_tokens" in kwargs:
            request.min_new_tokens = kwargs.get("min_new_tokens")
