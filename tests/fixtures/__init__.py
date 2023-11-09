from typing import Optional
from unittest import mock
from caikit_tgis_backend import TGISBackend
from caikit_tgis_backend.tgis_connection import TGISConnection
from caikit.core.module_backends.backend_types import register_backend_type


### Common TGIS stub classes

# Helper stubs / mocks; we use these to patch caikit so that we don't actually
# test the TGIS backend directly, and instead stub the client and inspect the
# args that we pass to it.
class StubTGISClient:

    def __init__(self, base_model_name):
        pass

    def Generate(self, request):
        return StubTGISClient.unary_generate(request)

    def GenerateStream(self, request):
        return StubTGISClient.stream_generate(request)

    @staticmethod
    def unary_generate(request):
        fake_response = mock.Mock()
        fake_result = mock.Mock()
        fake_result.stop_reason = 5
        fake_result.generated_token_count = 1
        fake_result.text = "moose"
        fake_result.input_token_count = 1
        fake_response.responses = [fake_result]
        return fake_response

    @staticmethod
    def stream_generate(request):
        fake_stream = mock.Mock()
        fake_stream.stop_reason = 5
        fake_stream.generated_token_count = 1
        fake_stream.seed = 10
        fake_stream.input_token_count = 1
        token = mock.Mock()
        token.text = "moose"
        token.logprob = 0.2
        fake_stream.tokens = [token]
        fake_stream.text = "moose"
        for _ in range(3):
            yield fake_stream


class StubTGISBackend(TGISBackend):

    def __init__(
            self,
            config: Optional[dict] = None,
            temp_dir: Optional[str] = None,
            mock_remote: bool = False,
    ):
        self._temp_dir = temp_dir
        if mock_remote:
            config = config or {}
            config.update({"connection": {"hostname": "foo.{model_id}:123"}})
        super().__init__(config)
        self.load_prompt_artifacts = mock.MagicMock()
        self.unload_prompt_artifacts = mock.MagicMock()

    def get_client(self, base_model_name):
        self._model_connections[base_model_name] = TGISConnection(
            hostname="foo.bar",
            model_id=base_model_name,
            prompt_dir=self._temp_dir,
        )
        return StubTGISClient(base_model_name)


# Register TGIS stub backend
register_backend_type(StubTGISBackend)
