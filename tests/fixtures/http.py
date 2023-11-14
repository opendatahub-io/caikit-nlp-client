import caikit
import pytest
import requests
from caikit_nlp_client.http_client import HttpConfig

from .utils import ConnectionType, get_random_port, wait_until


@pytest.fixture(scope="session")
def http_server_port():
    """default port for caikit grpc runtime"""
    return get_random_port()


@pytest.fixture(scope="session")
def http_config(
    caikit_nlp_runtime,
    connection_type,
    client_cert_file,
    client_key_file,
    ca_cert_file,
    server_cert_file,
):
    http_config = HttpConfig(
        host="localhost",
        port=caikit.config.get_config().runtime.http.port,
    )
    if connection_type is ConnectionType.INSECURE:
        return http_config

    elif connection_type is ConnectionType.TLS:
        http_config.tls = True
        return http_config
    else:
        http_config.mtls = True
        http_config.client_crt_path = client_cert_file
        http_config.client_key_path = client_key_file
        http_config.ca_crt_path = ca_cert_file
        return http_config

    raise ValueError(f"invalid {connection_type=}")


@pytest.fixture(scope="session")
def http_server(
    pytestconfig,
    caikit_nlp_runtime,
    http_config,
    mock_text_generation,
    connection_type,
    ca_cert_file,
    client_cert_file,
    client_key_file,
):
    from caikit.runtime.http_server import RuntimeHTTPServer

    http_server = RuntimeHTTPServer()
    http_server.start(blocking=False)

    def health_check(host: str, port: int):
        if connection_type is ConnectionType.INSECURE:
            scheme = "http"
            kwargs = {}
        elif connection_type is ConnectionType.TLS:
            scheme = "https"
            kwargs = {"verify": ca_cert_file}
        elif connection_type is ConnectionType.MTLS:
            scheme = "https"
            kwargs = {
                "verify": ca_cert_file,
                "cert": (client_cert_file, client_key_file),
            }
        else:
            raise ValueError(f"Invalid {connection_type=}")
        response = requests.get(f"{scheme}://{host}:{port}/health", **kwargs)
        assert response.status_code == 200
        assert response.text == "OK"

    wait_until(
        lambda: health_check(http_config.host, http_config.port),
        pause=0.5,
    )

    yield http_server

    http_server.stop()
