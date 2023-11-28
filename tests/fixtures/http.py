import pytest
import requests
from caikit_nlp_client.http_client import HttpClient

from .utils import ConnectionType, get_random_port, wait_until


@pytest.fixture(scope="session")
def http_server_thread_port():
    """port for caikit http runtime thread"""
    return get_random_port()


@pytest.fixture(scope="session")
def using_real_caikit(pytestconfig):
    return pytestconfig.option.real_caikit


@pytest.fixture(scope="session")
def http_client(
    caikit_nlp_runtime,
    http_server,
    using_real_caikit,
    request: pytest.FixtureRequest,
    connection_type,
    client_cert_file,
    client_key_file,
    ca_cert_file,
) -> HttpClient:
    if using_real_caikit:
        if connection_type is not ConnectionType.INSECURE:
            pytest.skip(reason="not testing TLS with a docker caikit instance")

        host, port = request.getfixturevalue("http_server_docker")
    else:
        host, port = request.getfixturevalue("http_server_thread")

    kwargs: dict = {}

    if connection_type is ConnectionType.INSECURE:
        url = f"http://{host}:{port}"
    elif connection_type is ConnectionType.TLS:
        # a valid certificate autority should validate the response with no extra args
        url = f"https://{host}:{port}"

    elif ConnectionType.MTLS:
        url = f"https://{host}:{port}"
        kwargs.update(
            ca_cert_path=ca_cert_file,
            client_cert_path=client_cert_file,
            client_key_path=client_key_file,
        )
    else:
        raise ValueError(f"invalid {connection_type=}")

    return HttpClient(url, **kwargs)


@pytest.fixture(scope="function")
def accept_self_signed_certs(ca_cert_file, monkeysession):
    """validates self signed certs by injecting REQUEST_CA_BUNDLE"""
    with monkeysession.context() as monkeypatch:
        monkeypatch.setenv("REQUESTS_CA_BUNDLE", ca_cert_file)

        yield


@pytest.fixture(scope="session")
def http_server_thread(
    caikit_nlp_runtime,
    http_server_thread_port,
    mock_text_generation,
):
    """spins a caikit http server in a thread for testing, returning host and port"""
    from caikit.runtime.http_server import RuntimeHTTPServer

    http_server = RuntimeHTTPServer()
    http_server.start(blocking=False)

    yield "localhost", http_server_thread_port

    http_server.stop()


@pytest.fixture(scope="session")
def http_server(
    pytestconfig,
    request: pytest.FixtureRequest,
    caikit_nlp_runtime,
    mock_text_generation,
    connection_type,
    ca_cert_file,
    client_cert_file,
    client_key_file,
):
    if pytestconfig.option.real_caikit:
        if connection_type is not ConnectionType.INSECURE:
            pytest.skip(reason="not testing TLS with a docker caikit instance")

        host, port = request.getfixturevalue("http_server_docker")
    else:
        host, port = request.getfixturevalue("http_server_thread")

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
        lambda: health_check(host, port),
        pause=0.5,
    )

    yield host, port
