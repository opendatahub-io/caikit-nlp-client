from typing import Optional

import grpc
import pytest
from caikit_nlp_client.grpc_client import GrpcClient
from grpc_health.v1 import health_pb2, health_pb2_grpc

from .utils import ConnectionType, get_random_port, wait_until


def channel_factory(
    host: str,
    port: int,
    connection_type: ConnectionType,
    ca_cert: Optional[bytes] = None,
    client_key: Optional[bytes] = None,
    client_cert: Optional[bytes] = None,
) -> grpc.Channel:
    connection = f"{host}:{port}"
    if connection_type is ConnectionType.INSECURE:
        return grpc.insecure_channel(connection)
    if connection_type is ConnectionType.TLS:
        return grpc.secure_channel(
            connection,
            grpc.ssl_channel_credentials(ca_cert),
        )
    if connection_type is ConnectionType.MTLS:
        return grpc.secure_channel(
            connection,
            grpc.ssl_channel_credentials(ca_cert, client_key, client_cert),
        )


@pytest.fixture(scope="session")
def grpc_server_thread_port():
    """port for caikit grpc runtime thread"""
    return get_random_port()


@pytest.fixture(scope="session")
def grpc_client(
    grpc_server,
    connection_type,
    ca_cert_file,
    client_key_file,
    client_cert_file,
) -> GrpcClient:
    if connection_type is ConnectionType.INSECURE:
        return GrpcClient(*grpc_server, insecure=True)

    if connection_type is ConnectionType.TLS:
        return GrpcClient(*grpc_server, ca_cert=ca_cert_file)

    if connection_type is ConnectionType.MTLS:
        return GrpcClient(
            *grpc_server,
            ca_cert=ca_cert_file,
            client_key=client_key_file,
            client_cert=client_cert_file,
        )

    raise ValueError(f"invalid {connection_type=}")


@pytest.fixture(scope="session")
def grpc_server_thread(
    caikit_nlp_runtime,
    grpc_server_thread_port,
    mock_text_generation,
):
    """spins a caikit grpc server in a thread for testing, returning host and port"""
    from caikit.runtime.grpc_server import RuntimeGRPCServer

    grpc_server = RuntimeGRPCServer()
    grpc_server.start(blocking=False)

    yield "localhost", grpc_server_thread_port

    grpc_server.stop()


@pytest.fixture(scope="session")
def grpc_server(
    pytestconfig,
    request: pytest.FixtureRequest,
    connection_type,
    ca_cert,
    client_key,
    client_cert,
):
    if pytestconfig.option.real_caikit:
        if connection_type is not ConnectionType.INSECURE:
            pytest.skip(reason="not testing TLS with a docker caikit instance")
        host, port = request.getfixturevalue("grpc_server_docker")
    else:
        host, port = request.getfixturevalue("grpc_server_thread")

    def health_check(host: str, port: int):
        if connection_type is ConnectionType.INSECURE:
            kwargs = {}
        elif connection_type is ConnectionType.TLS:
            kwargs = {
                "ca_cert": ca_cert,
            }
        elif connection_type is ConnectionType.MTLS:
            kwargs = {
                "client_cert": client_cert,
                "client_key": client_key,
                "ca_cert": ca_cert,
            }
        else:
            raise ValueError(f"Unknown {connection_type=}")

        channel = channel_factory(host, port, connection_type, **kwargs)
        stub = health_pb2_grpc.HealthStub(channel)
        health_check_request = health_pb2.HealthCheckRequest()
        stub.Check(health_check_request)

    wait_until(
        lambda: health_check(host, port),
        pause=0.5,
    )

    yield host, port
