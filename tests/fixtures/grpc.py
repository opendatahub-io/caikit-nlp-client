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
    server_cert: Optional[bytes] = None,
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
            grpc.ssl_channel_credentials(server_cert, client_key, client_cert),
        )


@pytest.fixture(scope="session")
def grpc_server_port():
    """default port for caikit grpc runtime"""
    return get_random_port()


@pytest.fixture(scope="session")
def grpc_client(
    grpc_server_port, connection_type, ca_cert, client_key, client_cert, server_cert
) -> GrpcClient:
    if connection_type is ConnectionType.INSECURE:
        return GrpcClient("localhost", grpc_server_port)

    if connection_type is ConnectionType.TLS:
        return GrpcClient("localhost", grpc_server_port, ca_cert=ca_cert)

    if connection_type is ConnectionType.MTLS:
        return GrpcClient(
            "localhost",
            grpc_server_port,
            server_cert=server_cert,
            client_key=client_key,
            client_cert=client_cert,
        )

    raise ValueError(f"invalid {connection_type=}")


@pytest.fixture(scope="session")
def grpc_server(
    caikit_nlp_runtime,
    grpc_server_port,
    mock_text_generation,
    connection_type,
    ca_cert,
    client_key,
    client_cert,
    server_cert,
):
    from caikit.runtime.grpc_server import RuntimeGRPCServer

    grpc_server = RuntimeGRPCServer()
    grpc_server.start(blocking=False)

    def health_check(host: str, port: int):
        if connection_type is ConnectionType.INSECURE:
            kwargs = {}
        elif connection_type is ConnectionType.TLS:
            kwargs = {
                "ca_cert": ca_cert,
            }
        elif connection_type is ConnectionType.MTLS:
            kwargs = {
                "server_cert": server_cert,
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

    host = "localhost"
    wait_until(
        lambda: health_check(host, grpc_server_port),
        pause=0.5,
    )

    yield grpc_server

    grpc_server.stop()
