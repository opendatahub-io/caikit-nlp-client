import logging
from dataclasses import dataclass
from typing import Optional

import grpc

log = logging.getLogger(__name__)


@dataclass
class GrpcChannelConfig:
    host: str
    port: int
    tls: bool = False
    mtls: bool = False
    ca_cert = Optional[bytes]
    client_key = Optional[bytes]
    client_cert = Optional[bytes]
    server_cert = Optional[bytes]


def make_channel(config: GrpcChannelConfig) -> grpc.Channel:
    log.debug(f"Making a channel from this config {config}")
    if config.host.strip() == "":
        raise ValueError("A non empty host name is required")
    if config.port <= 0:
        raise ValueError("A non zero port is required")

    connection = f"{config.host}:{config.port}"

    if not config.tls and not config.mtls:
        return grpc.insecure_channel(connection)

    if config.tls:
        if config.client_key is None:
            raise ValueError("A client key is required")
        if config.client_cert is None:
            raise ValueError("A client cert is required")
        if config.ca_cert is None:
            raise ValueError("A certificate authority certificate is required")
        return grpc.secure_channel(
            connection,
            grpc.ssl_channel_credentials(
                config.ca_cert, config.client_key, config.client_cert
            ),
        )
    if config.mtls:
        if config.client_key is None:
            raise ValueError("A client key is required")
        if config.client_cert is None:
            raise ValueError("A client certificate is required")
        if config.server_cert is None:
            raise ValueError("A server certificate is required")
        return grpc.secure_channel(
            connection,
            grpc.ssl_channel_credentials(
                config.server_cert, config.client_key, config.client_cert
            ),
        )
