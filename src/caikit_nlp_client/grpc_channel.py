import logging
from dataclasses import dataclass

import grpc

log = logging.getLogger(__name__)


@dataclass
class GrpcChannelConfig:
    host: str
    port: int
    insecure: bool = False
    root_certificates = None
    private_key = None
    certificate_chain = None


def make_channel(config: GrpcChannelConfig) -> grpc.Channel:
    log.debug(f"Making a channel from this config {config}")
    if config.host.strip() == "":
        raise ValueError("A non empty host name is required")
    if config.port <= 0:
        raise ValueError("A non zero port is required")
    connection = f"{config.host}:{config.port}"
    if config.insecure:
        return grpc.insecure_channel(connection)
    return grpc.secure_channel(
        connection,
        grpc.ssl_channel_credentials(
            config.root_certificates, config.private_key, config.certificate_chain
        ),
    )
