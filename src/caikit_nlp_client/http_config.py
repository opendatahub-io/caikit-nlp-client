from dataclasses import dataclass
from typing import Optional


@dataclass
class HTTPConfig:
    host: str
    port: int
    tls: bool = False
    mtls: bool = False
    client_key: Optional[str] = None
    client_crt: Optional[str] = None
    server_crt: Optional[str] = None
