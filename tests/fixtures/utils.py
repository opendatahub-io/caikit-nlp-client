import socket
import time
from contextlib import closing
from enum import Enum
from typing import Callable, TypeVar

WAIT_TIME_OUT = 10


_T = TypeVar("_T")


def get_random_port():
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(("", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        port = s.getsockname()[1]
        return port


class ConnectionType(Enum):
    INSECURE = 1  # grpc insecure (plaintext)
    TLS = 2
    MTLS = 3


def wait_until(
    pred: Callable[..., _T], timeout: float = WAIT_TIME_OUT, pause: float = 0.1
) -> _T:
    start = time.perf_counter()
    exc = None
    while (time.perf_counter() - start) < timeout:
        try:
            value = pred()
        except Exception as e:  # pylint: disable=broad-except
            exc = e
        else:
            return value
        time.sleep(pause)

    raise TimeoutError("timed out waiting") from exc
