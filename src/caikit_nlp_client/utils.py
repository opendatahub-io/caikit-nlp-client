import socket
import ssl
import sys


def get_server_certificate(host: str, port: int) -> str:
    """connect to host:port and get the certificate it presents

    This is almost the same as `ssl.get_server_certificate`, but
    when opening the TLS socket, `server_hostname` is also provided.

    This retrieves the correct certificate for hosts using name-based
    virtual hosting.
    """
    if sys.version_info >= (3, 10):
        # ssl.get_server_certificate supports TLS SNI only above 3.10
        # https://github.com/python/cpython/pull/16820
        return ssl.get_server_certificate((host, port))

    context = ssl.SSLContext()

    with socket.create_connection((host, port)) as sock, context.wrap_socket(
        sock, server_hostname=host
    ) as ssock:
        cert_der = ssock.getpeercert(binary_form=True)

    assert cert_der
    return ssl.DER_cert_to_PEM_cert(cert_der)
