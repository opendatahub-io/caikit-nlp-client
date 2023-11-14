from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def ca_cert_file():
    return str(Path(__file__).parent / "resources/ca.pem")


@pytest.fixture(scope="session")
def ca_cert(ca_cert_file):
    with open(ca_cert_file, "rb") as fh:
        return fh.read()


@pytest.fixture(scope="session")
def client_key_file():
    return str(Path(__file__).parent / "resources/client-key.pem")


@pytest.fixture(scope="session")
def client_key(client_key_file):
    with open(client_key_file, "rb") as fh:
        return fh.read()


@pytest.fixture(scope="session")
def client_cert_file():
    return str(Path(__file__).parent / "resources/client.pem")


@pytest.fixture(scope="session")
def client_cert(client_cert_file):
    with open(client_cert_file, "rb") as fh:
        return fh.read()


@pytest.fixture(scope="session")
def server_key_file():
    return str(Path(__file__).parent / "resources/server-key.pem")


@pytest.fixture(scope="session")
def server_key(server_key_file):
    with open(server_key_file, "rb") as fh:
        return fh.read()


@pytest.fixture(scope="session")
def server_cert_file():
    return str(Path(__file__).parent / "resources/server.pem")


@pytest.fixture(scope="session")
def server_cert(server_cert_file):
    with open(server_cert_file, "rb") as fh:
        return fh.read()
