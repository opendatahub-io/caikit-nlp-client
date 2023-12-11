import threading
import time

import fastapi
import pytest
import uvicorn
from caikit_nlp_client.utils import get_server_certificate

from tests.fixtures.utils import get_random_port


@pytest.fixture
def app():
    app = fastapi.FastAPI()

    @app.route
    def main():
        return "ok"

    return app


@pytest.fixture
def server(app, server_key_file, server_cert_file):
    port = get_random_port()
    server = uvicorn.Server(
        config=uvicorn.Config(
            app,
            port=port,
            ssl_keyfile=server_key_file,
            ssl_certfile=server_cert_file,
        )
    )

    t = threading.Thread(target=server.run)

    t.start()
    while not server.started:
        time.sleep(1e-3)

    yield "localhost", port

    server.should_exit = True
    t.join()


def test_get_server_certificate(server, server_cert):
    host, port = server
    assert get_server_certificate(host, port).encode() == server_cert
