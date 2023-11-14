import logging

import pytest
import requests

logger = logging.getLogger(__name__)


@pytest.fixture(scope="session")
def docker_compose_file(pytestconfig):
    return str(pytestconfig.rootdir / "tests/fixtures/resources/docker-compose.yml")


def pytest_addoption(parser):
    """Adds --real-caikit option to pytest to test vs a real caikit+tgis instance"""

    parser.addoption(
        "--real-caikit",
        action="store_true",
        default=False,
        help="Test against a real caikit+tgis instance",
    )


@pytest.fixture(scope="session")
def flan_t5_small_caikit(pytestconfig, monkeysession):
    """Points to a downloaded model that can be used with caikit.

    This may be downloaded if not available
    """
    model_dir = pytestconfig.rootdir / "tests/fixtures/resources/flan-t5-small-caikit"

    if not model_dir.isdir():
        model_dir.mkdir()

    if not (model_dir / "config.yml").exists():
        # Download the model so that it can be tested
        monkeysession.setenv("ALLOW_DOWNLOADS", 1)

        import caikit_nlp

        model = caikit_nlp.text_generation.TextGeneration.bootstrap(
            "google/flan-t5-small"
        )
        model.save(str(model_dir))

    return str(model_dir)


@pytest.fixture(scope="session")
def caikit_tgis_service(
    # NOTE: order is important: we need to make sure that the model is available before
    # requesting the `docker_services` fixture, which brings up the stack and mounts
    # volumes.  If model is not available, the stack will not come up.
    flan_t5_small_caikit,
    docker_services,
):
    """Spins up a caikit+tgis instance, returning a dict, grpc_port and http_port"""

    def is_up(url: str) -> False:
        try:
            return requests.get(url, timeout=5).status_code == 200
        except Exception:
            logger.debug(
                "Encountered exception while health checking caikit_tgis_service"
            )
            return False

    caikit_http_port = docker_services.port_for("caikit", 8080)
    caikit_grpc_port = docker_services.port_for("caikit", 8085)

    url_base = f"http://localhost:{caikit_http_port}"
    healthcheck_url = f"{url_base}/health"
    docker_services.wait_until_responsive(
        timeout=30.0, pause=0.1, check=lambda: is_up(healthcheck_url)
    )

    yield {
        "grpc_port": caikit_grpc_port,
        "http_port": caikit_http_port,
    }


@pytest.fixture(scope="session")
def grpc_server_docker(caikit_tgis_service):
    """Returns the grpc port for the caikit container"""
    return "localhost", caikit_tgis_service["grpc_port"]


@pytest.fixture(scope="session")
def http_server_docker(caikit_tgis_service):
    """Returns the http port for the caikit container"""
    return "localhost", caikit_tgis_service["http_port"]
