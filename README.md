# caikit-nlp-client

[![PyPI version](https://badge.fury.io/py/caikit-nlp-client.svg)](https://badge.fury.io/py/caikit-nlp-client)
[![Tests](https://github.com/opendatahub-io/caikit-nlp-client/actions/workflows/tests.yml/badge.svg)](https://github.com/opendatahub-io/caikit-nlp-client/actions/workflows/tests.yml)
[![codecov](https://codecov.io/gh/opendatahub-io/caikit-nlp-client/graph/badge.svg?token=6LYJLZDO52)](https://codecov.io/gh/opendatahub-io/caikit-nlp-client)

A client library for [caikit-nlp](https://github.com/caikit/caikit-nlp)

## Installation

Install from [PyPi](https://pypi.org/project/caikit-nlp-client/)

```bash
pip install caikit-nlp-client
```

## Usage

A few examples follow, see [`example.py`](/examples/example.py)

### http

To use the http protocol

```python
from caikit_nlp_client import HttpClient

host = "localhost"
port = 8080
model_name = "flan-t5-small-caikit"
http_client = HttpClient(f"http://{host}:{port}")

text = http_client.generate_text(model_name, "What is the boiling point of Nitrogen?")
```

### gRPC

To use the gRPC protocol

```python
from caikit_nlp_client import GrpcClient

host, port= "localhost", 8085
model_name = "flan-t5-small-caikit"
grpc_client = GrpcClient(host, port, insecure=True) # plain text mode

text = grpc_client.generate_text(model_name, "What is the boiling point of Nitrogen?")
```

Text generation methods may accept text generation parameters, which can be provided as kwargs
to `generate_text` and `generate_text_stream`.

Available values and types be retrieved as a dict for both the grpc and http clients:

```python
for param, default_value in client.get_text_generation_parameters():
    print(f"{param=}, {default_value=}")
```

### Self-signed certificates

To use a self-signed certificate, assuming we have a certificate authority cert `ca.pem`

```python
http_client = HttpClient(f"https://{host}:{http_port}", ca_cert_path="ca.pem")

with open("ca.pem", "rb") as fh:
    ca_cert = fh.read()
grpc_client = GrpcClient(host, grpc_port, ca_cert=ca_cert)
```

To skip certificate validation:

```python
# http
http_client = HttpClient(f"https://{host}:{http_port}", verify=False)
# grpc
grpc_client = GrpcClient(host, port, verify=False)
```

### mTLS

Assuming we have a `client.pem` and `client-key.pem` certificate files, and we require `ca.pem` to validate the server certificate:

```python
# http
http_client = HttpClient(
    f"https://{host}:{http_port}",
    ca_cert_path="ca.pem",
    client_cert_path="client.pem",
    client_key_path="client-key.pem"
)

# grpc
with open("ca.pem", "rb") as fh:
    ca_cert = fh.read()
with open("client.pem", "rb") as fh:
    client_cert = fh.read()
with open("client-key.pem", "rb") as fh:
    client_key = fh.read()

grpc_client = GrpcClient(
    host,
    port,
    ca_cert=ca_cert,
    client_key=client_key,
    client_cert=client_cert,
)
# alternatively you can pass the paths directly to the client constructor
grpc_client = GrpcClient(
    host,
    port,
    ca_cert="ca.pem",
    client_cert="client.pem",
    client_key="client-key.pem"
)
```

## Contributing

Set up [`pre-commit`](https://pre-commit.com) for linting/style/misc fixes:

```bash
pip install pre-commit
pre-commit install
```

This project uses [`nox`](https://github.com/wntrblm/nox) to manage test automation:

```bash
pip install nox
nox --list  # list available sessions
nox --python 3.10 -s tests # run tests session for a specific python version
```

### Testing against a real caikit+tgis stack

Tests are run against a mocked instance of a TGIS backend by default. To test against a real
caikit+tgis stack, it is sufficient to run using the `--real-caikit` flag when running `pytest`:

```bash
nox -s tests -- --real-caikit tests
```

_Notes:_

- The required images (`caikit-tgis-serving`, `text-generation-inference`), so it could take a while for tests to start while
  compose is pulling the required images, it may seem like the tests are hanging.
- This uses a real model ([google/flan-t5-small](https://huggingface.co/google/flan-t5-small)), which will be downloaded
  to `tests/fixtures/resources/flan-t5-small-caikit` and is around 300MB in size.
