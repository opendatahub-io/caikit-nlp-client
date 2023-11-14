# caikit-nlp-client

A client library for [`caikit-nlp`](https://github.com/caikit/caikit-nlp)

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
