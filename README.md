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
