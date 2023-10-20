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

> > > > > > > 0291428 (Initial commit)
