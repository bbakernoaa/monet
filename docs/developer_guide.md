# Developer Guide

## Aero Protocol Principles

MONET follows a set of architectural principles designed for performance and reliability in scientific computing:

1.  **Architecture**:
    *   Backend-agnostic (NumPy/Dask).
    *   No hidden computes.
    *   No hardcoded chunking.
    *   Use `apply_ufunc(dask='parallelized')` where possible.
2.  **Style**:
    *   NumPy-style docstrings.
    *   Strict type hints.
    *   Update `ds.attrs['history']` to maintain provenance.

## Versioning

MONET follows [Semantic Versioning (SemVer) 2.0.0](https://semver.org/). Versions are synchronized across:
*   `pyproject.toml`
*   `monet/__init__.py`
*   `CITATION.cff`

## Code Quality

Before submitting a Pull Request, please ensure you have run:

```bash
ruff format .
ruff check .
```

## Documentation

Documentation is built using MkDocs with the Material theme. To build the docs locally:

```bash
mkdocs build
```

To serve the docs with live-reloading:

```bash
mkdocs serve
```
