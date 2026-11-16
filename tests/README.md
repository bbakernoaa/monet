# Testing conventions

Use pytest fixtures from `conftest.py` for shared input that represents a
common use case. `sample_map_data` supplies a fresh, small, deterministic
latitude/longitude field, and `sample_map_frame` supplies paired observations
on its grid. For example:

```python
def test_example(sample_map_data):
    assert sample_map_data.dims == ("lat", "lon")
```

Prefer real, local data and assertions on observable behavior. Keep fixtures
small and independent so a test can modify its input without affecting another
test. Use `pytest.importorskip` or a specific skip condition for an unavailable
optional dependency; do not catch all exceptions around the behavior under
test. Keep network-dependent checks separate from routine collection.

Run the suite with `pytest -v --cov=monet`; run `ruff format --check .` and
`ruff check .` for the same Python checks used in CI. The Ruff pre-commit
revision is the source of the CI version; update it with
`pre-commit autoupdate`.
