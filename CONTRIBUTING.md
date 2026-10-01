# Contributing to ps-gnn

Thanks for your interest in contributing! This project is part of an
active Earth Observation research effort, so we especially welcome
contributions that improve scientific rigor (better baselines, more
realistic benchmarks, additional validation) alongside general software
engineering improvements.

## Development setup

```bash
git clone https://github.com/EOCoreINT/ps-gnn.git
cd ps-gnn
pip install -e ".[dev,geo-extra]"
pre-commit install
```

## Running tests

```bash
# Fast unit/model tests only (recommended during development)
pytest tests/ -m "not slow"

# Full suite, including integration tests
pytest tests/

# With coverage
pytest tests/ --cov=ps_gnn --cov-report=term-missing
```

New functionality should come with tests. Tests that exercise the full
training/inference/reporting pipeline, or that are otherwise
significantly heavier than a focused unit test, should be marked
`@pytest.mark.slow`.

## Code style

This project uses [`ruff`](https://docs.astral.sh/ruff/) for linting and
[`black`](https://black.readthedocs.io/) for formatting, both run in CI
and via `pre-commit`:

```bash
ruff check ps_gnn tests
black ps_gnn tests
```

Please also follow these project conventions:

- **Type hints everywhere.** All public functions should have complete
  type annotations.
- **NumPy-style docstrings.** Every public function/class needs a
  docstring with `Parameters`, `Returns`, and (where applicable) `Raises`
  sections — see any existing module for the expected format.
- **Cite scientific assumptions.** When a function encodes a modeling
  choice (a threshold, a weighting scheme, a simplification of a
  published method), document *why* in the docstring or a code comment,
  not just *what*.
- **Graceful degradation for optional integrations.** Code that touches
  optional dependencies (`shap`, `libpysal`, `onnxruntime`, ...) should
  raise a clear `ImportError` with an install hint if the dependency is
  missing, rather than a bare `ModuleNotFoundError` traceback.

## Submitting changes

1. Fork the repository and create a feature branch.
2. Make your changes, with tests.
3. Ensure `ruff check`, `black --check`, and `pytest tests/ -m "not slow"`
   all pass locally.
4. Open a pull request describing the change and, for anything touching
   model behavior or scientific methodology, the reasoning behind it.

## Reporting issues

Please include a minimal reproducible example, the output of
`pip show ps-gnn torch torch-geometric`, and (if relevant) the benchmark
site / data characteristics involved.
