# Developing

This document describes common development workflows for this repository. For
architecture, see `architecture.md`. For user setup, see `../README.md`.

## Development Environment

Use Python 3.12 and `uv` from the repository root:

```bash
uv sync --group dev
source .venv/bin/activate
```

The root project declares `regex-dfa-guide` as a local path dependency, so
`uv sync` builds and installs the Rust extension automatically.

After changing Rust sources, rebuild the extension into the active root virtual
environment:

```bash
maturin develop --release -m regex_dfa_guide/Cargo.toml
```

Do not run `uv run` from inside `regex_dfa_guide/` for this rebuild. That
subdirectory has its own project metadata, and `uv` may resolve the wrong
environment.

The optional WD-shift native kernel is only needed for faster evaluation:

```bash
uv run python scripts/build_wd_kernel.py
```

## Common Commands

Run from repository root unless noted otherwise.

| Task | Command |
| --- | --- |
| Format Python files | `uv run poe format` |
| Lint Python files | `uv run poe lint` |
| Run Python tests with coverage | `uv run poe test` |
| Run Rust tests | `uv run poe test-rust` |
| Run Rust coverage summary | `uv run poe cov-rust` |
| Build source and wheel distributions | `uv build --sdist --wheel` |
| Build optional WD kernel | `uv run python scripts/build_wd_kernel.py` |

The CI workflow in `.github/workflows/test.yml` runs Python tests, Rust tests,
Ruff linting, and PowerShell experiment-script checks.

## Public API Boundary

The root Python package exports the Rust-backed implementation from
`src/diverse_guide/__init__.py`:

- `diverse_regex`
- `baseline_regex`
- `StatefulSequenceGeneratorAdapter`
- `DiverseRegexLogitsProcessor`
- vocabulary helpers

The pure-Python implementation in `src/diverse_guide/guide_python.py` is kept as
a reference/debug implementation. Do not switch public exports from the
Rust-backed implementation unless the tests and documentation are updated to
explain the change.

## Adding a New Grammar

For ad hoc usage, users can pass any regex directly to `diverse_regex`.

For experiment-script support:

1. Add the regex to `GRAMMAR_REGEX` in `scripts/generate_re.py`.
2. Add the prompt to `GRAMMAR_PROMPT` in `scripts/generate_re.py`.
3. If the grammar needs non-default generation length, update
   `experiments/common.sh` and `experiments/common.ps1`.
4. Add or update tests for argument handling, path construction, and generated
   data serialization if the grammar changes output conventions.
5. Update `experiments/README.md` if the grammar participates in paper-result
   reproduction.

Keep regexes anchored consistently. The guide wraps user regexes as
`(?:regex)$` before DFA construction.

## Adding or Changing Metrics

Metric primitives live in `src/diverse_guide/evaluation/metrics.py`.

When adding a new metric:

1. Implement the pure metric function in `metrics.py`.
2. Add unit tests in `tests/test_metrics.py`.
3. Add CLI/result integration in `scripts/metrics_eval.py`.
4. Add structured CSV extraction in `scripts/collect_results.py` when the metric
   belongs in table outputs.
5. Add tests for the new JSON/CSV fields in `tests/test_metrics_eval_output.py`
   and `tests/test_collect_results.py`.

Keep metric functions deterministic and independent from file-system paths.
Scripts should handle argument parsing, file loading, and result writing.

## Changing the Generation Algorithm

Most algorithm changes belong in `src/diverse_guide/guide_rust.py`.

Use this workflow:

1. Update or add focused tests in `tests/test_guide.py`.
2. If the pure-Python reference should remain equivalent, update
   `src/diverse_guide/guide_python.py`.
3. Update `tests/test_guide_equivalence.py` to cover the intended equivalence
   boundary.
4. Run `uv run poe test`.
5. Run a small model-backed smoke command when the change affects Hugging Face
   generation behavior.

Do not bypass Hugging Face `LogitsProcessor` semantics. The processor mutates
and returns the score tensor expected by `model.generate`.

## Changing the Rust DFA Backend

Core Rust behavior lives in `regex_dfa_guide/src/diverse_guide_dfa.rs`.
Python bindings live in `regex_dfa_guide/src/python_bindings/`.
Type stubs live in
`regex_dfa_guide/python/regex_dfa_guide/diverse_guide_dfa_rs.pyi`.

When changing Rust public methods:

1. Update Rust unit tests in `regex_dfa_guide/src/diverse_guide_dfa.rs`.
2. Update Python bindings in `regex_dfa_guide/src/python_bindings/mod.rs`.
3. Update the `.pyi` stub.
4. Rebuild with `maturin develop --release -m regex_dfa_guide/Cargo.toml`.
5. Run `uv run poe test-rust`.
6. Run `uv run poe test`.

Keep `DfaIndex` immutable after construction. Runtime diversity state should
stay in `DiverseGuideDFA` counters so that `fork()` can share the expensive DFA
index while resetting counters.

## Changing the WD-Shift Native Kernel

The native implementation lives in `native/wd_kernel/wd_kernel.c`. Build logic
lives in `native/wd_kernel/Makefile` and `scripts/build_wd_kernel.py`.

The Python selector in `src/diverse_guide/evaluation/string_kernel.py` must keep
working when the native library is absent. Any native-kernel change should
preserve the pure Python/NumPy fallback behavior.

Run:

```bash
uv run python scripts/build_wd_kernel.py
uv run poe test
```

Use `scripts/benchmark_wd_kernel.py` when comparing C/OpenMP and Python backend
performance.

## Changing Experiment Scripts

Experiment scripts are part of the reproducibility artifact. Keep Linux/macOS
shell scripts and Windows PowerShell scripts behaviorally aligned.

When changing shared experiment behavior:

1. Update `experiments/common.sh`.
2. Update `experiments/common.ps1`.
3. Update `experiments/README.md` if command behavior or outputs change.
4. Run or mirror the PowerShell parse/helper checks from `.github/workflows/test.yml`.
5. Update tests for result path construction and collection when JSON/CSV
   layouts change.

Generated samples belong under `data/`. Structured results belong under
`results/`. These directories are ignored by git and should not be used as
source inputs for tests unless deliberately archived elsewhere.

## Case Study Development

`experiments/case_study/` is an isolated project. Work from that directory:

```bash
cd experiments/case_study
uv sync
make test
```

On Windows, use the PowerShell commands in
`experiments/case_study/README.md`.

Do not edit fetched `email_validator/` or `webcolors/` snapshots by hand. Use
the fetch scripts when refreshing pinned package snapshots.

## Release Checklist

Before a release or OSP artifact snapshot:

1. Run `uv run poe lint`.
2. Run `uv run poe test`.
3. Run `uv run poe test-rust`.
4. Run `uv build --sdist --wheel`.
5. Validate `CITATION.cff` with `cffconvert --validate --infile CITATION.cff`.
6. Confirm `REPRODUCIBILITY.md`, `docs/architecture.md`, and this file describe
   the current command surface.
7. Confirm generated `data/`, `results/`, native build artifacts, virtual
   environments, and caches are not part of the release unless intentionally
   archived as supplementary material.
