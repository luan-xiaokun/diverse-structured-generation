# Coding Conventions

**Analysis Date:** 2026-05-19

## Naming Patterns

**Files:**
- Use snake_case Python module names for package code and scripts: `src/diverse_guide/guide_rust.py`, `src/diverse_guide/evaluation/string_kernel_py.py`, `scripts/collect_results.py`.
- Use `test_*.py` for root pytest modules under `tests/`: `tests/test_guide.py`, `tests/test_metrics_eval_output.py`, `tests/test_reference_results.py`.
- Use Rust module filenames that match module names: `regex_dfa_guide/src/diverse_guide_dfa.rs`, `regex_dfa_guide/src/error.rs`, `regex_dfa_guide/src/python_bindings/mod.rs`.
- Use package `__init__.py` files only for public re-exports and `__all__`: `src/diverse_guide/__init__.py`, `src/diverse_guide/evaluation/__init__.py`, `regex_dfa_guide/python/regex_dfa_guide/__init__.py`.

**Functions:**
- Use snake_case for Python functions and methods: `build_token_id_map()` in `src/diverse_guide/vocab.py`, `compute_metrics()` in `scripts/metrics_eval.py`, `collect_all()` in `scripts/collect_results.py`.
- Use a leading underscore for private helper functions in tests and scripts: `_make_scores()` in `tests/test_guide.py`, `_write_json()` in `tests/test_collect_results.py`, `_json_paths()` in `scripts/collect_results.py`.
- Use Rust snake_case for methods and helpers: `get_next_token_state()` and `update_path_counter()` in `regex_dfa_guide/src/diverse_guide_dfa.rs`.
- Public Python factory helpers keep established API names: `diverse_regex()` and `baseline_regex()` in `src/diverse_guide/guide_rust.py`.

**Variables:**
- Use descriptive snake_case for local values and attributes: `token_id_to_token`, `allowed_tokens_batch`, `batch_indices_concat` in `src/diverse_guide/guide_rust.py`.
- Use uppercase module constants for schemas, options, and reference expectations: `METRIC_FIELDS` and `RUNTIME_FIELDS` in `scripts/collect_results.py`, `EXPECTED_VERSION` in `tests/test_release_metadata.py`, `GRAMMAR_REGEX` in `scripts/generate_re.py`.
- Use leading underscores for implementation state that tests may inspect directly when behavior depends on it: `_seq_start_idx`, `_guide_states_by_row`, `_processed_lengths_by_row` in `src/diverse_guide/guide_rust.py`.
- Rust type aliases are PascalCase and domain-specific: `StateId`, `TokenId`, and `Byte` in `regex_dfa_guide/src/diverse_guide_dfa.rs`.

**Types:**
- Use PascalCase for Python classes: `DiverseGuide`, `DiverseRegexLogitsProcessor`, `StatefulSequenceGeneratorAdapter` in `src/diverse_guide/guide_rust.py`.
- Use PascalCase for custom exceptions: `MissingResultInputsError` in `scripts/collect_results.py`.
- Use Python 3.12 native type syntax (`list[str]`, `dict[str, Any]`, `int | None`) throughout source files: `src/diverse_guide/evaluation/metrics.py`, `scripts/repro_results.py`, `src/diverse_guide/vocab.py`.
- Use Rust PascalCase for structs/enums and derive traits close to declarations: `DfaIndex`, `DiverseGuideDFA`, and `Error` in `regex_dfa_guide/src/diverse_guide_dfa.rs` and `regex_dfa_guide/src/error.rs`.

## Code Style

**Formatting:**
- Use Ruff formatting for Python. The configured command is `uv run poe format`, defined in `pyproject.toml`.
- Format scope is `src tests examples scripts`; Rust package Python stubs are linted but not included in the format task (`pyproject.toml`).
- Use 88-character Python line length and target Python 3.12, configured under `[tool.ruff]` in `pyproject.toml`.
- Keep generated JSON writer output indented with a trailing newline, following `write_json()` in `scripts/repro_results.py`.
- Use standard Rust formatting (`cargo fmt` style) for `regex_dfa_guide/src/*.rs`; Cargo metadata lives in `regex_dfa_guide/Cargo.toml`.

**Linting:**
- Use `uv run poe lint` for Python lint checks. It runs `ruff check src tests examples scripts regex_dfa_guide/python` from `pyproject.toml`.
- Ruff selects `E`, `F`, `W`, `UP`, and `I`; keep imports sorted and modernize syntax to Python 3.12 patterns.
- Use `uv run poe lint-fix` for safe Ruff fixes when appropriate.
- Rust lint configuration is minimal; `regex_dfa_guide/Cargo.toml` only configures `unexpected_cfgs` for `tarpaulin_include`.

## Import Organization

**Order:**
1. Standard library imports: `argparse`, `json`, `sys`, `pathlib.Path` in `scripts/collect_results.py`.
2. Third-party imports: `numpy`, `pytest`, `torch`, `transformers` in `tests/test_metrics.py`, `tests/test_guide.py`, and `src/diverse_guide/guide_rust.py`.
3. Project imports: `from diverse_guide...`, `from regex_dfa_guide...`, and `from scripts...` after third-party imports.

**Path Aliases:**
- No custom Python import aliases are configured. Pytest uses `pythonpath = ["src", "."]` in `pyproject.toml`.
- Script modules sometimes support both direct script execution and package import with a `try`/`except ModuleNotFoundError` fallback: `scripts/metrics_eval.py` imports `repro_results`, then falls back to `scripts.repro_results`.
- Public package exports must flow through `src/diverse_guide/__init__.py`; do not import implementation internals from top-level package users unless exported there.
- Evaluation public exports must flow through `src/diverse_guide/evaluation/__init__.py`.

## Error Handling

**Patterns:**
- Raise typed, user-readable exceptions for invalid CLI/data states: `MissingResultInputsError` in `scripts/collect_results.py`, `ValueError("n must be positive")` in `scripts/metrics_eval.py`, `RuntimeError` for missing or wrong Outlines version in `scripts/eval_runtime.py`.
- CLI entry points convert expected preflight errors into clean exits without tracebacks: `main()` in `scripts/collect_results.py` catches `MissingResultInputsError`, prints to `stderr`, and raises `SystemExit(1) from None`.
- Library functions should return sentinel values only where explicitly part of the behavior: `calculate_perplexity()` returns `math.inf` for NaN/Inf losses in `src/diverse_guide/evaluation/perplexity.py`.
- Rust core functions return `Result<T, Error>` and use the error enum in `regex_dfa_guide/src/error.rs`; Python bindings convert those errors into Python exceptions in `regex_dfa_guide/src/python_bindings/mod.rs`.
- Hugging Face generation state handling catches invalid-token `ValueError` only to tolerate finished/padded rows: `_advance_row()` in `src/diverse_guide/guide_rust.py` and state tracking in `src/diverse_guide/guide_python.py`.
- Avoid broad `except Exception` in core logic. Existing broad catches are constrained to compatibility/diagnostic paths such as `get_state_sequence_from_token_id()` in `src/diverse_guide/guide_python.py` and optional perplexity printing in `scripts/metrics_eval.py`.

## Logging

**Framework:** console

**Patterns:**
- CLI scripts use `print()` for progress and result locations: `scripts/generate_re.py`, `scripts/metrics_eval.py`, `scripts/eval_runtime.py`, `scripts/collect_results.py`.
- Expected CLI errors print to `stderr` when they should be visible without tracebacks: `scripts/collect_results.py`, `scripts/build_wd_kernel.py`.
- Package/library code should avoid routine logging. `src/diverse_guide/evaluation/perplexity.py` only prints a warning for invalid loss values.
- No structured logging framework is configured.

## Comments

**When to Comment:**
- Use comments to clarify domain-specific invariants, test scenarios, and compatibility behavior: DFA terminal sentinel `-1` in `tests/test_guide.py`, byte-token decoding branches in `src/diverse_guide/vocab.py`, native kernel path selection in `src/diverse_guide/evaluation/string_kernel.py`.
- Use section dividers in larger tests to make behavioral groups scannable: `tests/test_guide.py`, `tests/test_dfa.py`, `tests/test_metrics.py`, `tests/test_string_kernel.py`.
- Keep comments near non-obvious algorithm steps, not as restatements of simple assignments.

**JSDoc/TSDoc:**
- Not applicable. This repository does not use JavaScript or TypeScript.
- Use Python module docstrings and class/function docstrings for public behavior: `src/diverse_guide/guide_rust.py`, `src/diverse_guide/evaluation/metrics.py`, `scripts/collect_results.py`.
- Use Rust doc comments (`//!`, `///`) for crate/module and public type documentation: `regex_dfa_guide/src/lib.rs`, `regex_dfa_guide/src/diverse_guide_dfa.rs`.

## Function Design

**Size:** Keep pure metric/path helpers small and side-effect free, as in `src/diverse_guide/evaluation/metrics.py` and `src/diverse_guide/evaluation/paths.py`. Larger orchestrating functions are acceptable in CLI scripts and generation adapters when they own a workflow boundary, such as `main()` in `scripts/generate_re.py` and `generate_batch()` in `src/diverse_guide/guide_rust.py`.

**Parameters:** Prefer explicit typed parameters over unstructured option dictionaries in library code. Use `argparse.Namespace` only at CLI boundaries and in tests that model CLI inputs: `scripts/metrics_eval.py`, `scripts/eval_runtime.py`, `tests/test_metrics_eval_output.py`.

**Return Values:** Return structured dictionaries only for JSON/CSV result builders, preserving schema fields tested in `tests/test_metrics_eval_output.py`, `tests/test_eval_runtime_output.py`, and `tests/test_collect_results.py`. Library APIs return domain values (`float`, `list[str]`, `torch.FloatTensor`, `Path`) rather than printing or writing.

## Module Design

**Exports:** Public APIs are explicit. Add new top-level user-facing symbols to `src/diverse_guide/__init__.py` and its `__all__`; add evaluation exports to `src/diverse_guide/evaluation/__init__.py`.

**Barrel Files:** Use `__init__.py` barrel files only for stable package boundaries. `src/diverse_guide/guide_python.py` is a reference/debug implementation and is not exported from `src/diverse_guide/__init__.py`.

**Implementation Boundaries:**
- Put Rust-backed generation behavior in `src/diverse_guide/guide_rust.py`.
- Keep pure-Python reference behavior in `src/diverse_guide/guide_python.py` and maintain equivalence tests in `tests/test_guide_equivalence.py`.
- Put deterministic, pure metric functions in `src/diverse_guide/evaluation/metrics.py`; keep file I/O and CLI concerns in `scripts/metrics_eval.py` and `scripts/collect_results.py`.
- Put Rust core DFA logic in `regex_dfa_guide/src/diverse_guide_dfa.rs`, Python bindings in `regex_dfa_guide/src/python_bindings/mod.rs`, and Python package stubs under `regex_dfa_guide/python/regex_dfa_guide/`.

---

*Convention analysis: 2026-05-19*
