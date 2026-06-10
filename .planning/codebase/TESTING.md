# Testing Patterns

**Analysis Date:** 2026-05-19

## Test Framework

**Runner:**
- pytest 9.0+ for Python tests.
- Config: `pyproject.toml` under `[tool.pytest]`.
- Rust unit tests run with Cargo through `regex_dfa_guide/Cargo.toml`.
- Case-study tests are an isolated project under `experiments/case_study/`.

**Assertion Library:**
- Python uses built-in `assert`, `pytest.approx`, `pytest.raises`, and NumPy assertions (`np.testing.assert_allclose`) in files such as `tests/test_metrics.py`, `tests/test_string_kernel.py`, and `tests/test_eval_runtime_output.py`.
- Rust uses `assert!`, `assert_eq!`, `matches!`, `expect()`, and `expect_err()` in the `#[cfg(test)]` module in `regex_dfa_guide/src/diverse_guide_dfa.rs`.

**Run Commands:**
```bash
uv run poe test        # Run Python tests with branch coverage for diverse_guide
uv run poe test-rust   # Run Rust tests for regex_dfa_guide
uv run poe cov-rust    # Run Rust coverage summary, requires cargo-llvm-cov
uv run poe lint        # Run Ruff checks used by CI
```

## Test File Organization

**Location:**
- Root Python tests are centralized in `tests/` and configured as the only pytest testpath in `pyproject.toml`.
- Rust tests are co-located inside `regex_dfa_guide/src/diverse_guide_dfa.rs` in a `#[cfg(test)] mod tests` block.
- Case-study regression tests live separately in `experiments/case_study/tests/` and are run from the case-study project, not from the root pytest `testpaths`.
- Deprecated tests under `archive/deprecated/` are not part of the configured test suite.

**Naming:**
- Use `tests/test_<area>.py` for root tests: `tests/test_guide.py`, `tests/test_dfa.py`, `tests/test_string_kernel.py`, `tests/test_collect_results.py`.
- Use `test_<behavior>` for pytest function names, with behavior-oriented names such as `test_collect_all_reports_all_missing_or_empty_inputs_without_writing_tables()` in `tests/test_collect_results.py`.
- Use `Test<ClassifiedArea>` classes where grouping many related property tests improves readability: `TestNumpyKernelCorrectness` and `TestDispatchLayer` in `tests/test_string_kernel.py`.
- Use Rust test names that describe behavior: `new_rejects_invalid_regex()` and `counters_affect_compute_counts_and_can_be_reset()` in `regex_dfa_guide/src/diverse_guide_dfa.rs`.

**Structure:**
```text
tests/
├── conftest.py                    # shared lightweight tokenizer/DFA fixtures
├── test_guide.py                  # public generator/logits processor behavior
├── test_guide_equivalence.py      # pure-Python vs Rust-backed equivalence
├── test_dfa.py                    # Python binding behavior for Rust DFA
├── test_metrics.py                # pure metric functions
├── test_string_kernel.py          # NumPy/C dispatch and numerical properties
├── test_*_output.py               # CLI/result schema boundaries
└── test_*metadata*.py             # packaging/release/reference invariants

regex_dfa_guide/src/diverse_guide_dfa.rs
└── #[cfg(test)] mod tests          # Rust core DFA unit tests

experiments/case_study/tests/
├── conftest.py                    # isolated import path setup
├── case_study_helpers.py          # data loading helpers
├── test_webcolors.py              # package regression checks
└── test_email_validator.py        # package regression checks
```

## Test Structure

**Suite Organization:**
```python
# tests/test_metrics.py
def test_transition_coverage_partial(dfa_single):
    cov = transition_coverage(dfa_single, ["a"])
    assert cov == pytest.approx(0.5)


# tests/test_collect_results.py
def test_main_prints_clean_preflight_error_without_traceback(
    tmp_path, monkeypatch, capsys
):
    monkeypatch.setattr(sys, "argv", ["collect_results.py", "--results-dir", str(tmp_path)])
    with pytest.raises(SystemExit) as exc_info:
        main()
    assert exc_info.value.code == 1
```

**Patterns:**
- Use small, deterministic fixtures from `tests/conftest.py` for tokenizer and DFA setup (`mock_tokenizer`, `vocab_ab`, `dfa_multi`, `dfa_single`).
- Use helper builders for structured result payloads rather than duplicating large dictionaries: `_metrics_result()` and `_runtime_result()` in `tests/test_collect_results.py`, `_args()` in `tests/test_metrics_eval_output.py` and `tests/test_eval_runtime_output.py`.
- Test CLI functions by patching `sys.argv` and asserting clean stdout/stderr behavior with `capsys`: `tests/test_collect_results.py`, `tests/test_metrics_eval_output.py`, `tests/test_eval_runtime_output.py`.
- Test filesystem writes with `tmp_path`; never write generated test artifacts into `data/`, `results/`, or package directories.
- Test tensor mutation behavior directly for logits processors, using tiny synthetic tensors in `tests/test_guide.py`.
- Test numerical code with `pytest.approx` or `np.testing.assert_allclose`, not exact floating-point equality: `tests/test_metrics.py`, `tests/test_string_kernel.py`, `tests/test_perplexity.py`.

## Mocking

**Framework:** pytest `monkeypatch`, simple local fake classes, and `unittest.mock.Mock`.

**Patterns:**
```python
# tests/test_eval_runtime_output.py
monkeypatch.setattr(eval_runtime.torch.cuda, "is_available", lambda: False)
monkeypatch.setattr(
    eval_runtime.AutoModelForCausalLM,
    "from_pretrained",
    lambda *args, **kwargs: _FakeModel(),
)

# tests/test_generate_re_seed.py
random_seed = Mock()
monkeypatch.setattr("scripts.generate_re.random.seed", random_seed)
set_generation_seed(42)
random_seed.assert_called_once_with(42)
```

**What to Mock:**
- Mock model/tokenizer objects for Hugging Face behavior so unit tests do not download or execute real LLMs: `_FakeModel`, `_FakeTokenizer`, `_BatchTokenizerMock`, and `_DeterministicModel` in `tests/test_eval_runtime_output.py` and `tests/test_guide.py`.
- Mock optional external package availability/version checks: `package_version` and `sys.modules["outlines"]` in `tests/test_eval_runtime_output.py`.
- Mock random seed sinks to prove deterministic seeding calls: `tests/test_generate_re_seed.py`.
- Mock CUDA availability in tests that exercise runtime setup: `tests/test_eval_runtime_output.py`.

**What NOT to Mock:**
- Do not mock the Rust DFA extension for root Python behavior tests; `tests/test_dfa.py`, `tests/test_guide.py`, and `tests/test_metrics.py` intentionally exercise the installed `regex_dfa_guide.DiverseGuideDFA`.
- Do not mock pure metric functions when testing CLI result schemas; use real `compute_metrics()` outputs in `tests/test_metrics_eval_output.py`.
- Do not mock the Python string-kernel fallback in `tests/test_string_kernel.py`; tests compare dispatch behavior to the real NumPy implementation and skip C-specific tests only when the C backend is unavailable.

## Fixtures and Factories

**Test Data:**
```python
# tests/conftest.py
@pytest.fixture
def vocab_ab():
    return {10: "a", 20: "b"}

@pytest.fixture
def dfa_single(vocab_ab):
    return DiverseGuideDFA("(?:[ab])$", 99, vocab_ab)
```

**Location:**
- Root shared fixtures live in `tests/conftest.py`.
- Test-specific factories stay inside the test module that uses them: `_metrics_result()` in `tests/test_collect_results.py`, `_args()` in `tests/test_eval_runtime_output.py`, `_identity_kernel()` in `tests/test_metrics.py`.
- Case-study helpers live in `experiments/case_study/tests/case_study_helpers.py`, with expected counts in `experiments/case_study/tests/expected_metrics.py`.
- Reference artifacts used by tests live under `artifact-results/v0.2.0/`, validated by `tests/test_reference_results.py`.

## Coverage

**Requirements:** Python branch coverage is always collected for `diverse_guide` by pytest configuration in `pyproject.toml`, but no minimum coverage threshold is enforced. Rust coverage is available as an explicit optional command and is not part of the default Python test task.

**View Coverage:**
```bash
uv run poe test        # Shows pytest-cov term-missing branch coverage for diverse_guide
uv run poe cov-rust    # Shows Rust coverage summary for regex_dfa_guide
```

## Test Types

**Unit Tests:**
- Core Python unit tests cover generation adapters/logits processors in `tests/test_guide.py`, vocabulary decoding in `tests/test_vocab.py`, metrics in `tests/test_metrics.py`, paths in `tests/test_paths.py`, perplexity in `tests/test_perplexity.py`, and result helpers in `tests/test_repro_results.py`.
- Rust unit tests cover DFA construction, transitions, counters, serialization, and display in `regex_dfa_guide/src/diverse_guide_dfa.rs`.

**Integration Tests:**
- Python binding integration is covered by `tests/test_dfa.py`, which imports `regex_dfa_guide.DiverseGuideDFA` and exercises construction, transitions, counters, forking, equality, and pickle round trips.
- Implementation equivalence is covered by `tests/test_guide_equivalence.py`, comparing `src/diverse_guide/guide_python.py` and `src/diverse_guide/guide_rust.py`.
- CLI/result schema integration is covered by `tests/test_metrics_eval_output.py`, `tests/test_eval_runtime_output.py`, and `tests/test_collect_results.py`.
- Packaging and release boundaries are covered by `tests/test_packaging_imports.py`, `tests/test_release_metadata.py`, and `tests/test_reference_results.py`.
- CI also parses PowerShell experiment scripts and checks path helper behavior in `.github/workflows/test.yml`.

**E2E Tests:**
- No full model-download E2E suite is configured in root pytest.
- Model-backed smoke checks are documented as manual workflow guidance in `docs/developing.md`.
- Case-study regression tests in `experiments/case_study/tests/` exercise real pinned package snapshots and sample datasets within the isolated case-study project.

## Common Patterns

**Async Testing:**
```python
# Not used. The codebase has no async application surface in src/, scripts/, or tests/.
```

**Error Testing:**
```python
# tests/test_metrics_eval_output.py
with pytest.raises(ValueError, match="n must be positive"):
    compute_metrics(dfa_single, ["a", "b"], n=0)

# regex_dfa_guide/src/diverse_guide_dfa.rs
let err = DiverseGuideDFA::new("[", 0, Map::default()).expect_err("must fail");
assert!(matches!(err, Error::IndexDfaError(_)));
```

**Skip/Conditional Testing:**
```python
# tests/test_string_kernel.py
c_only = pytest.mark.skipif(not _is_c_backend(), reason="C extension not built")
```

**Parameterized Regression Testing:**
```python
# experiments/case_study/tests/test_email_validator.py
@pytest.mark.parametrize(
    ("allow_domain_literal", "allow_quoted_local", "allow_empty_local"),
    EMAIL_VALIDATION_OPTIONS,
    ids=EMAIL_VALIDATION_OPTION_IDS,
)
def test_email_validation_regression(...):
    ...
```

---

*Testing analysis: 2026-05-19*
