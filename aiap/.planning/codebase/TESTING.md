# Testing Patterns

**Analysis Date:** 2026-07-23

## Test Framework

**Runner:**
- No AIAP-specific runner or test configuration is present.
- The parent repository contains Python tests under `tests/`, but this demo currently has no automated coverage.

**Assertion Library:**
- No direct evidence within `aiap/`.
- New tests should use the parent repository's established Python test tooling after checking `pyproject.toml` and existing tests during implementation.

## Test File Organization

**Current location:**
- `aiap/` has only `cov_test.py`; it is a script, not a test despite its name.

**Recommended layout:**
```
aiap/
├── coverage_obligations.py
├── witness_search.py
├── suite_selection.py
└── tests/
    ├── test_coverage_obligations.py
    ├── test_witness_search.py
    └── test_suite_selection.py
```

## Essential Test Cases for This Project

**Obligation accounting:**
- Verify the initial DFA state is counted.
- Verify state, byte-transition, and state-pair path obligations are all reported below a threshold.
- Include an obligation occurring in the middle of a multi-byte token.

**Witness correctness:**
- Every returned witness must be accepted by the DFA.
- The DFA traversal trace must meet each targeted obligation.
- An unreachable or non-completable obligation must return `None` without an infinite loop.

**Future multi-obligation search:**
- Test that one generated witness covers the expected set of obligations, not merely the requested seed obligation.
- Compare objective values and coverage against the current one-obligation baseline on fixed tiny DFAs.

## Fixtures and Reproducibility

- Prefer tiny hand-built/serialized DFA fixtures for deterministic unit tests.
- Keep case-study JSON samples immutable and record random seeds, model identifiers, decoding parameters, and tokenizer revision for experiments.
- Mock or cache tokenizer loading in unit tests to avoid a network dependency.

## Coverage

- No target or enforcement exists.
- Prioritize semantic invariants of graph traversal and selection quality before measuring line coverage.

---
*Testing analysis: 2026-07-23*
