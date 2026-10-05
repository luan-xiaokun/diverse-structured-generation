# Codebase Structure

**Analysis Date:** 2026-07-23

## Directory Layout

```
aiap/
├── cov_test.py              # Exploratory coverage-obligation and witness demo
└── .planning/
    └── codebase/            # GSD map for the AIAP project (this directory)
```

## Directory Purposes

**`aiap/`:**
- Purpose: bounded workspace for the AIAP submission prototype.
- Contains: one Python experiment script and project-planning artifacts.
- Key file: `aiap/cov_test.py`.
- Parent dependency: source code, case-study inputs, and package configuration remain in the repository root.

**`aiap/.planning/codebase/`:**
- Purpose: codebase orientation documents used before project planning.
- Contains: stack, architecture, conventions, testing, integration, and concern notes.
- Generated/planning documentation, not experiment results.

## Key File Locations

**Entry Point:**
- `aiap/cov_test.py`: runs the current single-obligation witness demonstration.

**Input Configuration:**
- `aiap/cov_test.py`: specifies model name and relative case-study JSON locations.
- `experiments/case_study/data/`: parent-project case-study datasets consumed by the demo.

**Core Logic:**
- `aiap/cov_test.py`: all current data structures, coverage accounting, traversal, and experiment control flow.

**Testing:**
- No AIAP-specific test files exist.
- Parent repository has a separate `tests/` directory, but it does not cover this new demo directory yet.

## Naming Conventions

**Files:**
- Python modules use `snake_case` (`cov_test.py`).
- The existing filename describes a coverage-oriented experiment rather than a stable package API.

**Functions and Types:**
- Functions use `snake_case`.
- Data classes and types use `PascalCase` (`CoverageObligation`).

## Where to Add New Code

**New AIAP algorithm module:**
- Implementation: add a focused `snake_case.py` module under `aiap/`, or create an `aiap/` package once more than a few modules are needed.
- Tests: add `aiap/tests/` or parent-level tests with a clearly named AIAP test module.

**New case-study runner:**
- Implementation: keep input/output orchestration separate from DFA traversal logic.
- Artifacts: use a dedicated, ignored experiment-output directory rather than modifying source datasets.

---
*Structure analysis: 2026-07-23*
