---
phase: 01-baseline-replay-foundation
plan: 01
subsystem: testing
tags: [python, dfa, bfs, packaging, coverage]
requires: []
provides:
  - isolated nested AIAP package importable from its project directory
  - deterministic state/transition/path obligation catalog
  - reusable single-obligation DFA BFS baseline
affects: [02-canonical-byte-replay, 03-cli-and-parity]
tech-stack:
  added: [setuptools local package metadata]
  patterns: [typed obligation model, deterministic catalog ordering, token BFS with byte-level matching]
key-files:
  created: [pyproject.toml, aiap/__init__.py, aiap/__main__.py, aiap/obligations.py, aiap/baseline.py, tests/conftest.py, tests/test_packaging.py, tests/test_baseline.py]
  modified: []
key-decisions:
  - "Keep AIAP packaging metadata local to the nested project and do not alter parent package discovery."
  - "Derive stable catalog entries exclusively from parent DFA states and byte transitions."
requirements-completed: [BASE-01, BASE-02]
coverage:
  - id: D1
    description: "Nested AIAP package imports independently from aiap/."
    requirement: BASE-01
    verification:
      - kind: unit
        ref: "tests/test_packaging.py::test_nested_package_imports_from_project_directory"
        status: pass
    human_judgment: false
  - id: D2
    description: "Stable JSON-safe obligation catalog and single-obligation BFS baseline."
    requirement: BASE-02
    verification:
      - kind: unit
        ref: "tests/test_packaging.py::test_catalog_is_deterministic_and_json_safe"
        status: pass
      - kind: unit
        ref: "tests/test_baseline.py"
        status: pass
    human_judgment: false
duration: 15min
completed: 2026-07-23
status: complete
---

# Phase 1 Plan 1: Package and Baseline Summary

**An isolated AIAP package now provides deterministic DFA obligation IDs and an ICFEM-compatible single-obligation witness BFS.**

## Performance

- **Duration:** 15 min
- **Started:** 2026-07-23T11:54:00Z
- **Completed:** 2026-07-23T12:09:06Z
- **Tasks:** 2
- **Files modified:** 8

## Accomplishments

- Added local PEP 621/setuptools metadata and a thin `python -m aiap` boundary.
- Implemented typed `CoverageObligation` and deterministic `ObligationCatalog` with stable IDs and JSON-safe meanings.
- Extracted the legacy token BFS, preserving interior multi-byte matching, initial-state empty witnesses, and `None` for non-completable targets.
- Added no-network fixtures and package/baseline tests.

## Task Commits

1. **Task 1: Create the nested package and stable DFA obligation catalog** - `9afbcca` (tests), `6944b57` (implementation)
2. **Task 2: Extract the legacy single-obligation BFS into the baseline module** - `9afbcca` (tests), `6944b57` (implementation)

## Files Created/Modified

- `pyproject.toml` - local package/build and pytest metadata.
- `aiap/obligations.py` - typed obligations and deterministic catalog.
- `aiap/baseline.py` - single-obligation breadth-first witness search and legacy obligation filtering.
- `tests/conftest.py`, `tests/test_packaging.py`, `tests/test_baseline.py` - deterministic fixtures and focused tests.

## Decisions Made

Local package discovery remains independent of the parent project; the parent DFA remains the sole transition authority.

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

The repository Git index required escalated permission for atomic commits; both planned commits completed successfully. Existing parent extension edits and the untracked compatibility demo were left untouched.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

Plan 02 can consume `ObligationCatalog` and `construct_one_witness` while adding canonical byte replay and certificate persistence.

## Self-Check: PASSED

- All eight planned files exist.
- Commits `9afbcca` and `6944b57` are present in Git history.
- Verification: `cd aiap && ../.venv/bin/python -m pytest tests/test_packaging.py tests/test_baseline.py -q` (5 passed).

---
*Phase: 01-baseline-replay-foundation*
*Completed: 2026-07-23*
