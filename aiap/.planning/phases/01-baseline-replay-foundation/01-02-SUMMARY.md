---
phase: 01-baseline-replay-foundation
plan: 02
subsystem: replay
tags: [dfa, replay, certificates, jsonl, coverage]
requires: [01-01]
provides: [canonical-replay, accepted-aggregates, jsonl-certificates]
affects: [02-multi-obligation-search, 03-suite-selection]
tech-stack:
  added: [stdlib-dataclasses, stdlib-jsonl]
  patterns: [byte-level-replay, accepted-only-gate]
key-files:
  created: [aiap/replay.py, aiap/reporting.py, tests/test_replay.py]
  modified: [aiap/__init__.py]
decisions:
  - "Replay derives all credited obligations from authoritative byte traces."
  - "Failed and non-final candidates remain diagnostic evidence but are aggregate-ineligible."
metrics:
  duration: "~20 minutes"
  completed: "2026-07-23"
status: complete
---

# Phase 1 Plan 2: Canonical Replay and Evidence Summary

Canonical byte-level replay certificates now validate acceptance, preserve token and byte traces, and serialize self-contained semantic obligation evidence as JSONL.

## Delivered

- Added typed byte-step, token-action, failure-diagnostic, covered-obligation, and candidate certificate records.
- Implemented `replay_candidate` with initial-state semantics, multi-byte expansion, explicit EOS zero-byte actions, destination mismatch detection, categorized failures, and accepted-only coverage claims.
- Added ordered event aggregates and coverage ratios based exclusively on aggregate-eligible certificates.
- Added versioned JSONL writer/reader and accepted-certificate gate with tuple meanings encoded as JSON arrays.
- Added deterministic tests for empty input, multi-byte tokens, EOS, failure isolation, aggregate multiplicity, and JSONL round-trip recovery.

## Deviations from Plan

None - plan executed as written.

## Verification

`../.venv/bin/python -m pytest tests/test_replay.py -q` — 5 passed.

## Commits

- `805c0dd` — feat(01-02): add canonical replay certificates
- `92227bf` — feat(01-02): persist replay certificates as JSONL
- `d775626` — fix(01-02): expose replay payload semantic aliases

## Self-Check: PASSED

- Created files exist: `aiap/replay.py`, `aiap/reporting.py`, `tests/test_replay.py`.
- All listed commits are present in Git history.
- No known stubs or new threat surfaces beyond the planned replay-record → JSONL evidence boundary.
