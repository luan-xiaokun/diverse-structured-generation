---
phase: 01-baseline-replay-foundation
plan: 03
status: complete
---
# Phase 1 Plan 3: One-case CLI and parity tooling Summary

Implemented explicit one-case `python -m aiap` CLI with schema-validated case loading, immutable tokenizer provenance, no-special-token source adapter, replay-backed JSON/JSONL artifacts, and thin `cov_test.py` delegation. Added locked tokenizer provisioner and real-case parity test scaffold with offline preflight diagnostics and pytest marker registration.

## Deviations

- Git commit could not be created because the shared repository `.git/index` is read-only in this execution environment. Parent agent should stage and commit these files.
- Real tokenizer snapshot is not present locally; parity tests intentionally fail with the required provisioning command rather than skip.

## Verification

`python -m aiap --help` succeeds. Existing deterministic baseline/replay tests pass. Real-case parity fails diagnostically when the locked snapshot is absent.
