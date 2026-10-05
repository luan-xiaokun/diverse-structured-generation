---
phase: 01-baseline-replay-foundation
status: behavior_unverified
score: 2/3
---

# Phase 1 Verification

The canonical replay, certificate, one-case CLI, tokenizer lock, provisioner,
and dedicated CI job are implemented. Deterministic package/baseline/replay/CLI
tests pass (`13 passed`). The parity test now compares each accepted item's
legacy DFA state, byte-transition, and state-pair event sequences and compares
state/transition/path coverage ratios against the replay aggregate.

The real CSS/email parity command remains behavior-unverified in this
environment because the exact locked Hugging Face snapshot is not cached. It
fails with the prescribed provisioning command rather than skipping or using a
network fallback. Provision with:

`uv run python aiap/scripts/provision_real_case_tokenizers.py --lock aiap/tokenizer-lock.json`

then rerun with `HF_HUB_OFFLINE=1`.

## Checks

- `cd aiap && ../.venv/bin/python -m pytest tests/test_packaging.py tests/test_baseline.py tests/test_replay.py tests/test_cli.py -q` — 13 passed.
- `cd aiap && ../.venv/bin/python -m aiap --help` — one required explicit case argument; no batch/manifest mode.
- `HF_HUB_OFFLINE=1 ... tests/test_case_parity.py -m real_case_parity` — expected preflight failure until the locked snapshot is provisioned.

## Verdict

Implementation gaps are closed. Final acceptance requires running the locked
real-case parity job in an environment with Hugging Face access and retaining
its offline result.
