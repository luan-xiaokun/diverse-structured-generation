# Phase 1: Baseline Replay Foundation - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-07-23
**Phase:** 1-Baseline Replay Foundation
**Areas discussed:** 模块边界, 证书内容, 语义兼容性, 测试基准

---

## 模块边界

| Decision | Selected | Notes |
|---|---|---|
| Packaging level | Formal Python package | User chose a package rather than a thin script/module split. |
| Primary interface | Command-first | `python -m aiap` is primary; `cov_test.py` is a migration runner. |
| Initial CLI granularity | One explicit case-study JSON | Batch/manifest orchestration is deferred. |
| Output | Readable summary + optional JSON artifact | Enables human inspection and later automation. |

## 证书内容

| Decision | Selected | Notes |
|---|---|---|
| Trace detail | Complete byte-level audit trace | Includes tokens, decoded text, states, byte steps, acceptance, coverage, failures. |
| Obligation identity | Stable ID + raw semantic tuple | Supports both deterministic comparisons and manual inspection. |
| Record format | JSONL, one candidate per line | Suitable for streaming and later pool composition. |
| Failure handling | Preserve but never credit | Failed/rejected records remain diagnostic only. |

## 语义兼容性

| Decision | Selected | Notes |
|---|---|---|
| Legacy metric relation | Independent replay parity required | Existing ICFEM metrics remain the baseline comparator. |
| Initial-state accounting | Once per accepted input | Includes accepted empty inputs. |
| Path semantics | Ordered state-pair per byte transition | Longer paths are out of Phase 1. |
| EOS | Zero-byte token-level termination | Recorded, but not a coverage event. |

## 测试基准

| Decision | Selected | Notes |
|---|---|---|
| Fixture strategy | Minimal DFA fixtures + CSS/email cases | Covers semantics and real integration. |
| Boundary set | Complete core boundary set | Initial/empty/multi-byte/EOS/rejected/non-completable cases. |
| Real-case parity | Per-item parity | Compare ratios, sets, counts, and witness acceptance. |
| Test integration | `aiap/tests/` via pytest | Reuse parent environment. |

## the agent's Discretion

- Internal module names, typed field names, and minimal packaging configuration, subject to CONTEXT.md's locked external behavior.

## Deferred Ideas

- Full batch/manifest experiment orchestration is deferred to evaluation work.
- Multi-obligation search, suite selection, and LLM quality guidance remain in their planned later phases.
