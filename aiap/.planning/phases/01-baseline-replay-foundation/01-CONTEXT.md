# Phase 1: Baseline Replay Foundation - Context

**Gathered:** 2026-07-23
**Status:** Ready for planning

<domain>
## Phase Boundary

Turn the exploratory single-obligation witness demo into a formal Python package with a command-first interface, while establishing the canonical DFA replay contract that validates every candidate's acceptance and byte-level coverage claim. This phase provides a lean baseline and verification boundary for later multi-obligation search, suite selection, LLM quality guidance, and evaluation; it does not add those later algorithms or a full experiment platform.

</domain>

<decisions>
## Implementation Decisions

### Module Boundary and CLI
- **D-01:** Build a formal Python package now; do not leave the baseline solely as an ad-hoc script.
- **D-02:** Make `python -m aiap` the primary public entry point. Keep `cov_test.py` only as a thin migration/compatibility runner.
- **D-03:** The first command operates on one explicit case-study JSON input, rather than a default batch or manifest runner.
- **D-04:** The command emits a human-readable stdout summary and can write a machine-readable JSON result/certificate artifact.

### Replay Certificate
- **D-05:** Every candidate gets a complete auditable byte-level replay certificate, including token IDs, decoded text, initial/final states, every byte step `(byte, from_state, to_state)`, acceptance, covered obligation IDs, and failure diagnostics.
- **D-06:** Each covered obligation has both a stable semantic ID and raw DFA meaning: state, byte-transition, or ordered state-pair tuple.
- **D-07:** Persist candidate records as JSONL, one candidate per line, so results can be streamed and later combined into candidate pools.
- **D-08:** Preserve invalid, rejected, or trace-mismatched candidates with a categorized failure record, but make it impossible for them to contribute to coverage aggregates or later selection inputs.

### Metric and Replay Semantics
- **D-09:** Retain the existing ICFEM coverage metrics as the baseline comparator. Independently recompute coverage from replay certificates and require fixed-case parity.
- **D-10:** Count the initial DFA state once for every accepted input, including accepted empty strings, matching the current baseline behavior.
- **D-11:** Define path coverage as the ordered state-pair of each actual byte transition; do not introduce longer paths in Phase 1.
- **D-12:** Record EOS as a zero-byte token-level termination action. It can establish/confirm a final state but never creates state, byte-transition, or path coverage events.

### Test Baseline
- **D-13:** Use layered tests: deterministic minimal DFA fixtures for semantic boundaries, plus CSS color and email case studies for real tokenizer/DFA parity.
- **D-14:** Minimal fixtures must cover initial state, accepted empty input, a hit inside a multi-byte token, EOS, rejected input, and a reachable but non-completable obligation.
- **D-15:** Real-case parity is per-item: compare legacy metric ratios, replayed state/transition/path sets and counts, and witness acceptance. Every discrepancy must have an explicit category.
- **D-16:** Put AIAP tests under `aiap/tests/` and run them with the parent repository's pytest environment.

### the agent's Discretion
- Exact internal module names and typed record field names may follow repository conventions, provided the command-first package boundary and certificate contract above remain intact.
- The planner may choose the minimal safe packaging metadata/configuration needed to support `python -m aiap` and pytest without turning Phase 1 into a general experiment framework.

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Project and Phase Contract
- `.planning/PROJECT.md` — project core value, isolation constraint, and DFA/LLM correctness boundary.
- `.planning/REQUIREMENTS.md` — `BASE-01` and `BASE-02`, plus traceability constraints from later phases.
- `.planning/ROADMAP.md` — Phase 1 goal, success criteria, and dependency role.
- `.planning/STATE.md` — current planning position and cross-phase decisions.

### Current Prototype and Parent DFA APIs
- `cov_test.py` — existing one-obligation BFS, obligation counting, token/byte handling, and current case-study demo.
- `../regex_dfa_guide/src/diverse_guide_dfa.rs` — authoritative DFA acceptance, string transition sequences, and byte transition API semantics.
- `../regex_dfa_guide/src/python_bindings/mod.rs` — Python-visible signatures for DFA state, transition, and byte-trace methods.
- `../tests/test_metrics.py` — existing state/transition/path metric semantics used for required parity.
- `../tests/test_dfa.py` — parent DFA transition and final-state test patterns.

### Existing AIAP Orientation
- `.planning/codebase/TESTING.md` — current testing gaps and recommended fixture strategy.
- `.planning/codebase/CONVENTIONS.md` — Python typing, import, error-handling, and naming patterns.
- `.planning/codebase/STRUCTURE.md` — current AIAP placement and parent-repository integration points.
- `.planning/research/SUMMARY.md` — trace-first correctness safeguard and later-phase dependency chain.

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `cov_test.py:CoverageObligation` — current typed representation for state, transition, and path obligations; can guide the packaged domain model.
- `cov_test.py:get_coverage_obligations` — current counting semantics to preserve for legacy parity.
- `cov_test.py:construct_one_witness` — single-obligation BFS reference implementation to retain as the baseline.
- `DiverseGuideDFA` methods exposed through `regex_dfa_guide` — authoritative initial/final states, accepted token IDs, token transitions, string traces, and byte-level token traces.
- Parent `tests/test_metrics.py` and `tests/test_dfa.py` — existing pytest fixture/assertion patterns for metric and DFA behavior.

### Established Patterns
- Python uses `snake_case` functions/modules, `PascalCase` typed records, explicit annotations, and standard/third-party/local import grouping.
- Current code recognizes that a token can traverse multiple byte transitions; any replay implementation must preserve this distinction.
- `ValueError` from the byte-transition API currently signals EOS-like token handling; Phase 1 must make this behavior explicit in certificates rather than silently treating it as ordinary bytes.

### Integration Points
- The new package lives under `aiap/`, reuses the parent `regex_dfa_guide` and `diverse_guide` packages, and reads existing case-study JSON files under `../experiments/case_study/data/` when run from `aiap/`.
- `python -m aiap` and `aiap/tests/` must work within the parent repository's existing Python environment without altering the parent library's behavior.

</code_context>

<specifics>
## Specific Ideas

- A certificate must be sufficiently complete to recompute coverage from raw traces rather than trusting search-time predictions.
- Failure records are research evidence: retain them for diagnosis, but enforce a hard separation from accepted coverage data.
- The first CLI remains intentionally single-case; batch/manifest orchestration is deferred to the evaluation work.

</specifics>

<deferred>
## Deferred Ideas

- Complete batch/manifest experiment CLI orchestration — belongs to the locked evaluation evidence phase.
- Multi-obligation grouping and dominance-aware search — Phase 2.
- Candidate-pool selection — Phase 3.
- LLM scoring/reranking — Phase 4.

</deferred>

---
*Phase: 1-Baseline Replay Foundation*
*Context gathered: 2026-07-23*
