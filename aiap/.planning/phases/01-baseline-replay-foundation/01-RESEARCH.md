# Phase 1: Baseline Replay Foundation - Research

**Researched:** 2026-07-23  
**Domain:** Python package boundary, canonical DFA byte replay, replay-backed coverage evidence  
**Confidence:** HIGH for local API/metric semantics; MEDIUM for the standalone package layout because it is a new project boundary.

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

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

### Deferred Ideas (OUT OF SCOPE)

- Complete batch/manifest experiment CLI orchestration — belongs to the locked evaluation evidence phase.
- Multi-obligation grouping and dominance-aware search — Phase 2.
- Candidate-pool selection — Phase 3.
- LLM scoring/reranking — Phase 4.
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| BASE-01 | Convert the single-obligation BFS demo into an importable baseline that reproduces fixed-case accepted witnesses and coverage. | Preserve the `CoverageObligation`, `get_coverage_obligations`, and `construct_one_witness` semantics in focused modules; use one explicit case JSON through the CLI. |
| BASE-02 | Use the same DFA trace replay to validate acceptance and actual state, byte-transition, and ordered state-pair coverage for every witness. | Make replay the only constructor of coverage claims; recompute aggregate counts from accepted certificates and compare them with the parent metrics. |
</phase_requirements>

## Summary

Phase 1 should be a deliberately small, flat Python package rooted at `aiap/aiap/`, with `aiap/__main__.py` delegating to an importable `cli.main()`. Running from the AIAP project directory makes `python -m aiap CASE.json` work without changing the parent project's `src/` package discovery; an `aiap/pyproject.toml` can make the same package editable/installable later. Python executes a package's `__main__.py` for `python -m package`, and recommends keeping it short and delegating testable behavior to ordinary modules. [CITED: https://docs.python.org/3/library/__main__.html]

The canonical boundary is not tokenizer decoding or the search prediction. It is an ordered replay of the proposed token IDs through `DiverseGuideDFA`: obtain the token-level destination, expand the exact byte trace from the pre-token state, and derive all coverage events only from those byte steps. The parent Rust API returns a sentinel initial entry `(0, initial_state)` for string and token byte sequences; its EOS branch validates the token transition but returns only that sentinel, so replay must emit a token action with `byte_steps: []`. [VERIFIED: local `../regex_dfa_guide/src/diverse_guide_dfa.rs`]

**Primary recommendation:** Build `case_io`, `obligations`, `baseline`, `replay`, `reporting`, and `cli` as small typed modules; store one self-contained `CandidateCertificate` per JSONL line, and let only a successful replay populate a separate accepted-certificate aggregate.

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| Single-case CLI and artifact paths | Python CLI | Filesystem | The command parses one explicit case path and streams result records. |
| DFA construction and legal token transitions | Parent DFA library | AIAP adapter | `DiverseRegexGuide` builds the DFA from the case regex and tokenizer; AIAP must not recreate formal-language semantics. [VERIFIED: local `../src/diverse_guide/guide_rust.py`] |
| Single-obligation BFS | AIAP baseline module | Parent DFA library | AIAP owns traversal policy while calling parent legal-token and byte-trace APIs. |
| Acceptance, byte trace, coverage evidence | AIAP replay module | Parent DFA library | AIAP derives certificates; the parent DFA remains authoritative for transitions/final states. |
| JSONL certificate persistence | AIAP reporting module | Filesystem | One serialized candidate record per line supports streaming and later candidate-pool intake. |
| Legacy metric comparison | Parent metrics module | AIAP aggregate module | Parent ratios remain the comparator; independently counted replay data provides parity evidence. |

## Standard Stack

### Core

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| Python standard library (`argparse`, `dataclasses`, `json`, `pathlib`, `collections`) | Python 3.12.13 available | CLI, typed records, JSONL, paths, BFS queue | No new dependency is required for this bounded baseline. [VERIFIED: local environment] |
| `regex_dfa_guide` | parent editable source | DFA states, byte transitions, token transitions, final-state queries | It is the project’s existing formal-transition authority. [VERIFIED: local `../regex_dfa_guide/src/python_bindings/mod.rs`] |
| `diverse_guide` | parent editable source | `DiverseRegexGuide`, `build_token_id_map`, legacy coverage metrics | This is the existing ICFEM baseline implementation to preserve for parity. [VERIFIED: local `../src/diverse_guide/guide_rust.py`, `../src/diverse_guide/evaluation/metrics.py`] |
| `transformers` | `>=5.3.0` declared by parent | Load the case’s tokenizer and decode display text | Already declared by the parent project; Phase 1 adds no model or package. [VERIFIED: local `../pyproject.toml`] |
| `pytest` | 9.0.2 available | Unit and real-case parity tests | Parent project already configures and uses pytest. [VERIFIED: local `../pyproject.toml`, environment] |

### Supporting

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| Parent `state_coverage`, `transition_coverage`, `path_coverage` | current source | Legacy ratios | Only as the independently maintained comparator in parity tests and CLI summary. [VERIFIED: local `../src/diverse_guide/evaluation/metrics.py`] |

### Alternatives Considered

| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Flat `aiap/aiap/` package plus local metadata | Alter the parent root’s `src/` layout | The parent `pyproject.toml` discovers only `src/`; changing it expands scope and risks the unrelated parent package. [VERIFIED: local `../pyproject.toml`] |
| Typed stdlib records + JSONL | Pydantic/database/artifact platform | Extra dependencies and a generalized experiment platform are outside Phase 1. |
| Direct DFA replay | A second Python DFA implementation | A duplicate semantics source could diverge from the Rust-backed production DFA and invalidate parity. |

**Installation:** No external package installation is recommended. Reuse the parent `.venv` (Python 3.12.13, pytest 9.0.2, uv 0.11.23, maturin 1.12.6 are available). [VERIFIED: local environment]

**Package layout recommendation:**

```text
aiap/
├── pyproject.toml             # minimal project metadata; no new dependencies
├── aiap/
│   ├── __init__.py
│   ├── __main__.py            # from .cli import main; raise SystemExit(main())
│   ├── cli.py
│   ├── case_io.py
│   ├── obligations.py
│   ├── baseline.py
│   ├── replay.py
│   └── reporting.py
├── tests/
│   ├── conftest.py
│   ├── test_replay.py
│   ├── test_baseline.py
│   └── test_case_parity.py
└── cov_test.py                # thin compatibility call into `aiap.cli`
```

Run package and tests from `aiap/`: `python -m aiap ../experiments/case_study/data/diverse-email.json --jsonl out.jsonl` and `pytest tests`. This uses the parent virtual environment while avoiding any mutation to the parent root’s package-discovery configuration. [VERIFIED: local `../pyproject.toml`]

## Architecture Patterns

### System Architecture Diagram

```text
explicit case JSON ──► case_io ──► tokenizer + DiverseRegexGuide ──► DFA
                                   │                                │
                                   │                                ├──► legacy metrics (comparison only)
                                   ▼                                │
                           initial samples ──► obligation catalog ──┤
                                                                    ▼
                                                      single-obligation BFS
                                                                    │ token IDs
                                                                    ▼
candidate proposal ─────────────────────────────────────────► canonical replay
                                                               │      │
                                             malformed/rejected │      │ accepted
                                                               ▼      ▼
                                                        failure record  covered IDs + byte trace
                                                               │      │
                                                               └──► JSONL certificate stream
                                                                       │
                                                                       ▼
                                                               replay aggregate + legacy parity
```

### Recommended Project Structure

Use the package layout above. Keep the compatibility runner free of logic, and do not move parent code or datasets into AIAP. The current demo already isolates its traversal from the parent library and is untracked; Phase 1 should preserve that isolation. [VERIFIED: local `cov_test.py`, `git status`]

### Pattern 1: One source of coverage truth

**What:** `replay_candidate(dfa, token_ids, tokenizer, catalog)` returns either an accepted certificate or a failure certificate. Only the accepted variant exposes `covered_obligation_ids` to `aggregate_certificates`.

**When to use:** For every witness—existing sample replay, BFS output, and future Phase 2–4 candidates.

**Example:**

```python
def replay_token_ids(dfa, token_ids, eos_token_id, catalog):
    state = dfa.get_initial_state()
    state_events = [state]
    byte_steps = []
    token_actions = []

    for token_id in token_ids:
        before = state
        after = dfa.get_next_token_state(before, token_id)
        sequence = dfa.get_byte_transition_sequence_from_state(before, token_id)
        steps = []
        previous = before
        for byte, next_state in sequence[1:]:
            steps.append((byte, previous, next_state))
            byte_steps.append((byte, previous, next_state))
            state_events.append(next_state)
            previous = next_state
        token_actions.append((token_id, before, after, token_id == eos_token_id, steps))
        state = after

    accepted = dfa.is_final_state(state)
    return accepted, state_events, byte_steps, token_actions
```

The initial `state_events` entry is the required per-input initial-state count. An EOS action has no `sequence[1:]` entries and adds no state/transition/path event. [VERIFIED: local `../regex_dfa_guide/src/diverse_guide_dfa.rs`]

### Pattern 2: Stable semantic catalog, raw transition payload

**What:** Construct a frozen per-DFA `ObligationCatalog` from `get_states()` and `get_transitions()`: `state:<n>`, `transition:<from>:<byte>:<to>`, and `path:<from>:<to>`. Keep raw integer values as JSON arrays, never Python tuples, so JSON round-tripping is unambiguous.

**When to use:** Before measuring initial samples and before replaying any candidate for the same case/tokenizer/DFA configuration.

**Example certificate shape:**

```json
{
  "schema_version": 1,
  "candidate_id": "b1-transition-0-97-1",
  "proposal": {"kind": "baseline_bfs", "target_obligation_ids": ["transition:0:97:1"]},
  "token_ids": [10, 99],
  "decoded_text": "a",
  "replay": {
    "initial_state": 0,
    "final_state": 1,
    "token_actions": [
      {"token_id": 10, "from_state": 0, "to_state": 1, "is_eos": false,
       "byte_steps": [{"byte": 97, "from_state": 0, "to_state": 1}]},
      {"token_id": 99, "from_state": 1, "to_state": 1, "is_eos": true, "byte_steps": []}
    ],
    "accepted": true,
    "covered_obligation_ids": ["state:0", "state:1", "transition:0:97:1", "path:0:1"]
  },
  "failure": null
}
```

`decoded_text` is a display/provenance field; transition truth comes from `token_ids` expanded with `get_byte_transition_sequence_from_state`, because tokenizer presentation can differ from the byte vocabulary used to build the DFA. The parent vocabulary builder specifically handles byte/BPE and invalid UTF-8 token representations. [VERIFIED: local `../src/diverse_guide/vocab.py`]

### Pattern 3: Failure isolation is structural

**What:** A replay error returns `accepted: false`, `covered_obligation_ids: []`, `aggregate_eligible: false`, and a required `failure` object. Suggested categories: `unknown_token`, `illegal_token_transition`, `byte_trace_error`, `token_destination_mismatch`, `not_final`, `decoded_text_mismatch`, and `case_schema_error`.

**When to use:** Any exception from a parent DFA call, any discrepancy between token-level destination and final byte state, and any candidate ending outside a final state.

**Anti-patterns to Avoid**

- **Trusting BFS intent:** A target obligation is not evidence. Recompute the covered IDs from the replay trace.
- **Crediting rejected candidates:** Do not merge their traces/counts into a coverage aggregate, even for diagnostics.
- **Counting EOS as a byte transition:** It is zero byte by contract.
- **Replaying only `decoded_text`:** Preserve token action identity and inspect each action’s byte path; a token may cross multiple DFA transitions.
- **Using current working directory for case lookup:** Require the case path argument; resolve it with `Path(...).resolve()` and write artifacts only to the explicit output path.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| DFA acceptance/transitions | A Python regex/DFA mirror | `DiverseGuideDFA` methods | Parent Rust extension defines actual state IDs and legal byte/token paths. [VERIFIED: local bindings and Rust source] |
| Regex-to-DFA/token vocabulary binding | A separate tokenizer-vocabulary conversion | `DiverseRegexGuide` and `build_token_id_map` | Parent code handles tokenizer-specific byte and replacement-token cases. [VERIFIED: local `../src/diverse_guide/guide_rust.py`, `../src/diverse_guide/vocab.py`] |
| Legacy ratios | A renamed copy of parent metrics | Parent metrics plus independent replay aggregate | Retain the ICFEM comparator while making parity meaningful. [VERIFIED: local `../src/diverse_guide/evaluation/metrics.py`] |
| CLI framework | Click/Typer or a general experiment runner | `argparse` | One command with explicit case/output needs no new dependency. |

**Key insight:** The AIAP code may own how it records and validates a trace, but must not own an alternate interpretation of the DFA.

## Common Pitfalls

### Pitfall 1: Mixing token and byte boundaries

**What goes wrong:** A token can contain several bytes and cross several DFA edges, but an implementation credits only the token’s end state.  
**Why it happens:** BFS is token-level while all legacy coverage universes are byte-level.  
**How to avoid:** Record every `sequence[1:]` byte step for each non-EOS token and derive state, transition, and path coverage from that flattened list.  
**Warning signs:** A minimal multi-byte-token fixture reports no interior transition/path, or replay counts differ from `get_transition_sequence(decoded_text)`.

### Pitfall 2: Incorrect initial-state and empty-input accounting

**What goes wrong:** The initial state is omitted, especially for an accepted empty string.  
**Why it happens:** Transition-only code initializes its count inside a byte loop.  
**How to avoid:** Begin every successful replay with exactly one initial-state event before action expansion; accept empty token lists iff the initial state is final.  
**Warning signs:** Empty-input state coverage is zero or differs from parent `state_coverage`. [VERIFIED: local `../src/diverse_guide/evaluation/metrics.py`]

### Pitfall 3: Treating EOS as an ordinary byte

**What goes wrong:** The sentinel `(0, state)` appears as an invented `byte=0` edge or adds a state visit.  
**Why it happens:** The API intentionally uses a sentinel first element to anchor a sequence.  
**How to avoid:** Persist EOS as a token action with `byte_steps: []`, then use only `sequence[1:]` to create coverage events.  
**Warning signs:** An obligation ID contains byte zero, or EOS changes metric ratios. [VERIFIED: local `../regex_dfa_guide/src/diverse_guide_dfa.rs`]

### Pitfall 4: Path-definition drift

**What goes wrong:** “Path” becomes token path or a longer multi-edge path.  
**Why it happens:** The term is overloaded.  
**How to avoid:** Preserve the parent definition: one ordered `(from_state, to_state)` pair for every actual byte transition; distinct bytes with the same pair share the path obligation. [VERIFIED: local `../src/diverse_guide/evaluation/metrics.py`]

### Pitfall 5: Making parity non-diagnostic

**What goes wrong:** A single aggregate mismatch leaves no way to identify the first divergent input or criterion.  
**How to avoid:** For each fixed sample and BFS witness, compare acceptance, state/transition/path event lists and counters, then compare the three final ratios. Emit `parity_category` (`acceptance`, `state_events`, `transition_events`, `path_events`, `ratio`, or `none`).

### Pitfall 6: CWD-dependent behavior and accidental parent changes

**What goes wrong:** The old script’s `experiments/...` paths only work from a particular directory; changing the parent `pyproject.toml` to compensate expands scope.  
**How to avoid:** Require one explicit input path, test from `aiap/`, and leave parent library files and root package metadata untouched.

## Code Examples

### Legacy-compatible aggregate from certificates

```python
def aggregate_accepted(certs, catalog, threshold=1):
    counts = {item.id: 0 for item in catalog.items}
    for cert in certs:
        if not cert.replay.accepted or cert.failure is not None:
            continue
        for obligation_id in cert.replay.covered_obligation_ids:
            counts[obligation_id] += 1
    return counts
```

For strict legacy parity, use per-input event multiplicity (not only unique covered IDs) when building counts. A certificate can retain both a deduplicated `covered_obligation_ids` list and full ordered `state_events`/`byte_steps`; the latter drives counters. The legacy metric functions count every state-sequence entry and every byte transition/path occurrence. [VERIFIED: local `../src/diverse_guide/evaluation/metrics.py`]

### Minimal `__main__.py`

```python
from .cli import main

raise SystemExit(main())
```

The short `__main__.py` and reusable `main()` arrangement follows the documented package-entry-point pattern. [CITED: https://docs.python.org/3/library/__main__.html]

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Ad-hoc `cov_test.py` prints one witness from hard-coded case paths | Importable package + explicit single-case command + JSONL evidence | Phase 1 | Makes the baseline reusable and inspectable without creating a batch platform. |
| Predicted obligation match during BFS | Fresh canonical byte replay before any credit | Phase 1 | Later algorithms cannot claim coverage without trace evidence. |

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | The intended command is executed from the `aiap/` project directory, where a nested `aiap/` package gives direct `python -m aiap` support without changing the parent `src/` package layout. | Standard Stack | If users require execution from the parent root without installation, a wrapper or explicit editable installation instruction will be needed. |

## Open Questions

1. **Tokenizer provenance granularity**
   - What we know: Case JSON includes a model identifier, and the current demo hard-codes a different tokenizer identifier (`Qwen/Qwen2.5-0.5B`). [VERIFIED: local case JSON and `cov_test.py`]
   - What's unclear: Whether Phase 1 should default to the case `model` field or require a CLI `--tokenizer` override.
   - Recommendation: Make tokenizer identity explicit in the CLI/result, defaulting only from the case field if it exists; do not silently retain the demo’s hard-coded model.

2. **Display-text assertion**
   - What we know: DFA legality is built from `build_token_id_map`, whereas `tokenizer.decode` is a user-facing rendering operation. [VERIFIED: local `../src/diverse_guide/vocab.py`]
   - What's unclear: Whether all supported tokenizer variants round-trip every token sequence byte-for-byte through `decode`.
   - Recommendation: Record both, treat token-ID replay as authoritative, and classify a rendering discrepancy rather than crediting it or crashing.

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|-------------|-----------|---------|----------|
| Python | package/CLI/tests | ✓ | 3.12.13 | — |
| pytest | AIAP tests | ✓ | 9.0.2 | — |
| uv | existing parent environment management | ✓ | 0.11.23 | direct `.venv/bin/python`/`pytest` |
| maturin | existing Rust extension rebuilds, if needed | ✓ | 1.12.6 | not needed for Python-only Phase 1 changes |
| Parent `diverse_guide` and `regex_dfa_guide` imports | DFA construction/replay | ✓ | editable local source | — |
| Hugging Face tokenizer artifacts | real CSS/email tests | conditional | case/model dependent | Unit fixtures must not require network; cache/pin artifacts for real parity |

**Missing dependencies with no fallback:** None identified for package and fixture work.  
**Missing dependencies with fallback:** Real-case tokenizer artifacts may be unavailable offline; mark those tests separately and use the deterministic fixture layer for the fast required test command.

## Security Domain

### Applicable ASVS Categories

| ASVS Category | Applies | Standard Control |
|---------------|---------|-----------------|
| V2 Authentication | no | No users or credentials in scope. |
| V3 Session Management | no | No sessions in scope. |
| V4 Access Control | no | Local command operates on user-supplied files only. |
| V5 Input Validation | yes | Validate case JSON structure/types, constrain output path to an explicit filesystem path, catch DFA/token exceptions into failure records. |
| V6 Cryptography | no | No cryptographic feature in scope. |

### Known Threat Patterns for Python CLI/artifacts

| Pattern | STRIDE | Standard Mitigation |
|---------|--------|---------------------|
| Oversized/malformed case JSON | Denial of service | Validate top-level schema before tokenizer/DFA work; impose a documented input-size/candidate limit when reading untrusted artifacts. Python’s JSON documentation warns that malicious JSON can consume CPU/memory. [CITED: https://docs.python.org/3/library/json.html] |
| Candidate text used in shell/log format | Tampering / information disclosure | Never shell-interpolate witness text; serialize with `json.dump` and use structured diagnostic fields. |
| Arbitrary output overwrite | Tampering | Require an explicit output path and refuse directory targets; do not write to input datasets. |

## Sources

### Primary (HIGH confidence)

- Local `../regex_dfa_guide/src/diverse_guide_dfa.rs` and `../regex_dfa_guide/src/python_bindings/mod.rs` — authoritative byte/token transition, sequence, EOS, and Python binding semantics.
- Local `../src/diverse_guide/evaluation/metrics.py` — exact ICFEM state/transition/path counting and ratio semantics.
- Local `cov_test.py` — current single-obligation catalog and BFS baseline behavior.
- Local `../src/diverse_guide/guide_rust.py` and `../src/diverse_guide/vocab.py` — DFA construction and tokenizer byte-vocabulary mapping.
- Local `../pyproject.toml`, `../tests/conftest.py`, `../tests/test_metrics.py`, and `../tests/test_dfa.py` — parent environment, fixture, and assertion patterns.

### Secondary (MEDIUM confidence)

- [Python `__main__` documentation](https://docs.python.org/3/library/__main__.html) — package execution and minimal CLI entry-point pattern.
- [Python `json` documentation](https://docs.python.org/3/library/json.html) — stdlib serialization and untrusted-input caution.

### Tertiary (LOW confidence)

- None.

## Metadata

**Confidence breakdown:**

- Standard stack: HIGH — all recommended runtime dependencies already exist in the parent project; no new package is proposed.
- Architecture: HIGH — driven by inspected parent DFA/metric code and locked Phase 1 decisions.
- Pitfalls: HIGH — token/byte/EOS and metric risks are directly evidenced in the local Rust and Python implementations.

**Research date:** 2026-07-23  
**Valid until:** 2026-08-22 for local integration assumptions; revisit if the parent DFA API changes.
