# Roadmap: AIAP Multi-Obligation Witness Generation

## Overview

v1 turns the exploratory single-obligation DFA traversal into a trace-first research prototype. It first freezes canonical replay correctness, then establishes feasible multi-obligation search, selection from a shared verified candidate pool, DFA-masked LLM quality guidance, and finally a locked case-study protocol with artifact-backed evidence. Each phase preserves DFA replay as the sole source of validity and coverage claims; exact optimization is used only as a small-instance diagnostic oracle.

## Phases

**Phase Numbering:**

- Integer phases (1, 2, 3): Planned milestone work
- Decimal phases (2.1, 2.2): Urgent insertions (marked with INSERTED)

Decimal phases appear between their surrounding integers in numeric order.

- [x] **Phase 1: Baseline Replay Foundation** - Modularize the demo and make canonical trace replay the correctness boundary.
- [ ] **Phase 2: Feasible Multi-Obligation Search** - Build jointly feasible targets and bounded coverage-aware search with oracle diagnostics.
- [ ] **Phase 3: Complementary Suite Selection** - Select compact, high-coverage suites from a shared verified candidate pool.
- [ ] **Phase 4: DFA-Masked LLM Quality Guidance** - Improve quality only within DFA-legal actions and retain replay evidence.
- [ ] **Phase 5: Locked Evaluation Evidence** - Run reproducible, equal-budget case studies and retain paper-ready evidence.

## Phase Details

### Phase 1: Baseline Replay Foundation

**Goal**: Researchers can reproduce the existing single-obligation baseline while mechanically validating every candidate's acceptance and claimed coverage with one canonical DFA replay contract.
**Depends on**: Nothing (first phase)
**Requirements**: BASE-01, BASE-02
**Success Criteria** (what must be TRUE):

  1. A researcher can import and run the modular single-obligation BFS baseline on a fixed case-study input and reproduce its accepted witnesses and coverage results.
  2. Every baseline or later candidate can be replayed through the same DFA trace path to confirm acceptance and actual state, byte-transition, and ordered state-pair coverage.
  3. A reported witness carries a replayed coverage certificate, so no result is credited from predicted rather than actual trace coverage.

**Plans**: 3/3 plans executed

- [x] 01-01-PLAN.md
- [x] 01-02-PLAN.md
- [x] 01-03-PLAN.md

### Phase 01.1: Repair tokenizer vocabulary and byte-DFA contract (INSERTED)

**Goal:** Tokenizer vocabulary entries traverse the DFA as exact raw bytes across the Python/Rust boundary, persisted artifacts enforce the new bytes-only schema, and witness selection separates byte-level-unreachable uncovered obligations from actual generation failures.
**Requirements**: Contract repair for the existing Phase 1 baseline; no new milestone requirement IDs
**Depends on:** Phase 1
**Plans:** 7 plans

**Success Criteria** (what must be TRUE):

  1. The public vocabulary API is bytes-only at version 0.3.0, reconstructs source UTF-8 bytes exactly for supported tokenizer families, and rejects string-valued vocabularies with an actionable `TypeError`.
  2. Serialized DFA artifacts carry an explicit magic header and schema version, restore without tokenizer/network access, and strictly distinguish legacy, unsupported-schema, and corrupt inputs.
  3. Synthetic boundary fixtures and the locked offline Qwen/CSS case demonstrate exact acceptance, ordered-state, transition, and path parity between token-driven and direct byte traversal.
  4. Only structures missed by the selected baseline samples are obligations; byte-level-unreachable obligations remain diagnosed in aggregate JSON but do not consume `--obligation-num`, while JSONL remains limited to attempted witness certificates.

Plans:

- [ ] 01.1-01-PLAN.md — Replace tokenizer vocabulary, PyO3, and Rust traversal with one exact bytes-only contract.
- [ ] 01.1-02-PLAN.md — Convert repository callers and prove tests use the rebuilt native extension.
- [ ] 01.1-03-PLAN.md — Add the strict self-contained schema-1 artifact envelope and failure taxonomy.
- [ ] 01.1-04-PLAN.md — Synchronize release 0.3.0, regenerate locks, and install the final native source state.
- [ ] 01.1-05-PLAN.md — Classify accepting token-action reachability and define bounded replay-backed outcomes.
- [ ] 01.1-06-PLAN.md — Enforce missing/classify/select ordering and population-correct JSON, JSONL, and terminal reporting.
- [ ] 01.1-07-PLAN.md — Enforce hermetic synthetic and locked offline full-trace parity gates.

### Phase 2: Feasible Multi-Obligation Search

**Goal**: Researchers can generate trace-certified witnesses that jointly cover feasible obligations and quantify the bounded search's approximation behavior without claiming global optimality.
**Depends on**: Phase 1
**Requirements**: MULTI-01, MULTI-02, MULTI-03, MULTI-04
**Success Criteria** (what must be TRUE):

  1. A researcher can inspect a stable-ID catalog of live uncovered obligations, including recorded reasons for items or bundles that are unreachable or not jointly feasible on an accepting trace.
  2. A researcher can form and validate target groups containing multiple obligations, and obtain accepted witnesses that replay as covering their reported joint coverage.
  3. A deterministic dominance-aware search retains coverage-distinct labels at the same DFA state and reports pruning, cap, and failure outcomes alongside each coverage certificate.
  4. On small DFA or candidate-pool fixtures, a researcher can compare bounded search with exhaustive or restricted exact-oracle results to measure cap/pruning effects, with no global-optimality assertion.

**Plans**: TBD

### Phase 3: Complementary Suite Selection

**Goal**: Researchers can assemble a compact, reproducible complementary suite from verified ICFEM and AIAP candidates and compare its coverage trade-offs with fixed-pool baselines.
**Depends on**: Phase 2
**Requirements**: SUIT-01, SUIT-02, SUIT-03
**Success Criteria** (what must be TRUE):

  1. A researcher can replay ICFEM 2025 samples and new witnesses into one candidate pool that records source, cost, and obligation-incidence evidence.
  2. A researcher can rerun deterministic greedy marginal-gain selection and obtain the same selected suite, tie-breaks, marginal gains, and source composition.
  3. On the same validated pool, a researcher can compare ICFEM-only, AIAP-only, union, random, and greedy methods by coverage@k and suite size needed for target coverage.

**Plans**: TBD

### Phase 4: DFA-Masked LLM Quality Guidance

**Goal**: Researchers can use an LLM only to rank or guide DFA-legal actions and measure quality effects without delegating validity or coverage decisions to the model.
**Depends on**: Phase 3
**Requirements**: QUAL-01, QUAL-02, QUAL-03
**Success Criteria** (what must be TRUE):

  1. A researcher can run beam/search or reranking restricted to DFA-legal token actions, while the LLM has no authority to accept a witness or credit coverage.
  2. Every LLM-influenced witness is freshly replayed and reported with DFA validity, coverage retention, length, and generation cost.
  3. At matched coverage and length, a researcher can compare quality-aware and quality-unaware valid witnesses using a fixed automatic proxy and a documented blinded naturalness pilot protocol.

**Plans**: TBD

### Phase 5: Locked Evaluation Evidence

**Goal**: Researchers can reproduce the study's scoped coverage, quality, and complementarity evidence from locked manifests and preserved per-case artifacts.
**Depends on**: Phase 4
**Requirements**: EVAL-01, EVAL-02, EVAL-03, EVAL-04
**Success Criteria** (what must be TRUE):

  1. A versioned manifest identifies each case study, initial samples, DFA/tokenizer/model revisions, seeds, budgets, and decoding/search parameters used in a result.
  2. A researcher can run B0/B1, grouping, selection, and LLM ablations under a fixed development/held-out protocol with equivalent resource budgets.
  3. Results are available per case, seed, and coverage criterion with absolute/relative coverage, suite size, efficiency, failures, and uncertainty rather than only one aggregate metric.
  4. A researcher can rerun core experiments from preserved raw artifacts and trace conclusions only to measured DFA coverage, witness quality, and suite complementarity.

**Plans**: TBD

## Progress

**Execution Order:**
Phases execute in numeric order: 1 → 2 → 3 → 4 → 5

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. Baseline Replay Foundation | 3/3 | Complete | 2026-07-23 |
| 2. Feasible Multi-Obligation Search | 0/TBD | Not started | - |
| 3. Complementary Suite Selection | 0/TBD | Not started | - |
| 4. DFA-Masked LLM Quality Guidance | 0/TBD | Not started | - |
| 5. Locked Evaluation Evidence | 0/TBD | Not started | - |
