---
gsd_state_version: 1.0
milestone: v1.0
milestone_name: milestone
current_phase: 2
current_phase_name: Feasible Multi-Obligation Search
status: planning
stopped_at: Phase 1 and 01.1 complete; Phase 2 awaiting discussion and planning
last_updated: "2026-10-05T07:58:30.858Z"
last_activity: 2026-10-05
last_activity_desc: Refreshed project state and exported HANDOFF.md
progress:
  total_phases: 6
  completed_phases: 2
  total_plans: 10
  completed_plans: 10
  percent: 33
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-07-23)

**Core value:** 在保持 DFA 约束正确性的前提下，以少量、自然且互补的测试样本显著提高结构化生成输出的 DFA 覆盖。
**Current focus:** Phase 2 — Feasible Multi-Obligation Search (awaiting discussion and planning)

## Current Position

Phase: 2 — Feasible Multi-Obligation Search
Plan: Not created; implementation not started
Status: Ready to discuss and plan
Last activity: 2026-10-05 — Refreshed project state and exported HANDOFF.md

Progress: 2/6 phases complete (33%, including inserted Phase 01.1); 10/10 existing phase plans complete. Phase 2–5 plans are not yet created, so this is not an estimate of research effort completed.

## Performance Metrics

**Velocity:**

- Total phase plans completed: 10 (Phase 1: 3; Phase 01.1: 7)
- Average duration: -
- Total execution time: 0 hours

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| 1 | 3 | - | - |
| 01.1 | 7 | - | - |

**Recent Trend:**

- Last 5 plans: -
- Trend: Not established

| Phase 1 P1 | 15min | 2 tasks | 8 files |
| Phase 01.1 P02 | 20 minutes | 1 tasks | 11 files |
| Phase 01.1 P03 | 3min | 1 tasks | 4 files |
| Phase 01.1 P04 | 22min | 2 tasks | 1 files |
| Phase 01.1 P05 | 16min | 2 tasks | 8 files |
| Phase 01.1 P06 | 22min | 1 tasks | 2 files |
| Phase 01.1 P07 | 15min | 2 tasks | 2 files |

## Accumulated Context

### Decisions

Decisions are logged in PROJECT.md Key Decisions table. Recent decisions affecting current work:

- [Roadmap]: Use five horizontal, dependency-driven research layers: replay foundation → multi-obligation search → suite selection → LLM quality guidance → locked evaluation.
- [Roadmap]: DFA replay is the sole acceptance and coverage oracle for every claimed result.
- [Roadmap]: Do not claim globally optimal suite selection; quantify search/selection behavior only with small exact-oracle comparisons.
- [Phase 1]: Local AIAP package discovery remains independent of parent project
- [Phase 1]: Parent DFA remains sole transition authority for baseline traversal
- [Phase 01.1]: EOS is omitted from bytes vocabularies and retained as a zero-byte token action.
- [Phase 01.1]: Native verification requires regex-dfa-guide 0.3.0, an active-environment extension, and an exact byte trace.
- [Phase 01.1]: Schema-1 DFA artifacts use fixed magic, a package-independent version, and bounded full-consumption decode.
- [Phase 01.1]: Keep regex-dfa-guide version dynamic in its Python packaging metadata. — Avoid a duplicate native Python version source.
- [Phase 01.1]: Verify artifact dispatch against ARTIFACT_SCHEMA_VERSION rather than package metadata. — Artifact schema compatibility remains independent from release numbering.
- [Phase 01.1]: Classify only caller-supplied missing obligations; never rebuild the full catalog in reachability. — Preserves D-14 selected-sample obligation semantics.
- [Phase 01.1]: A witness succeeds only after aggregate-eligible canonical replay credits its exact stable target ID. — Prevents predicted search matches from being reported as coverage.
- [Phase 01.1]: Use max_expansions=None for exhaustive BFS; emit search_limit only when an explicit cap is exhausted. — Separates bounded resource exhaustion from graph infeasibility.
- [Phase 01.1]: Classify only selected-baseline missing obligations before reachable-only witness selection. — Preserves the locked D-14 through D-17 witness-selection contract.
- [Phase 01.1]: Treat the initial DFA state as the first ordered event for direct/token parity. — Direct get_state_sequence includes the initial state.
- [Phase 01.1]: Run locked Qwen parity only after provisioning and under HF_HUB_OFFLINE=1. — Preserves immutable tokenizer evidence without online fallback.
- [Experiment semantics]: Obligations are only DFA structures not visited by replay of the selected baseline samples; the structure catalog itself is not the missing-obligation set.
- [Experiment semantics]: --sample-num selects the baseline sample prefix; --obligation-num/--ob-num caps reachable targets attempted. JSONL contains attempted witness certificates only.
- [Post-phase fix d5618da]: Count matching accepting actions by indexing supplied missing meanings and scanning actions once; preserve classification and per-action count semantics.

### Pending Todos

- Discuss Phase 2, then research and plan multi-obligation grouping/search before implementation.
- Register the real_case_parity pytest marker (non-blocking warning recorded in Phase 01.1 verification).

### Blockers/Concerns

- Phase 2 must make the dominance relation, cap policy, and small-instance oracle concrete against the parent DFA API.
- Phase 4 needs a pinned model/tokenizer and a disclosed length-matched naturalness protocol.
- Phase 5 must freeze the development/held-out case-study split before final comparisons.
- Verification and CSS timing below are historical July 2026 evidence; tests were not rerun during this documentation refresh.
- PROJECT.md and generated AGENTS.md/codebase sections contain initialization-era descriptions. For current implementation, use HANDOFF.md, the completed phase summaries, and source code.

### Quick Tasks Completed

| # | Description | Date | Commit | Directory |
|---|-------------|------|--------|-----------|
| 261005-m6r | Refresh project state and export AIAP handoff | 2026-10-05 | See task summary and git log | [261005-m6r-refresh-project-state-and-export-aiap-ha](./quick/261005-m6r-refresh-project-state-and-export-aiap-ha/) |

### Roadmap Evolution

- Phase 01.1 inserted after Phase 1: Repair tokenizer vocabulary and byte-DFA contract (URGENT)

## Deferred Items

| Category | Item | Status | Deferred At |
|----------|------|--------|-------------|
| Optimization | Globally optimal minimum suite solver | v2 | 2026-07-23 |
| Quality model | LLM training or fine-tuning | v2 | 2026-07-23 |

## Session Continuity

Last session: 2026-10-05T07:58:30.858Z
Stopped at: Phase 1 and 01.1 complete; Phase 2 awaiting discussion and planning
Resume file: [HANDOFF.md](../HANDOFF.md)
Next command: `$gsd-discuss-phase 2` from the `aiap/` project root.
