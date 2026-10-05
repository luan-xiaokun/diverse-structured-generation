# Architecture Patterns

**Domain:** Multi-obligation DFA-constrained witness-generation research prototype  
**Researched:** 2026-07-23  
**Confidence:** HIGH for local integration and boundaries; LOW for external ecosystem evidence because the configured research provider had no API credential.

## Recommended Architecture

Replace the exploratory script with a small Python package and thin CLI. DFA replay is the sole correctness authority: grouping, search, LLM scoring and selection are heuristics, while every retained candidate is independently replayed through the target DFA before it affects a metric.

```text
ICFEM input + samples ─► CaseSpec / RunConfig ─► DFA adapter + Coverage analyzer
                                                     │
      manifest / hashes ◄────────────────────────────┤
                                                     ▼
Obligation catalog ─► Group planner ─► DFA-safe generator ◄─ LLM scorer
                                             │  (scores only)
                                             ▼
                                     replay validator
                                             ▼
                          candidate × obligation incidence matrix
                                      │                 │
                                      ▼                 ▼
                               suite selector      evaluator
                                      └──────► immutable run artifacts
```

### Component Boundaries

| Component | Responsibility | Input → output | Must not do |
|---|---|---|---|
| `specs` | Typed, serializable `CaseSpec`, obligation/candidate IDs, configuration, manifest. | JSON/YAML → canonical records. | Import model libraries or mutate counters. |
| `dfa_adapter` | Wrap parent `DiverseGuideDFA`: legal token arcs, byte traces, accepting states and replay; normalize EOS. | Regex + tokenizer → DFA view; token IDs → trace. | Score language or decide coverage. |
| `coverage` | Enumerate and count state, byte-transition, and state-pair obligations; produce stable IDs and incidence sets. | DFA traces → catalog/counters. | Generate candidates. |
| `grouping` | Form and score feasible groups from unmet obligations, retaining feasibility evidence and tie-break keys. | Catalog/reachability → `TargetGroup`s. | Assert a group was achieved. |
| `search` | Maintain DFA-safe frontier records: prefix IDs, state, byte trace, met-obligation bitset, cost and parent pointer. | Group + legal arcs → frontier. | Freely generate text then repair it. |
| `scoring` | Pluggable BFS cost, LLM log-probability/naturalness, novelty and length scores with provenance. | Legal prefixes → score vector. | Change allowed token arcs or validity. |
| `generation` | Expand only legal DFA token IDs, apply beam/pruning policy, complete at finals. | Frontier + scorer → raw candidates. | Write metrics/results. |
| `validation` | Replay raw candidate, check final acceptance, reconstruct byte transitions and actual covered IDs; reject mismatches/duplicates. | Raw → validated/rejection record. | Trust planned coverage. |
| `selection` | Select from ICFEM and new candidates against an explicit coverage/budget objective with deterministic ties. | Incidence matrix → selected IDs/trace. | Recompute traces or call the LLM. |
| `evaluation` | Compare methods/ablations under equal case, budget and seed; report coverage, cost, naturalness and downstream effects. | Selected artifacts → summaries. | Generate candidates. |
| `artifacts` / CLI | Create isolated run directories and `generate`, `select`, `evaluate`, `replay` commands. | Config → immutable artifacts. | Hide default parameters. |

### Guarantees vs. Quality Objectives

| Category | Property | Enforcement |
|---|---|---|
| Hard guarantee | Retained witness is accepted by the specified DFA. | Validator replays it; fail closed. |
| Hard guarantee | Reported coverage is derived from replayed byte traces, never intended routes. | Coverage accepts validator output only. |
| Hard guarantee | Inputs/configuration/provenance are identifiable. | Checksummed manifest and immutable run directory. |
| Quality objective | A witness covers several unmet obligations. | Search/grouping objective; report intended versus achieved. |
| Quality objective | Text is natural/useful. | Separate scorer/evaluator; never implies validity. |
| Quality objective | Suite is small but covers much. | Selector curve; no global-optimum claim. |
| Quality objective | Improvements transfer to downstream case studies. | Existing case-study harness and reports. |

## Data Flow

1. `CaseLoader` reads an ICFEM-style case (`regex`, samples), resolves tokenizer/model identifiers and writes hashes before search. AIAP never changes the parent datasets.
2. `dfa_adapter` creates the parent guide/DFA. `coverage` replays the existing samples and emits a stable catalog such as `transition:12:97:13`, rather than a process-local index.
3. `grouping` uses reachability/co-reachability summaries to order target groups. A group is a search target, not coverage evidence.
4. `generation` expands only `allowed_tokens(state)`. It records all byte steps within multi-byte tokens, then uses the LLM only to rank legal prefixes/completions.
5. `validation` replays each completion from scratch, checks final acceptance, derives actual coverage, records token/text/trace hashes, and rejects planned-versus-actual disagreement. Only validated candidates enter the pool.
6. Build a candidate-by-obligation binary incidence matrix covering ICFEM samples, AIAP candidates and their union. Select with deterministic greedy marginal gain, retaining every choice and tie-break. This makes the set-cover-style test-suite reduction baseline inspectable. [Test-suite minimization study](https://doi.org/10.1145/1108792.1108802)
7. `evaluation` reads selected IDs and the manifest—not mutable memory—to compute coverage, runtime/model cost, naturalness, suite-size curves and downstream reports.

## Evaluation Artifact Schema

Use an immutable directory per run: `aiap/runs/<case>/<method>/<config-hash>/<seed>/`. Generated results are ignored; schemas, configs and small golden fixtures are tracked.

```text
run/
  manifest.json        # identity and reproducibility contract
  input_snapshot.json  # regex + input IDs/hashes
  obligations.json     # stable IDs and initial coverage counts
  groups.jsonl         # target groups, feasibility and decision trace
  candidates.jsonl     # raw/validated/rejected candidates
  incidence.npz        # sparse candidate × obligation matrix
  selection.json       # objective, selected IDs, marginal gains
  metrics.json         # final metrics and curves
  downstream/          # copied/linked ICFEM case-study reports
  logs.jsonl           # structured events
```

| Artifact | Minimum fields |
|---|---|
| `manifest.json` | Schema version, git revision/dirty status, Python/package lock IDs, device, case ID, regex/dataset hashes, tokenizer/model revision, prompt hash, seed(s), grouping/beam/selection parameters and command. |
| `obligations.json` | Catalog version, threshold, stable ID/type, DFA graph fields, reachability, original count and source suite. |
| Candidate row | ID, parent/group, rank, token IDs, text hash, DFA trace, final state, acceptance, intended/actual IDs, scorer versions/vector, elapsed time and rejection reason. |
| `selection.json` | Pool-manifest hash, universe, algorithm/version, budget/cost policy, ordered IDs, marginal gain, remaining uncovered IDs and tie-break values. |
| `metrics.json` | Method/ablation labels, validated/rejected counts, all three DFA coverage metrics and curves, suite size, time/model cost, naturalness evaluator version and downstream-report hashes. |

Pin all defaults and repeated seeds in the manifest. Current experiment guidance likewise emphasizes explicit artifacts, tracking, documentation and repeated runs; this source is supplementary, not a correctness basis. [Google ML experiment guidance](https://developers.google.com/machine-learning/managing-ml-projects/experiments)

## Patterns to Follow

### Validate at the Symbolic Boundary

Every method returns `RawCandidate`; only validation may create `ValidatedCandidate`.

```python
raw = generator.propose(target_group, dfa, scorer, config)
validated = [validator.replay(item, dfa, catalog) for item in raw]
pool = [item for item in validated if item.accepted]
matrix = coverage.incidence_matrix(pool, catalog)
```

The parent DFA API provides legal-token queries and byte-transition sequences, so this verification is mechanical and model-independent.

### Strategy Interfaces, Shared Evaluation Contract

Use protocols for `GroupPlanner`, `CandidateGenerator`, `CandidateScorer` and `SuiteSelector`; all consume immutable records and emit the same artifacts. This permits BFS-only, multi-obligation, beam/no-beam, no-LLM, and selection ablations without changing data semantics.

### Pure Core, Imperative Shell

Graph algorithms, replay-to-incidence conversion and selection should be deterministic pure functions. Keep model calls, I/O and timestamps in the CLI/orchestrator. This follows the parent metrics module's pure-function style and makes hand-built DFA unit tests practical.

## Anti-Patterns to Avoid

### Monolithic "Generate and Print" Script

It conflates proposal, validation, scoring and metrics, making comparisons invalid. Compose modules through persisted boundary objects instead.

### Counting Intended Obligations

Targets may be met mid-token and beam branches may differ. Recompute coverage from the completed replay trace and report intended/achieved overlap.

### Letting an LLM Bypass the DFA

Generate-then-validate wastes budget and weakens the formal claim. Enumerate legal arcs first; use the LLM only for ranking legal candidates.

### Comparing Different Candidate Pools

Selectors can appear superior merely because they saw more/better candidates. Persist pool hashes; compare selectors over identical validated pools, separately from end-to-end results.

## Dependency-Driven Build Order

1. **Foundation:** typed specs, DFA adapter, replay validator and hand-built golden DFAs. Test acceptance, EOS, mid-token bytes and unreachable/no-final obligations.
2. **Coverage/artifacts:** stable catalog, baseline counts, snapshots/manifests and replay CLI. Confirm aggregate results equal the parent state/transition/path metric functions on ICFEM inputs.
3. **Reference method:** reimplement current single-obligation deterministic BFS behind these contracts; persist candidates, matrix and metrics as the regression anchor.
4. **Multi-obligation search:** add feasibility-aware grouping and coverage-gain graph search while retaining BFS. Validate every return by replay.
5. **Constrained beam + LLM scoring:** add model-free beam first, then LLM/naturalness scorers. Test that score changes never alter legal arcs or validator outcomes; hold search budgets equal.
6. **Suite selection:** deterministic greedy marginal coverage plus ICFEM-only, AIAP-only, union, random and size-budget baselines; preserve the matrix and decision trace.
7. **Evaluation/reproduction:** execute method × ablation × case × seed, import downstream reports, aggregate tables and perform a clean-environment replay check.

## Scalability Considerations

| Concern | Prototype | Larger response |
|---|---|---|
| DFA frontier | Parent pointers and exact visited-state keys. | Configured beam bounds, token-arc/suffix-reachability memoization and recorded truncation. |
| Obligation tracking | Sets / integer bitsets. | Stable catalog with packed bitsets or sparse matrices. |
| LLM calls | Batched calls with score provenance. | Cache on model/prompt/prefix IDs; separate model-call and graph-node budgets. |
| Selection | Deterministic greedy scan. | Sparse incremental gains/heap; retain exact reference on small cases. |
| Repetition | Directory per seed. | Aggregate only compatible manifest/input hashes. |

## Sources

- Local evidence (HIGH): `aiap/cov_test.py`, `src/diverse_guide/evaluation/metrics.py`, the DFA Python interface, and `experiments/case_study/README.md`, inspected 2026-07-23.
- [Google ML experiment guidance](https://developers.google.com/machine-learning/managing-ml-projects/experiments) — LOW confidence from configured-provider classification; artifact/repetition guidance only.
- [A concept analysis inspired greedy algorithm for test suite minimization](https://doi.org/10.1145/1108792.1108802) — LOW confidence from configured-provider classification; motivates a transparent greedy baseline only.
