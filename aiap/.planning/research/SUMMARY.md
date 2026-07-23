# Project Research Summary

**Project:** AIAP Multi-Obligation Witness Generation  
**Domain:** Reproducible research prototype for DFA-constrained structured generation and complementary test-suite construction  
**Researched:** 2026-07-23  
**Confidence:** MEDIUM-HIGH

## Executive Summary

This project should be built as a trace-first research prototype, not as an unconstrained text generator. It extends the current single-obligation BFS baseline with direct traversal of the existing DFA, where every token action is expanded into its byte-level DFA trace. The principal algorithmic contribution is a bounded, dominance-aware label search over `(DFA state, covered-obligation set)`, paired with feasibility-aware target grouping. Every reported witness must be independently replayed through the DFA before it is retained or credited with coverage.

The recommended hybrid method preserves that symbolic boundary: an LLM may score or rank only DFA-legal prefixes/candidates, using constrained beam search or reranking, but it never supplies validity or coverage evidence. Candidate traces from both the ICFEM 2025 corpus and the new generator then feed a deterministic greedy weighted set-cover/maximum-coverage selector. The paper should make three separable claims: verified joint-obligation gain, quality improvement among matched valid candidates, and complementary coverage-versus-suite-size improvement. It must not claim global-optimal suite selection, universal naturalness, or downstream fault detection without a separately measured executable oracle.

The main risks are measurement unsoundness at the token/byte/EOS boundary, unsound state-only pruning, infeasible obligation bundles, and experimental leakage. Mitigate them with one canonical replay-and-coverage contract; frozen coverage universes and manifests; explicit label dominance plus caps and a small-instance oracle; equal candidate/compute budgets for all baselines; pinned model/tokenizer revisions; and per-case, multi-seed artifact-backed reporting.

## Key Findings

### Recommended Stack

Use the parent project's Python 3.12+ environment and pinned DFA implementation as the correctness substrate. Keep the hot search loop direct (`collections.deque`/`heapq`) rather than introducing a generic graph layer: it must retain per-token, byte-inside-token traces and project-specific dominance metadata. Store coverage incidence as NumPy bitsets/sparse representations; use `pytest` for trace and invariant tests. Add OR-Tools CP-SAT only as a time-limited exact oracle on small candidate pools, not as the production selector. See [STACK.md](STACK.md).

**Core technologies:**

- `regex_dfa_guide` / `diverse_guide`: authoritative legal-token transitions, acceptance, and byte traces — never recreate a second DFA semantics.
- Python stdlib search + `numpy`: transparent bounded label search, deterministic queues/ties, and compact obligation masks.
- `pytest`: golden DFA fixtures, replay properties, and regression parity with `cov_test.py`.
- Pinned `torch` and `transformers`: optional batched LLM likelihood/beam adapter; pin model and tokenizer to immutable revisions.
- OR-Tools CP-SAT: small-pool diagnostic optimum/feasibility oracle; report solver status and time limit.

The search recommendation is a bounded label-setting algorithm. A label retains DFA state, covered-obligation mask, symbolic cost, prefix/parent pointer, and optional naturalness bound. At the same state, discard a label only when another has a superset mask at no worse declared symbolic cost and retained score; record all cap-based drops. Reverse distance-to-acceptance is a useful pruning lower bound. This is an empirical heuristic with an oracle-checked approximation profile, **not** a global-optimality method. [NetworkX shortest paths](https://networkx.org/documentation/stable/reference/algorithms/shortest_paths/index.html), [OR-Tools CP-SAT](https://developers.google.com/optimization/cp/cp_solver).

### Expected Features and Testable Contribution Claims

The project needs a versioned benchmark manifest, feasibility-filtered obligation universe, canonical replay validator, shared candidate artifacts, explicit baselines, deterministic suite selection, and an evaluation runner before research claims are credible. See [FEATURES.md](FEATURES.md).

**Must have (table stakes):**

- A frozen `CoverageUniverse` with stable state, byte-transition, and ordered state-pair IDs; all metrics use the same replayed denominator.
- Token IDs → exact bytes → DFA trace → actual obligation IDs as the only acceptance and coverage oracle, including explicit EOS handling.
- Reachability/co-reachability filtering and recorded joint-feasibility evidence for target groups.
- Reproducible manifests and JSONL artifacts: input/model hashes, prompts, seeds, budgets, traces, scores, decisions, timings, and failures.
- Equal-budget baselines: existing samples (B0), one-obligation BFS (B1), random feasible groups (B2), quality-unaware multi-obligation search (B3), and greedy selection (B4).

**Should have (the contribution slices):**

- **C1 — grouping/search:** higher *replayed* new-obligation gain per witness than B1 and B2 at matched token/time/candidate budgets.
- **C2 — LLM guidance:** better preregistered quality on valid, coverage- and length-matched witnesses than C1 without the LLM; report validity and coverage retention, an external length-normalized proxy, and blinded human evaluation rather than relying on an LLM judge alone.
- **C3 — complementary selection:** better coverage@k, area under the coverage–budget curve, or lower size at target coverage than ICFEM-only, AIAP-only, union, random, and coverage-only alternatives on equivalent validated pools.
- **C4 — optional utility:** downstream property/fault/mutation evidence only where an executable consumer and oracle exist; otherwise call the result DFA coverage and witness quality.

**Defer / do not build:** a service or IDE UI, LLM fine-tuning, unconstrained generate-then-repair workflows, one opaque aggregate coverage number, and exact global suite minimization. Greedy's guarantee is only for its precisely declared monotone coverage/cost objective. [Chvátal's set-cover analysis](https://doi.org/10.1287/moor.4.3.233).

### Architecture Approach

Replace the exploratory script with a small Python package plus thin CLI. `specs` freezes case/config identity; a `dfa_adapter` owns legal token arcs and replay; `coverage` owns the stable catalog; `grouping`, `search`, and `scoring` propose legal candidates; `validation` alone creates accepted/annotated candidates; `selection` consumes the persisted candidate × obligation matrix; and `evaluation` reads immutable artifacts rather than mutable process state. See [ARCHITECTURE.md](ARCHITECTURE.md).

**Major components:**

1. **DFA adapter, coverage, and validation** — canonical byte trace, acceptance, stable obligation incidence, and fail-closed replay.
2. **Feasibility grouping and dominance search** — create target groups with evidence; explore DFA-safe labels without state-only pruning.
3. **Scoring and constrained generation** — score only legal prefixes/completions; never alter legal arcs or post-edit a witness.
4. **Candidate pool and suite selection** — combine verified old/new traces, use deterministic marginal-gain decisions and preserve tie-break provenance.
5. **Experiment runner and artifacts** — run `generate`, `select`, `evaluate`, and `replay` into immutable per-case/method/config/seed directories.

### Critical Pitfalls and Non-Negotiable Safeguards

1. **Token-level actions can hide byte-level transitions.** Precompute full action traces/bitsets, treat EOS as typed zero-byte behavior, and reject any incremental-versus-fresh-replay mismatch.
2. **State-only `visited` pruning is unsound for joint coverage.** Use coverage-aware labels and explicit dominance; log dominance and beam pruning, then compare small instances against exhaustive/CP-SAT oracles.
3. **Locally reachable obligations may not be jointly coverable on an accepted start-to-final trace.** Filter to live obligations and test group feasibility; publish infeasibility categories rather than treating them as generator misses.
4. **Changing universes or denominators manufactures coverage gains.** Freeze/identify the universe per DFA/tokenizer/normalization/threshold/path definition and report state, transition, and path results separately.
5. **Naturalness can leak validity or be confounded by length.** The LLM has no validity authority; revalidate after every model-influenced operation, preregister scoring, and use blinded length-matched human evaluation plus a disclosed proxy.
6. **Unequal tuning/budgets and drifting artifacts invalidate comparison.** Develop separately from held-out cases, pin revisions and seeds, retain append-only manifests, compare per case/seed, and give every baseline the same DFA, tokenizer, candidate, and compute budget.
7. **Coverage-only selection is not fault detection or global optimization.** Name the exact objective, use deterministic ties, compare random/cost-aware/small-exact selectors, and make downstream claims only when measured.
8. **Generated witnesses and remote model code are untrusted.** Do not shell-interpolate witnesses; bound any execution; prefer `trust_remote_code=False`; pin/audit any unavoidable remote revision.

## Implications for Roadmap

### Phase 1: Trace Semantics, Measurement Contract, and Research Harness

**Rationale:** All later claims depend on a single correct byte-level interpretation. The present one-file demo has a known token/byte/EOS boundary and coupling risk; postponing this work makes every ablation suspect.

**Delivers:** Typed `CaseSpec`, `CoverageUniverse`, `TraceCertificate`, `Candidate`, immutable manifest/artifact schema, DFA adapter, fresh replay validator, feasibility filtering, baseline coverage parity tests, safe input/runner policy, and a frozen B0/B1 reference interface.

**Testable exit criteria:** Golden fixtures cover mid-token obligation hits, UTF-8, initial state, EOS, dead/unreachable/no-final obligations; fresh replay equals incremental trace annotation; parent metric parity is demonstrated; every result includes the universe ID and denominators.

**Avoids:** byte/token accounting errors, mutable coverage denominators, prototype coupling, unsafe witness handling, and artifact drift.

### Phase 2: Feasible Multi-Obligation Grouping and Dominance Search

**Rationale:** This is the core symbolic contribution and must be established without LLM confounds. A correct one-target baseline and verifier from Phase 1 make its incremental gain measurable.

**Delivers:** Live/co-reachable obligation catalog; target-group planner with feasibility evidence; bounded product-label search; symbolic reverse-distance pruning; deterministic cap/tie policy; group-size/frontier sensitivity configuration; and a tiny-DFA exhaustive/CP-SAT comparison oracle.

**Testable contribution claim:** At equal budget, C1 improves verified joint marginal coverage versus B1 and random feasible groups, while reporting infeasible/timeout/cap-pruned rates and per-criterion coverage.

**Avoids:** state-only pruning, impossible bundles, and hidden exponential behavior. It makes no optimality claim.

### Phase 3: Candidate-Pool Selection and Complementary-Suite Baselines

**Rationale:** Suite complementarity should be measured before LLM guidance so a coverage improvement can be attributed to grouping or selection rather than model preference.

**Delivers:** Persisted old/new validated candidate pools, incidence matrix, deterministic greedy weighted marginal-gain selector, coverage@k and minimum-size@target curves, stable ties, plus ICFEM-only/AIAP-only/union/random/cost-aware and small-exact oracle comparisons.

**Testable contribution claim:** C3 attains better specified coverage-versus-cardinality/cost trade-offs from equivalent pools, with source mix, redundancy, marginal gains, and solver gaps reported.

**Avoids:** conflating selection with candidate generation, coverage-only overclaims, and unjustified global-optimum assertions.

### Phase 4: DFA-Masked LLM Scoring and Naturalness Protocol

**Rationale:** LLM work is only meaningful after legality, coverage, and suite selection have independent baselines. Introduce a model-free beam first, then LLM likelihood/reranking behind an adapter.

**Delivers:** DFA-masked beam/logits adapter, prompt/model revision provenance, score formula and caching, score/no-score/beam ablations, development-versus-held-out configuration policy, and a preregistered length-matched naturalness study with a disclosed automatic proxy.

**Testable contribution claim:** C2 improves quality among accepted, equal-coverage candidates without reducing replayed validity or verified coverage; inconclusive quality results remain valid negative findings.

**Avoids:** generate-then-repair, validity leakage, proxy-only naturalness claims, and post-hoc parameter tuning. [Transformers generation API](https://huggingface.co/docs/transformers/main_classes/text_generation), [PyTorch reproducibility](https://docs.pytorch.org/docs/stable/notes/randomness).

### Phase 5: Locked Case Studies, Ablations, and Reproducible Paper Evidence

**Rationale:** The contribution is an empirical comparison; all methods and selectors must run under a common finalized protocol before claims can be made.

**Delivers:** Stratified development/held-out case manifest, fixed seed protocol, equal resource budgets, method × ablation × case × seed execution, raw artifact release, clean-environment replay command, per-case paired analysis, uncertainty intervals, scalability and failure tables, and optional downstream utility result where a valid oracle exists.

**Testable exit criteria:** Each aggregate cell links to manifest/input/result hashes; deterministic smoke artifacts byte-compare across reruns; all reported results include per-criterion counts, costs, failures, case-level deltas, and uncertainty. Conclusions state scope and heterogeneity rather than pooling away negative cases.

**Avoids:** evaluation leakage, model/tokenizer drift, favorable-case timing, and overgeneralized claims.

### Phase Ordering Rationale

- The dependency chain is **canonical replay → frozen obligation universe → B1 parity → feasible grouped search → shared validated pool → selector comparison → LLM quality guidance → locked evaluation**. Reversing it would make observed gains impossible to attribute.
- Search and selection use the same trace-certified incidence matrix but should remain separate phases: first establish whether joint candidates are useful, then whether the selector uses a fixed pool well.
- LLM guidance is a quality layer on legal candidates, not a replacement for the direct DFA method. This keeps formal guarantees invariant across model ablations.
- Evaluation is an implementation deliverable, not a final write-up task: manifests, deterministic artifacts, baseline budgets, and holdout rules start in Phase 1 and become locked before Phase 5.

### Research Flags

Phases likely needing deeper research during planning:

- **Phase 2:** Project-specific dominance relation, group-feasibility formulation, caps, and oracle construction need a concrete proof/experiment design against the actual DFA APIs.
- **Phase 4:** Constrained-generation hook behavior for the chosen pinned model/tokenizer, score normalization, and the domain-specific naturalness rubric need targeted API and human-study research.
- **Phase 5:** Case-study stratification, holdout policy, statistical analysis, and any downstream utility oracle need validation against available datasets and execution environments.

Phases with standard patterns (can minimize research-phase):

- **Phase 1:** Typed records, immutable JSONL manifests, replay/golden/property tests, and safe runner boundaries are established engineering patterns; repository integration still needs inspection.
- **Phase 3:** Deterministic greedy set cover and small CP-SAT baselines are well-documented; the project-specific work is incidence/cost semantics, not the solver pattern.

## Confidence Assessment

| Area | Confidence | Notes |
|------|------------|-------|
| Stack | MEDIUM-HIGH | Direct DFA approach and tool roles are grounded in the repository plus official NetworkX, Transformers, OR-Tools, Hugging Face, and PyTorch documentation; caps and score weights remain empirical. |
| Features | MEDIUM | Contribution/ablation structure is well-motivated by primary literature, but final case studies, model access, and naturalness protocol are not frozen. |
| Architecture | HIGH | Boundaries follow the inspected local DFA, metrics, and `cov_test.py`; external architecture evidence is secondary. |
| Pitfalls | MEDIUM-HIGH | Byte/EOS, pruning, and accounting risks derive directly from the local prototype; evaluation/suite-selection guidance is supported by primary or peer-reviewed work. |

**Overall confidence:** MEDIUM-HIGH. The project direction is strong and bounded, but its research claims depend on implementation validation and a locked evaluation design rather than theoretical optimality.

### Gaps to Address

- **Case-study protocol:** Freeze a diverse development/held-out regex family split, initial-sample conditions, and whether an executable downstream oracle exists. If no oracle exists, remove fault-detection claims.
- **Obligation semantics:** Decide and version path length/taxonomy, repeated-coverage thresholds, normalization, and how overlapping state/transition/path obligations are reported without double counting.
- **Search approximation:** Define the dominance policy and caps precisely; quantify loss on small instances rather than implying a general approximation guarantee.
- **Naturalness construct:** Define domain-specific appropriateness/fluency anchors; identify independent annotators and an external proxy. LLM likelihood alone is insufficient.
- **Cost and environment:** Choose the pinned model(s), hardware availability, maximum budgets, and whether remote code can be disabled; document all deviations from deterministic behavior.

## Sources

### Primary / official sources

- [NetworkX shortest-path algorithms](https://networkx.org/documentation/stable/reference/algorithms/shortest_paths/index.html) — unweighted BFS and non-negative Dijkstra roles.
- [Hugging Face Transformers generation API](https://huggingface.co/docs/transformers/main_classes/text_generation) — constrained generation hooks and beam configuration.
- [Hugging Face Hub revisions](https://huggingface.co/docs/huggingface_hub/en/guides/cli) — immutable model/tokenizer identity.
- [Google OR-Tools CP-SAT solver](https://developers.google.com/optimization/cp/cp_solver) — small-pool binary selection oracle.
- [PyTorch reproducibility notes](https://docs.pytorch.org/docs/stable/notes/randomness) and [deterministic algorithms](https://docs.pytorch.org/docs/main/generated/torch.use_deterministic_algorithms.html) — reproducibility controls and limitations.
- [Chvátal, *A Greedy Heuristic for the Set-Covering Problem*](https://doi.org/10.1287/moor.4.3.233) — scope of greedy set-cover guarantees.
- [Cognetta & Okazaki, *Tokenization as Finite-State Transduction*](https://aclanthology.org/2025.cl-4.2/) — tokenization-aware formal constraints.

### Secondary / contextual sources

- [Automata-based constraints for language model decoding](https://arxiv.org/abs/2407.08103) — constrained decoding context.
- [Arcuri, *Many Independent Objective Algorithm*](https://arxiv.org/abs/1901.01541) — many-objective test-generation framing.
- [Lin et al., greedy-based test-suite reduction](https://doi.org/10.1016/j.scico.2017.05.004) — cost and fault-detection caveats.
- [BERTScore](https://arxiv.org/abs/1904.09675) and [G-Eval](https://arxiv.org/abs/2303.16634) — automatic evaluation use and limits.
- [ReproEvalCard](https://aclanthology.org/2026.acl-short.22), [Decoding Decoded](https://aclanthology.org/2025.coling-main.668), and [Illusions of the Gold Standard](https://aclanthology.org/2026.acl-long.635) — transparent evaluation and decoding reporting.
- Local evidence: [PROJECT.md](../PROJECT.md), [STACK.md](STACK.md), [FEATURES.md](FEATURES.md), [ARCHITECTURE.md](ARCHITECTURE.md), [PITFALLS.md](PITFALLS.md), and `aiap/cov_test.py`.

---
*Research completed: 2026-07-23*  
*Ready for roadmap: yes*
