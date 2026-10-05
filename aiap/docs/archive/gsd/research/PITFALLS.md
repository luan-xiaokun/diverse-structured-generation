# Domain Pitfalls

**Domain:** Multi-obligation DFA witness generation with LLM-guided naturalness and complementary-suite selection  
**Researched:** 2026-07-23  
**Confidence:** MEDIUM — project-specific risks were inspected in `aiap/cov_test.py`; external algorithm/evaluation claims use primary or peer-reviewed sources.

## Critical Pitfalls

### Pitfall 1: Token actions are mistaken for byte-level coverage

**What goes wrong:** A token can consume several UTF-8 bytes and therefore traverse several DFA arcs. A search that credits only its token-level source/destination misses an obligation traversed in the token interior, or credits a different byte transition. EOS is especially dangerous: in the current demo it is a token-level final-state self-loop but has no byte transition sequence.

**Why it happens:** The search action space is tokenizer vocabulary, whereas the stated coverage universe is DFA states, labelled byte transitions, and state-pair paths. `cov_test.py` already has a special EOS exception and manually uses `transitions[1:]`, which signals this boundary is real rather than theoretical.

**Consequences:** Claimed multi-obligation gain can be false even when decoded output is accepted; comparisons may optimize a token-level proxy while reporting byte coverage.

**Prevention:** Define one canonical trace contract: token IDs → exact decoded bytes (with explicit normalization policy) → DFA byte trace → covered obligation IDs. Precompute each allowed token action's full byte trace and coverage bitset; handle EOS as a separately typed zero-byte action. Require the final decoded bytes to replay to a final DFA state and reproduce every claimed obligation. Add fixtures for an obligation in the middle of a multi-byte token, a non-ASCII/UTF-8 token, initial-state coverage, and EOS.

**Detection:** Fail a property test whenever incremental action annotations differ from a fresh byte-trace replay; report `claimed − replayed` and `replayed − claimed` obligation sets for every experiment run. A nonzero difference is a release blocker.

**Must be addressed:** **Phase 1 — DFA/coverage semantics and verifier**, before multi-obligation optimization or any LLM integration.

### Pitfall 2: Unsound search pruning destroys jointly useful prefixes

**What goes wrong:** Reusing the demo's `visited_states: set[state]` retains one prefix per DFA state. For multi-obligation search, two prefixes ending at the same state can have different covered-obligation sets, costs, likelihoods, lengths, or lexical quality; deleting one can delete the only path to the best feasible witness.

**Why it happens:** State-only deduplication is correct for unweighted shortest reachability but not for a multi-objective resource-constrained search. The proposed objective adds history-dependent coverage and LLM scores.

**Consequences:** Negative results become artifacts of pruning; coverage plateaus unpredictably; an apparent LLM benefit can be caused by accidentally retaining a different prefix rather than by naturalness guidance.

**Prevention:** Make the search label include `(dfa_state, coverage_bitset or sparse set, budget)` plus objective values. Use explicit dominance: only discard label A if another label reaches the same DFA state with a superset of obligations at no greater hard cost and no worse retained score under the declared policy. Bound growth with a reproducible beam/label cap, then log all cap-triggered drops and run an exact small-instance oracle/ILP or exhaustive search to quantify approximation loss.

**Detection:** On small DFAs, compare coverage and objective with the oracle; track number of nondominated labels, dominance-pruned labels, beam-pruned labels, and failures to complete to a final state. Large sensitivity to beam width or tie order is a warning.

**Must be addressed:** **Phase 2 — multi-obligation search**, with regression tests introduced in Phase 1.

### Pitfall 3: Reachable obligations are not necessarily co-coverable in an accepted witness

**What goes wrong:** A grouping step selects individually uncovered state/transition/path obligations that cannot occur together in a single accepted trace because of branch choice, ordering, length budgets, or dead-end regions. The existing `construct_one_witness` already notes that a reachable obligation can lack an accepting completion.

**Why it happens:** The coverage matrix is built from local graph membership, but the feasibility constraint is a start-to-final path over the whole DFA. State-pair path coverage is also less discriminating than labelled transition coverage and can conflate distinct byte choices.

**Consequences:** Search spends budget on impossible bundles, deceptively low joint-coverage rates are attributed to the algorithm, and suite selection receives candidates whose advertised sets are infeasible.

**Prevention:** Compute forward reachable and backward co-reachable states first; reject non-live obligations. Represent a candidate group as a feasibility query over the product of DFA state and group-progress automaton, not merely a union of local obligations. Preserve the exact obligation taxonomy and report duplicates/subsumption (a transition implies endpoint states and its state pair) instead of summing all three categories as independent wins.

**Detection:** Before search, publish counts for total, live, individually feasible, and jointly feasible obligations/groups. Keep a witness-plus-replayed-trace certificate for each success and a categorized infeasibility certificate for each rejection.

**Must be addressed:** **Phase 2 — obligation grouping and feasibility**, then enforced again in **Phase 4 — suite selection**.

### Pitfall 4: Coverage accounting double-counts or changes denominator between methods

**What goes wrong:** Coverage is reported with an inconsistent universe: all DFA arcs versus only live arcs, byte transitions versus state pairs, threshold one versus a higher threshold, or one count per input versus one count per occurrence. Adding a witness may be scored against a denominator that was recomputed after filtering or after a tokenizer/DFA change.

**Why it happens:** `get_coverage_obligations` and the parent metric functions are separate code paths; the demo independently creates state, transition, and path sets. Overlapping criteria make aggregate “total coverage” particularly easy to inflate.

**Consequences:** Absolute and relative gains cannot be reproduced or compared fairly; one method can win by changing accounting rather than increasing exercised behaviour.

**Prevention:** Version a frozen `CoverageUniverse` artifact per `(regex, DFA construction, tokenizer, normalization, threshold, path definition)`. Use one trace-replay implementation for candidate annotation and all final metrics. Report each metric separately, numerator/denominator, per-case-study values, incremental union gain, witness count, byte/token length, and wall-clock budget; do not average incompatible percentages.

**Detection:** Metamorphic tests: permuting input order must not change union coverage; duplicating an input must not alter threshold-1 coverage; re-annotating stored traces must match the saved bitsets. CI should reject result files missing universe IDs and denominators.

**Must be addressed:** **Phase 1 — measurement contract** and **Phase 5 — experiments/reporting**.

### Pitfall 5: Naturalness optimisation leaks validity or measures the wrong construct

**What goes wrong:** An LLM scorer or reranker is permitted to edit output after constrained traversal, or ranks only the decoded string without rechecking its byte trace. Conversely, perplexity/log-probability can reward short frequent strings rather than grammaticality, task appropriateness, or human-perceived naturalness.

**Why it happens:** “LLM-guided” is underspecified. A shortest accepted witness is not automatically natural, but an unconstrained LLM completion has no DFA guarantee. Naturalness has no single objective ground truth.

**Consequences:** A method can report fluent but invalid strings, or claim a naturalness gain that is really a length/model-family effect. LLM-as-judge self-preference and prompt-order bias can further confound results.

**Prevention:** Keep acceptance and trace coverage as hard gates after *every* model-influenced operation; use the LLM only to score/prioritize already-valid prefixes/actions or generate proposals that are projected/rejected by the DFA. Pre-register the score direction, model and prompt. Evaluate naturalness with a blinded, randomized human pairwise study on matched valid witnesses, clear domain-specific rubric, inter-annotator agreement, and a separate automatic proxy; neither proxy may be the sole claim.

**Detection:** Log rejection rate after every model step and require zero invalid outputs in released results. Audit score correlations with token/byte length and template frequency; report judge agreement, ties, and per-case outcomes. If the naturalness effect vanishes under length matching or an independent judge, describe it as inconclusive.

**Must be addressed:** **Phase 3 — LLM-guided search and naturalness evaluation**, with validity gates inherited from Phase 1.

### Pitfall 6: Tuning on the final cases produces objective leakage and unfair comparators

**What goes wrong:** Group size, beam width, scoring prompts, decoding temperature, suite budget, coverage threshold, or selected examples are adjusted after observing the same case-study results used in the paper. Baselines may receive fewer candidates, a different tokenizer/DFA, or no equivalent tuning budget.

**Why it happens:** The prototype has few case studies and several coupled knobs; coverage, naturalness, and suite size tempt iterative hand-tuning. LLM benchmarks are additionally vulnerable to training/evaluation contamination.

**Consequences:** Reported improvement estimates are optimistic and cannot establish that the contribution is multi-obligation search, LLM guidance, or selection rather than privileged information.

**Prevention:** Split regex/case studies into development and sealed evaluation partitions (or use nested cross-validation/leave-one-regex-family-out where data are scarce). Lock algorithms and hyperparameters before opening held-out metrics. Give every comparator the same DFA, tokenizer, coverage universe, candidate-generation budget, compute budget, random seeds, and suite-size budget. Include: ICFEM single-obligation BFS baseline, multi-obligation without LLM, LLM scoring without grouping, and selection ablations. State explicitly where baseline artifacts come from and whether they are regenerated.

**Detection:** Maintain an append-only experiment manifest with timestamped config hashes and a holdout-access log. Compare outcomes across all seeds/cases rather than only a pooled maximum; flag post-hoc config changes and any baseline that lacks an equal-budget run.

**Must be addressed:** **Phase 3 — model/parameter development** and locked for **Phase 5 — comparative evaluation**.

### Pitfall 7: Suite selection is evaluated as coverage-only set cover

**What goes wrong:** The selector greedily maximizes uncovered DFA obligations and calls the result a better test suite. It may prefer syntactically odd, near-duplicate, very long, expensive, or low-fault-revealing witnesses; a coverage-preserving reduced suite can have different downstream test effectiveness.

**Why it happens:** Set cover has a clear objective and greedy has a logarithmic approximation guarantee, but that guarantee applies only to the declared set-cover cost/universe—not to naturalness, generation/execution cost, or fault detection.

**Consequences:** “Complementary” means only coverage-complementary; conclusions about testing utility overreach and may be unstable under tie-breaking.

**Prevention:** Start with deterministic greedy weighted set cover and disclose its actual objective. Define candidate costs before selection (at least cardinality; optionally byte length/generation cost) and preserve exact covered bitsets. Evaluate a coverage-only selector alongside random, length/cost-aware greedy, the unreduced union, and an exact solver on small instances. Report suite size, cost, per-metric coverage, marginal gains, overlap/Jaccard, runtime, and—where an executable SUT/mutants exist—fault/mutation detection separately. Do not claim fault detection from DFA coverage alone.

**Detection:** Tie permutations/seeds should yield a distribution, not one lucky suite. Compare achieved size to exact optimum on small cases and inspect candidate-set duplicates/subsumption. A coverage tie with materially different cost or quality signals the need for the declared secondary criterion.

**Must be addressed:** **Phase 4 — complementary suite selection** and **Phase 5 — utility evaluation**.

### Pitfall 8: Results are irreproducible due to tokenizer/model and decoding drift

**What goes wrong:** A model alias, tokenizer revision, remote-code implementation, vocabulary, special-token map, CUDA/kernel setting, or nondeterministic tie-break changes the allowed token graph and output trace. The same `AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B", trust_remote_code=True)` call need not identify immutable executable code.

**Why it happens:** The method depends on both formal artifacts and stochastic/model-serving artifacts; neither is presently captured by the single-file demo.

**Consequences:** A rerun can change validity, coverage, naturalness, or selected suite, making the paper’s central comparisons non-reproducible.

**Prevention:** Pin and record regex/DFA serialization hash, tokenizer/model repository revision and file hashes, Transformers/Python/Rust/package versions, device, deterministic flags, seeds, prompts, decoding/beam/score parameters, candidate order, tie-breaker, and raw token IDs plus decoded bytes and trace certificates. Package an offline lockfile/container and release a one-command evaluator that verifies stored results without contacting an LLM service.

**Detection:** CI reruns a fixed smoke matrix twice and byte-compares deterministic artifacts; stochastic runs must reproduce their seed-indexed distribution within declared tolerance. Require every aggregate table cell to link to input/config/result hashes.

**Must be addressed:** **Phase 1 — experiment artifact schema**, maintained in **Phases 3–5**.

### Pitfall 9: Generated text and remote model code create an unnecessary safety boundary breach

**What goes wrong:** Generated witnesses are fed into shells, regex engines with pathological inputs, file paths, HTML/JSON renderers, or test harnesses without treating them as untrusted data; a later experiment evaluates arbitrary generated text as code. Separately, enabling `trust_remote_code=True` executes repository-supplied tokenizer/model code during setup.

**Why it happens:** The project calls witnesses “tests,” which invites execution, while the current demo already opts into remote code. Formal acceptance says nothing about safety in downstream consumers.

**Consequences:** Prompt injection through corpus text, code execution/supply-chain compromise, resource exhaustion, or unsafe test-harness side effects can compromise experiment hosts and result integrity.

**Prevention:** Treat all generated/loaded samples as data: never interpolate them into a shell; use argument vectors, bounded input length/time/memory, sandboxed non-networked execution for any SUT, output escaping, and a corpus allowlist. Pin and audit model/tokenizer revisions; prefer `trust_remote_code=False`, or document a reviewed commit hash and isolated environment when it is indispensable. Do not execute generated programs or regexes as part of “naturalness” evaluation.

**Detection:** Static checks reject shell-string construction from witness text; integration tests inject quotes, control characters, Unicode, and oversized strings; CI records whether remote code is enabled and runs it in a disposable environment. Monitor timeouts/memory spikes per witness.

**Must be addressed:** **Phase 1 — safe runner and dependency policy**, before any automated case-study execution.

## Moderate Pitfalls

### Pitfall 10: Search cost is hidden by small or favorable DFAs

**What goes wrong:** Full-vocabulary token expansion times byte replay and multi-label state tracking grows sharply with DFA size, obligation density, and beam width; only two easy case studies are timed.

**Prevention:** Benchmark construction, search, LLM scoring, verification, and selection separately across DFA sizes/densities; use precomputed action annotations and bounded deterministic caps. Report failed/timeout cases and peak labels/memory, not just successful mean runtime.

**Detection:** Plot runtime and memory versus DFA states, live arcs, vocabulary actions, obligation count, and group size. A sudden cap/timeout rate is a result, not data to discard.

**Must be addressed:** **Phase 2 — search implementation** and **Phase 5 — scalability experiments**.

### Pitfall 11: Aggregate claims hide case-study and statistical instability

**What goes wrong:** A pooled percentage or a single best seed masks that only one regex family benefits, or that random LLM/beam tie-breaking dominates the effect.

**Prevention:** Treat case study/regex family as the unit of comparison; publish per-case paired deltas, all seed runs, effect sizes and uncertainty intervals. Predefine primary endpoints and distinguish exploratory analyses. Keep failure/infeasibility counts in tables.

**Detection:** Inspect leave-one-case-out conclusions and seed distributions; if signs vary, report heterogeneity rather than a universal improvement claim.

**Must be addressed:** **Phase 5 — experimental analysis and paper artifacts**.

## Minor Pitfalls

### Pitfall 12: Prototype coupling makes verification and ablation brittle

**What goes wrong:** Data loading, coverage enumeration, search, decoding, LLM scoring, selection, and printing remain coupled in `cov_test.py`; an ablation accidentally changes shared measurement logic.

**Prevention:** Extract immutable domain objects (`CoverageUniverse`, `TraceCertificate`, `Candidate`) and pure modules for verifier, search, scorer adapter, selector, and experiment runner. Unit-test each module against stored trace fixtures.

**Detection:** An ablation must be a config change that reuses the same verifier and coverage artifact; duplicated metric code or manual CSV edits are review blockers.

**Must be addressed:** **Phase 1 — foundations**, before research features accumulate.

## Phase-Specific Warnings

| Phase topic | Likely pitfall | Mitigation |
|-------------|---------------|------------|
| 1. DFA/measurement foundations | Byte/token/EOS mismatch; mutable denominators; unsafe runner | Canonical byte-trace verifier, frozen coverage universe, trace fixtures, dependency/sandbox policy |
| 2. Multi-obligation search | State-only pruning; infeasible bundles; hidden exponential growth | Dominance-labelled search, co-reachability/group feasibility checks, small exact oracle, resource telemetry |
| 3. LLM-guided naturalness | Invalid post-edits; proxy/length confound; tuning leakage | DFA hard gate after each operation, preregistered scorer, blinded length-matched evaluation, development/holdout split |
| 4. Suite selection | Coverage-only overclaim; arbitrary greedy tie effects | Deterministic weighted objective, candidate bitsets/costs, random/cost-aware/exact-small comparators, distribution over ties |
| 5. Case studies and paper | Comparator asymmetry; unreproducible model drift; pooled-only statistics | Equal budgets and artifacts, pinned revisions/seeds/traces, per-case paired outcomes and uncertainty |

## Sources

- [Cognetta & Okazaki, *Tokenization as Finite-State Transduction* (Computational Linguistics 2025)](https://aclanthology.org/2025.cl-4.2/) — **MEDIUM** (peer-reviewed primary source; supports tokenization-aware formal constraints).
- [Chvátal, *A Greedy Heuristic for the Set-Covering Problem* (1979)](https://doi.org/10.1287/moor.4.3.233) — **HIGH** (primary proof of the greedy set-cover bound).
- [Lin et al., *Empirically evaluating Greedy-based test suite reduction methods* (2017)](https://doi.org/10.1016/j.scico.2017.05.004) — **MEDIUM** (peer-reviewed empirical evidence that cost and fault-detection criteria matter alongside reduction).
- [Singh et al., *Evaluation data contamination in LLMs* (2024)](https://arxiv.org/abs/2411.03923) — **MEDIUM** (research preprint; supports contamination/tuning caution).
- [Pattnayak & Bhatia, *ReproEvalCard* (ACL 2026)](https://aclanthology.org/2026.acl-short.22/) — **MEDIUM** (peer-reviewed reporting evidence for seeds/traces/configuration).
- [Garces Arias et al., *Decoding Decoded* (COLING 2025)](https://aclanthology.org/2025.coling-main.668/) — **MEDIUM** (peer-reviewed evidence that decoding settings affect generation outcomes).
- [Mei et al., *Illusions of the Gold Standard* (ACL 2026)](https://aclanthology.org/2026.acl-long.635/) — **MEDIUM** (peer-reviewed evidence for transparent human-evaluation protocols).
