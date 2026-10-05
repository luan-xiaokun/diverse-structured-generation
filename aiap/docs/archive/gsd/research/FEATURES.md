# Feature Landscape: AIAP Multi-Obligation Witness Generation

**Domain:** A reproducible research prototype for DFA-constrained LLM witness generation and compact coverage suites  
**Researched:** 2026-07-23  
**Evidence confidence:** MEDIUM — recommendations triangulate primary papers and the checked local baseline; the target case-study corpus and model access still need to be frozen.

## Research Frame

The paper should make three separable, falsifiable claims rather than claim that a demo makes text “better”:

1. **Joint-obligation construction:** one accepted witness can cover more previously-unsatisfied DFA obligations than the current one-obligation shortest-path BFS.
2. **Quality-aware constrained generation:** an LLM-derived score can improve naturalness among *already DFA-valid* candidates without reducing independently verified validity or claimed coverage.
3. **Complementary suite selection:** under the same sample budget, selecting from ICFEM and new witnesses yields more coverage (or reaches a target coverage with fewer samples) than simple alternatives.

All three operate on the same trace-grounded obligation universe: reachable and co-reachable DFA states, byte transitions, and adjacent state-pair paths. This is consistent with established automata-constrained decoding, but the paper’s novelty must be the joint coverage accounting and empirical trade-off, not merely token masking. [Automata-based constraints for language model decoding](https://arxiv.org/abs/2407.08103) (primary source; MEDIUM confidence via web verification).

## Table Stakes Research Capabilities

Missing any item below makes results non-auditable or confounds the proposed contributions.

| Capability | Why expected | Complexity | Operational acceptance criterion |
|---|---|---:|---|
| Frozen benchmark manifest | A research result requires fixed regexes, seed samples, tokenizer/model IDs, prompts, decoding parameters, seeds and hardware/runtime. | Med | One versioned JSON/CSV manifest reproduces every reported run and links each output to its case study, method, seed and budget. |
| DFA/trace oracle | Formal validity and coverage must not depend on an LLM’s self-report. | Med | Independently replay each decoded byte string; report `accepted`, exact state/transition/path trace and covered obligation IDs. Invalid or mismatched claims are a hard failure. |
| Feasible-obligation preprocessing | An unreachable or non-co-reachable obligation makes a method look incomplete for the wrong reason. | Med | Remove and count obligations not on an initial-to-final accepting path; publish the remaining universe size by criterion and case study. |
| Multi-obligation objective model | The present `CoverageObligation` is one target only, so grouping is not yet testable. | High | Candidate carries a set/bitset of verified obligations; objective exposes weighted marginal gain, token cost and optional quality score. |
| Reproducible candidate generator | Comparison needs identical DFA, tokenizer, time/query budget and cap on generated candidates. | High | Every method consumes the same obligation universe and budget; timeout, candidate count, generated tokens and API calls are logged. |
| Explicit baselines | The new method must be distinguished from both current work and trivial grouping. | Med | Implement: (B0) existing samples only; (B1) current one-obligation BFS shortest witness per target; (B2) random feasible-obligation groups + BFS; (B3) quality-unaware multi-obligation search; (B4) greedy suite selection. |
| Budgeted suite selection | “Smaller suite” is otherwise undefined. | Med | Given candidate-to-obligation sets, output deterministic greedy marginal-gain suites for budgets `k` and target coverage `τ`; preserve tie-breaking seed/order. |
| Evaluation runner and raw artifacts | Aggregated plots alone cannot expose validity or selection bugs. | Med | Emit per-run JSONL plus aggregate tables; retain selected IDs, traces, scores, wall time and failure reason. |
| Statistical protocol | Multi-seed generative results cannot rest on one favorable run. | Med | At least 5 fixed seeds where sampling/search is stochastic; report median, IQR/95% bootstrap CI, paired effect sizes, and the number of case studies/seeds. |

## Differentiating Claims and Experiments

| Contribution slice | Testable claim | Required comparison / ablation | Primary outcome measures | Guardrail |
|---|---|---|---|---|
| **C1: obligation grouping** | Adaptive groups generate witnesses with higher verified joint gain than one-target construction at equal candidate/token/time budget. | B1; random group B2; disable group scoring; group size 1. | `new_obligations_per_witness`, cumulative state/transition/path coverage, joint-hit distribution, time/token/query cost. | Measure only replayed traces; report per-criterion values, not one hidden aggregate. |
| **C2: LLM quality guidance** | Ranking/search guidance increases naturalness of *valid, equal-coverage* witnesses. | quality-unaware C1; shortest-path-only; optionally unconstrained LLM followed by rejection (as a feasibility/efficiency baseline). Ablate scorer, reranker and beam width separately. | DFA-valid rate; coverage retained; length; frozen external-LM NLL/perplexity; blinded human preference / fluency score. | Do not call an LLM judge the sole naturalness ground truth; judge may favor its own style. |
| **C3: complementary selection** | A greedy weighted marginal-gain suite combines old and new candidates to dominate naive unions/selection at equal `k`, or meets `τ` with fewer tests. | ICFEM set only; new set only; concatenated union; random `k`; greedy coverage-only; quality-weighted greedy; small-instance ILP oracle. | coverage@k curves, minimum size for `τ` (where feasible), area under coverage–budget curve, source mix, redundancy/Jaccard overlap, selection time. | Claim approximation only for the precisely defined monotone objective; do not claim global optimality for quality/diversity variants. |
| **C4: end-to-end testing utility** | Coverage improvements transfer to a task-facing signal, not just automaton bookkeeping. | Best C1–C3 method vs B0/B1 under matched total generated strings. | task-specific acceptance/property violations or mutation/fault exposure when an executable consumer exists; otherwise explicitly label this as unavailable. | Never substitute regex acceptance for software fault detection. |

The test-suite slice is naturally a budgeted maximum-coverage / set-cover formulation: each candidate is a verified set of obligations. Greedy selection is appropriate as an interpretable baseline; its classical guarantee applies to the specified set-cover objective, not to arbitrary multi-objective additions. [Chvátal, *A Greedy Heuristic for the Set-Covering Problem*](https://doi.org/10.1287/moor.4.3.233) (primary source; MEDIUM confidence). Many-objective test generation also treats individual coverage targets as objectives, supporting the need to compare against target-at-a-time approaches rather than inventing an untested composite score. [Arcuri, *Many Independent Objective Algorithm*](https://arxiv.org/abs/1901.01541) (primary source; MEDIUM confidence).

## Evaluation Measures and Reporting Contract

### Core correctness and coverage

For obligation universe `U_c` for criterion `c ∈ {state, transition, path}` and selected suite `S`:

```text
validity_rate(S)        = accepted_by_DFA(S) / |S|
verified_coverage_c(S)  = |⋃ trace_c(x), x∈S| / |U_c|
marginal_gain(x | S)    = |trace(x) \ ⋃ trace(S)|
gain_per_cost(x)        = marginal_gain(x | S) / generated_tokens_or_time(x)
coverage@k              = verified_coverage(S_k), |S_k|≤k
min_size@τ              = min |S| s.t. verified_coverage(S)≥τ
```

Report the denominator (`|U_c|`) after feasibility filtering, absolute counts and percentages. Specify whether paths mean ordered state pairs (as in the current script) or fixed-length sequences; do not rename one as the other. Coverage thresholds must be preset (`≥1` for reachability plus sensitivity runs at `≥2` or `≥3` if repeated exercise is a claim).

### Quality and efficiency

- **Naturalness:** pre-register a blinded, randomized pairwise preference task (new method versus matched-coverage baseline) and/or 1–5 fluency/appropriateness ratings with written anchors. Use at least two annotators, report sample count, agreement (Krippendorff’s α or equivalent), ties and disagreement handling. Show the strings and regex/task context to raters; redact method identity.
- **Automatic quality:** use a fixed, disclosed external language model’s length-normalized NLL/perplexity only as a proxy, and report it alongside—not instead of—human ratings. Reference-based BERTScore is appropriate only if an actual reference text exists; it measures semantic similarity, not generic naturalness. [Zhang et al., *BERTScore*](https://arxiv.org/abs/1904.09675) (primary source; MEDIUM confidence).
- **Optional LLM judge:** keep it secondary, use a held-out evaluator distinct from the generator, disclose prompt/model/temperature, and calibrate correlation against the human subset. G-Eval itself identifies potential bias toward LLM-generated text. [Liu et al., *G-Eval*](https://arxiv.org/abs/2303.16634) (primary source; MEDIUM confidence).
- **Efficiency:** median and tail (`p95`) wall-clock per valid witness, generated tokens, expanded DFA/token states, number of LLM forward passes/API calls, timeout rate and memory. Report costs separately for grouping, construction, ranking and selection.

### Case-study protocol

1. **Pre-register the manifest:** select a diverse, fixed set of regular languages from existing ICFEM data plus held-out regexes; stratify by DFA state/transition counts, accepting density, tokenization fragmentation and domain (e.g., CSS color, email). Do not tune on the held-out set.
2. **Initial-suite conditions:** for every case study, provide the identical ICFEM sample set and calculate its feasible uncovered universe. Include at least a low-, medium- and high-initial-coverage condition if samples can be subsampled reproducibly.
3. **Matched budgets:** give every generator the same wall-clock or expanded-state budget, maximum candidate count, token cap and seed schedule. If an LLM has an API cost, report it rather than treating it as free.
4. **Generation then selection:** generate candidate pools first; replay and deduplicate them; run all selectors on the exact same pool where the selector is being compared. This separates candidate quality from selection quality.
5. **Analysis:** plot per-case coverage@k curves, paired deltas against B1/B3, confidence intervals and a case-study table. Aggregate only after showing per-case data; use paired bootstrap across case studies/seeds. Publish raw traces and failure categories.
6. **Human study:** sample witnesses stratified by case study and matched joint coverage/length; blind method labels; preregister exclusion rules. Treat the human protocol as a separate claim from DFA correctness.

## Anti-Features

| Do not build / claim | Why it weakens the AIAP paper | Use instead |
|---|---|---|
| A generic web service, IDE plugin, chat UI, or dashboard | Product scope consumes time without validating the three research hypotheses. | CLI + manifests + scripts + raw artifacts sufficient for reproducibility. |
| Fine-tuning or training an LLM | Adds data, safety, compute and attribution confounds unrelated to DFA-constrained inference. | Frozen off-the-shelf model(s), disclosed prompts and inference-time scores. |
| “Naturalness” based only on a self-judging generator or one opaque score | It is circular and vulnerable to judge/style bias. | Blinded human subset plus disclosed fixed-model proxy and agreement/calibration. |
| Exact global minimum-suite solver as the production method | It is outside the stated scope and obscures a scalable prototype. | Deterministic greedy selector; ILP/exhaustive optimum only as an offline small-instance upper bound. |
| Unconstrained LLM output accepted on parse success alone | Parse success does not demonstrate correct DFA acceptance or the claimed byte-level trace. | Independent DFA replay and trace-to-obligation check for every output. |
| One aggregate “coverage” number | It can hide regressions in transition/path coverage and mismatched denominators. | Separate state, byte-transition and ordered-pair path metrics plus raw counts. |
| Curated anecdotal examples as principal evidence | Easy regexes and hand-picked natural strings create selection bias. | Fixed stratified case-study manifest, seeds, budgets, and per-case results. |
| A new weighted mega-objective with no sensitivity study | Gains may be an artifact of arbitrary weights. | Preset weights, coverage-only ablation, and sensitivity analysis over quality/cost weights. |

## Feature Dependencies

```text
Frozen tokenizer + DFA adapter
    → independent trace oracle + feasibility filtering
        → canonical obligation universe / bitset representation
            → C1 multi-obligation candidate construction
                → C2 quality-aware ranking over valid candidates
                → shared replayed candidate pool
                    → C3 greedy complementary suite selection
                        → coverage@k / min_size@τ evaluation
                            → blinded naturalness study + end-to-end case analysis
```

## MVP Recommendation

Prioritize:

1. Canonical replay oracle, feasible obligation sets, manifest and B0/B1 baselines.
2. C1 grouped construction with a quality-unaware ablation and fully trace-verified coverage@k experiment.
3. C3 deterministic greedy selector over shared candidate pools, then C2 quality guidance plus a blinded pilot study.

Defer an end-to-end fault-detection claim until a case study offers an executable system/property oracle; otherwise describe the paper accurately as **DFA coverage and structured-witness quality** research.

## Sources

- [Automata-based constraints for language model decoding](https://arxiv.org/abs/2407.08103) — primary paper on finite-automaton constrained LM decoding (MEDIUM).
- [A Greedy Heuristic for the Set-Covering Problem](https://doi.org/10.1287/moor.4.3.233) — primary greedy set-cover analysis (MEDIUM).
- [Many Independent Objective Algorithm for Test Suite Generation](https://arxiv.org/abs/1901.01541) — primary many-objective test-generation framing (MEDIUM).
- [BERTScore: Evaluating Text Generation with BERT](https://arxiv.org/abs/1904.09675) — primary automatic text-evaluation metric (MEDIUM).
- [G-Eval: NLG Evaluation Using GPT-4 with Better Human Alignment](https://arxiv.org/abs/2303.16634) — primary LLM-judge method and bias caveat (MEDIUM).

## Research Gaps to Resolve During Planning

- The final held-out case-study split and executable downstream fault/property oracle are not specified yet.
- “Naturalness” must be defined per data domain: CSS/email-like values may need appropriateness/readability anchors rather than generic prose fluency.
- The exact weighted multi-criterion objective needs a theorem/claim audit: report a greedy approximation ratio only when its mathematical assumptions hold.
