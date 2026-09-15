# Refinement Effectiveness Evidence Plan

Date: 2026-06-12

This document fixes the experimental plan for evaluating DiverseGuide+
refinements after the valid-only global-counter repair. It is intended to keep
the refinement evidence task stable and prevent later drift in goals, metrics,
or baselines.

## Objective

Find and justify a single robust DiverseGuide+ parameter setting
`gamma / eta / estimator` that improves over the DiverseGuide baseline on most
cells in the final `3 models x 6 grammars` matrix, while preserving validity and
generation efficiency.

The evidence should support two mechanism-level claims:

1. With `eta = 0`, the harmonic path-frequency estimator can improve diversity
   or accelerate coverage growth compared with the min estimator.
2. With the min estimator fixed, positive `eta` can improve completion,
   validity rate, or valid samples per second on long or rejection-prone
   generation tasks, without causing a broad diversity regression.

## Fixed Semantics

All experiments in this plan must use the repaired counter semantics:

- Global path counters are updated only for valid accepted samples.
- Invalid or rejected samples must not update global counters.
- Local counters remain per generation call and are reset before each call.
- DG baseline and DG+ must use the same valid-only update semantics.

If any script cannot guarantee these semantics, fix or instrument it before
running the corresponding experiment.

## Method Definitions

Use these names consistently:

- `DG baseline`: DiverseGuide with `path_frequency_estimator=min`,
  `completion_bias_eta=0`, and the baseline gamma selected for comparison.
- `DG+ harmonic-only`: `path_frequency_estimator=harmonic`,
  `completion_bias_eta=0`.
- `DG+ eta-only`: `path_frequency_estimator=min`,
  `completion_bias_eta>0`.
- `DG+ selected`: the final single global parameter setting selected from the
  ablations and tuning stage.

Do not call origin/master `old DG` in this task. If historical implementation
comparisons are needed, use `true-old DG`, `new DG`, and `DG+` as defined in the
engineering evidence plan.

## Common Generation Settings

Default common settings unless a phase explicitly overrides them:

- Sample target: `n = 500` accepted samples for full ablations and confirmation.
- Smoke/pilot target: `n = 50` generation attempts per cell when estimating
  rejection rate; accepted-sample pilots may be used only when throughput is the
  primary quantity.
- Models:
  - `Qwen/Qwen2.5-0.5B-Instruct`
  - `Qwen/Qwen2.5-1.5B-Instruct`
  - `microsoft/Phi-4-mini-instruct`
- Grammars:
  - `email`
  - `css-color`
  - `json`
  - `no-bomb`
  - `semver`
  - `api-route`
- Max-token defaults for later refinement runs:
  - non-json: `max_tokens = 24`
  - json: `max_tokens = 72`
- Generation prompts, regexes, tokenizer handling, CUDA checks, and sampling
  settings should match the main matrix unless the phase explicitly changes a
  variable.

## Required Metrics

Every generated artifact should be evaluated with:

- Validity:
  - accepted sample count
  - rejected sample count
  - rejection rate
  - validity rate
  - accepted samples per second
  - attempted batches
  - max-token exhaustion rate if instrumented
- Length:
  - average generated token length
  - average generated character length
  - length distribution summary
- Diversity:
  - state coverage
  - transition coverage
  - path coverage
  - distinct-2
  - distinct-3
  - Vendi score
- Trends:
  - coverage@50
  - coverage@100
  - coverage@250
  - coverage@500
  - coverage trend AUC
  - raw trend data for plotting

If max-token exhaustion is not currently separated from generic rejection,
instrument generation or post-hoc validation before interpreting eta as a
completion-efficiency improvement.

## Output Layout

Use isolated paths so this task does not overwrite main-matrix artifacts.

Proposed generation root:

```text
data/extension/refinement_effectiveness_500/
```

Proposed pilot root:

```text
data/extension/refinement_budget_pilot_n50/
```

Proposed result root:

```text
results/extension/refinement_effectiveness_500/
results/extension/refinement_budget_pilot_n50/
```

Proposed log root:

```text
logs/extension/refinement_effectiveness_500/
logs/extension/refinement_budget_pilot_n50/
```

## Phase 0: JSON Budget Pilot

Purpose: determine whether the high JSON rejection rate is mainly caused by an
insufficient max-token budget. This must be done before interpreting eta
effects, because eta and max-token budget can both change completion behavior.

Scope:

- Model: `Qwen/Qwen2.5-1.5B-Instruct`
- Grammar: `json`
- Attempts: `n = 50` generation attempts per cell
- Candidate max-token budgets:
  - `54` previous main-matrix JSON budget
  - `72`
  - `96`
  - `128`
- Methods:
  - `DG baseline`: `estimator=min`, `eta=0`
  - one representative DG+ candidate, initially
    `estimator=harmonic`, `eta=0.1`, `gamma=0.3`, unless this conflicts with
    the current repaired implementation defaults

Primary questions:

1. Does increasing JSON `max_tokens` materially reduce rejection rate?
2. Does it improve accepted samples per second, or does the extra generation
   budget slow down accepted throughput?
3. Does it change average length enough to affect diversity comparisons?
4. Is the rejection mostly max-token exhaustion, or another validity failure?

Decision rule used for the pilot:

- If a larger JSON budget reduces rejection rate substantially without hurting
  accepted samples per second too much, use that budget for JSON in later
  refinement experiments.
- Do not start full ablations until the user has reviewed the pilot summary.

Decision:

- Use `json_max_tokens = 72` for later JSON refinement cells.
- Keep `default_max_tokens = 24` for non-JSON grammars.
- The pilot showed JSON rejections were max-token exhaustion under the current
  permissive `name` field regex; increasing from 54 to 72 materially reduced
  rejection without the throughput cost observed at larger budgets.

Minimum pilot outputs:

```text
results/extension/refinement_budget_pilot_n50/json_budget_pilot_metrics.json
results/extension/refinement_budget_pilot_n50/json_budget_pilot_metrics.csv
results/extension/refinement_budget_pilot_n50/json_budget_pilot_summary.md
```

## Phase 1: Harmonic vs Min Estimator Ablation

Purpose: test whether harmonic path-frequency estimation improves diversity or
coverage growth when `eta=0`.

Scope:

- Model: `Qwen/Qwen2.5-1.5B-Instruct`
- Grammars: all 6 grammars
- Default screening samples: `n = 100`
- Eta: `0`
- Estimators:
  - `min`
  - `harmonic`
- Gamma grid:
  - `0.1`
  - `0.2`
  - `0.3`

Primary evidence:

- coverage trend AUC
- coverage@50/100/250/500
- final state / transition / path coverage
- distinct-2 / distinct-3
- Vendi

Guardrails:

- validity rate should not drop materially against DG baseline
- accepted samples per second should not drop materially against DG baseline
- average length should be reported to detect length-driven diversity artifacts

Expected conclusion shape:

- Harmonic does not need to win every final metric.
- Harmonic evidence is acceptable if it improves early coverage growth or trend
  AUC on several grammars without broad negative side effects.

Stop gate:

- Summarize harmonic vs min results before selecting the estimator for final
  tuning.

Revised execution strategy:

- Do not run the full `6 grammars x 2 estimators x 3 gamma x 500 samples`
  matrix upfront; generation is too slow for email/json-like cases.
- Use already-completed `n=500` representative cells for `email` and
  `css-color` as the high-confidence slice.
- Use `n=100` screening for the remaining grammars to identify where harmonic
  beats min and whether the effect depends on gamma.
- After a promising gamma is identified, run only the selected gamma for
  `min` vs `harmonic` at `n=500` on the grammars needed for confirmation.

Runner:

```text
experiments/run_refinement_phase1_harmonic_min.sh
experiments/evaluate_refinement_phase1_harmonic_min.sh
```

Default artifact roots:

```text
data/extension/refinement_phase1_harmonic_min_screen_n100/
logs/extension/refinement_phase1_harmonic_min_screen_n100/
results/extension/refinement_phase1_harmonic_min_screen_n100/
```

Completed representative slice:

```text
data/extension/refinement_phase1_harmonic_min_500/
results/extension/refinement_phase1_harmonic_min_500/evaluation_email_css/
```

## Phase 2: Eta Effectiveness Ablation

Purpose: test whether `eta > 0` improves completion, validity, or valid
generation efficiency under the min estimator.

Scope:

- Estimator: `min`
- Gamma: fixed from Phase 1 if possible; otherwise start with `0.2`
- Eta grid:
  - `0`
  - `0.05`
  - `0.10`
  - `0.15`
  - `0.20`
- Primary model: `Qwen/Qwen2.5-1.5B-Instruct`
- Confirmation model if needed: `microsoft/Phi-4-mini-instruct`
- Primary grammars:
  - `json`
  - `no-bomb`
  - `email`
  - `api-route`
- Sanity grammars:
  - `css-color`
  - `semver`

Primary evidence:

- rejection rate
- validity rate
- accepted samples per second
- max-token exhaustion rate if available
- average generated length

Guardrails:

- final coverage and Vendi should not suffer a broad regression
- eta should not simply shorten outputs in a way that hides diversity loss

Stop gate:

- Summarize eta results and decide whether eta is a final parameter, a
  grammar-specific diagnostic, or not supported by evidence.

## Phase 3: Joint Parameter Selection

Purpose: select one global DG+ parameter setting without per-grammar
cherry-picking.

Tuning set:

- `Qwen/Qwen2.5-1.5B-Instruct x 6 grammars`

Candidate grid:

```text
gamma in {0.1, 0.2, 0.3}
eta in {0, 0.05, 0.10, 0.15}
estimator in {min, harmonic}
```

Selection rule:

- Prefer the setting with the best aggregate rank over:
  - path coverage
  - transition coverage
  - Vendi
  - distinct-2 / distinct-3
  - coverage trend AUC
- Enforce guardrails:
  - validity rate should not be materially worse than DG baseline
  - accepted samples per second should not be materially worse than DG baseline
  - no systematic harmful length collapse

The selected setting must be a single global setting, not per grammar, unless
the paper explicitly frames it as per-task tuning.

Stop gate:

- Present the selected setting and its tuning-set evidence to the user before
  running the final 3-model confirmation matrix.

## Phase 4: Final Confirmation Matrix

Purpose: evaluate the selected DG+ setting against DG baseline without further
tuning.

Scope:

- Models: 3 models
- Grammars: 6 grammars
- Samples: `n = 500`
- Methods:
  - DG baseline
  - DG+ selected

Report:

- per-cell metrics table
- win counts by metric
- coverage trend plots or raw trend data for plotting
- validity and accepted samples per second
- failure cases and negative cells

Main claim criterion:

- DG+ selected should improve over DG baseline in most cells on diversity or
  coverage metrics, while keeping validity and valid samples per second within
  acceptable guardrails.

## Phase 5: Paper Packaging

Organize evidence into:

1. Mechanism ablation: harmonic vs min at `eta=0`.
2. Completion/validity ablation: eta grid under min estimator.
3. Final matrix: selected DG+ vs DG baseline on `3 models x 6 grammars`.
4. Engineering context: build time and processor latency evidence remains
   separate from refinement effectiveness evidence.

## Immediate Next Step

Summarize the completed `email` and `css-color` representative slice, then run
only the reduced `n=100` screening for the remaining grammars if the completed
slice is insufficient to choose a promising gamma. Use paper-facing gamma
values:

```text
gamma in {0.1, 0.2, 0.3}
```

Do not use `1/6` in new refinement ablations. Existing historical artifacts may
still record `0.166666...`, but those should be treated as historical DG
baseline runs rather than the paper-facing refinement grid.

Run `Qwen/Qwen2.5-1.5B-Instruct x 6 grammars x 2 estimators x 3 gamma values`,
then summarize:

- coverage trend AUC and coverage@50/100/250/500
- final state / transition / path coverage
- distinct-2 / distinct-3 / Vendi
- validity, rejection rate, accepted samples per second, and average length

Do not launch Phase 2 or later full ablations until the Phase 1 summary has
been reviewed.
