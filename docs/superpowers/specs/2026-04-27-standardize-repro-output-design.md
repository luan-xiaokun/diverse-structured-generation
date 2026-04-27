# Standardize Reproduction Output Design

## Goal

Make the paper reproduction scripts produce stable, machine-readable outputs
under `results/`, then generate table-oriented CSV summaries from those outputs.
The first implementation phase must complete the raw JSON output path before
adding table aggregation.

## Current State

Generated samples are already structured JSON files written under `data/`.
The case study experiment also writes a dedicated summary JSON. The main
reproduction gap is in the root experiment scripts: diversity metrics and
runtime measurements are primarily printed to stdout, so reviewers must save
logs manually and extract values by hand.

The existing command flow should remain recognizable:

- `uv run poe gen ...` generates sample JSON files.
- `uv run poe eval ...` computes diversity metrics.
- `uv run poe eval-runtime ...` measures generation speed.
- `experiments/*.sh` and `experiments/*.ps1` orchestrate full experiment
  groups.

## Scope

In scope:

- Add structured JSON output to `scripts/metrics_eval.py`.
- Add structured JSON output to `scripts/eval_runtime.py`.
- Update Linux/macOS and PowerShell experiment scripts to write JSON results
  under `results/`.
- Add a table collection script that reads JSON result files and writes CSV
  summaries.
- Update experiment documentation to describe `results/` and table collection.
- Add focused tests for serialization, output writing, and CSV collection.

Out of scope:

- Moving generated sample files out of `data/`.
- Replacing the shell and PowerShell experiment entrypoints with a new runner.
- Adding expected-result baselines or tolerance-based comparisons.
- Making LLM generation exactly deterministic across machines.
- Changing the isolated `experiments/case_study` workflow.

## Directory Layout

Generated samples remain under the existing `data/` tree. Evaluation and
aggregation outputs use a new `results/` tree:

```text
results/
  diversity/
    diverse/<grammar>.json
    baseline/<grammar>.json
  runtime/
    diverse/<grammar>.json
    baseline/<grammar>.json
  temperature_ablation/
    diverse/temperature-1.5/<grammar>.json
    baseline/temperature-1.5/<grammar>.json
  component_ablation/
    default/css-color.json
    reward/css-color.json
    penalty/css-color.json
    range_scaling/css-color.json
  tables/
    diversity.csv
    runtime.csv
    temperature_ablation.csv
    component_ablation.csv
```

The first implementation phase creates all JSON files. The second phase creates
the `results/tables/` CSV files from those JSON files without rerunning
experiments.

## Metrics JSON Schema

`scripts/metrics_eval.py` gets an optional `--output` argument. It continues to
print the current human-readable summary and, when `--output` is provided,
writes a JSON file with this shape:

```json
{
  "schema_version": 1,
  "experiment": "diversity",
  "setting": "diverse",
  "grammar": "css-color",
  "model": "Qwen/Qwen2.5-1.5B-Instruct",
  "input_path": "data/diverse/Qwen_Qwen2.5-1.5B-Instruct/css-color.json",
  "parameters": {
    "d": 5,
    "s": 1,
    "n": null,
    "temperature": null,
    "top_k": null,
    "top_p": null,
    "ablation_component": null,
    "ppl": false,
    "ppl_model": "microsoft/Phi-4-mini-instruct"
  },
  "sample_count": 1000,
  "average_length": 12.34,
  "dfa": {
    "state_count": 123,
    "transition_count": 456
  },
  "metrics": {
    "state_coverage": 0.98,
    "transition_coverage": 0.91,
    "path_coverage": 0.32,
    "distinct_2gram": 0.44,
    "distinct_3gram": 0.52,
    "vendi_score": 18.7,
    "average_perplexity": null,
    "perplexity_count": 0,
    "perplexity_error_count": 0
  },
  "metadata": {
    "timestamp_utc": "2026-04-27T00:00:00Z",
    "git_commit": "0219598...",
    "python": "3.12.0",
    "platform": "Linux-...",
    "string_kernel_backend": "c"
  }
}
```

All coverage values are stored as numeric ratios in the `0.0` to `1.0` range.
Percent formatting remains a presentation concern for stdout or downstream
tables. Metadata is best effort: missing git information is recorded as `null`
instead of failing the experiment.

Perplexity failures should not abort the whole evaluation. The output records
the number of successful perplexity values and the number of errors.

## Runtime JSON Schema

`scripts/eval_runtime.py` gets an optional `--output` argument. It continues to
print the current human-readable timing summary and, when `--output` is
provided, writes a JSON file with this shape:

```json
{
  "schema_version": 1,
  "experiment": "runtime",
  "setting": "baseline",
  "grammar": "css-color",
  "model": "Qwen/Qwen2.5-1.5B-Instruct",
  "parameters": {
    "n": 2000,
    "max_tokens": 60,
    "temperature": null,
    "top_k": null,
    "top_p": null
  },
  "tokens": {
    "generated": 2031,
    "target": 2000
  },
  "timing": {
    "seconds": 123.45,
    "tokens_per_second": 16.45
  },
  "metadata": {
    "timestamp_utc": "2026-04-27T00:00:00Z",
    "git_commit": "0219598...",
    "python": "3.12.0",
    "platform": "Linux-..."
  }
}
```

## Experiment Script Behavior

`experiments/common.sh` and `experiments/common.ps1` are responsible for
constructing result paths and creating parent directories. Existing experiment
entrypoints stay the same from the user's perspective.

Examples of the CLI calls generated by the scripts:

```bash
uv run poe eval css-color --output results/diversity/diverse/css-color.json
uv run poe eval css-color --baseline --output results/diversity/baseline/css-color.json
uv run poe eval-runtime css-color --output results/runtime/diverse/css-color.json
uv run poe eval-runtime css-color --baseline --output results/runtime/baseline/css-color.json
```

Temperature ablation writes to
`results/temperature_ablation/<setting>/temperature-1.5/<grammar>.json`.
Component ablation writes to
`results/component_ablation/<component>/css-color.json`, with the unablated run
stored under `results/component_ablation/default/css-color.json`.

## Table Collection

Add `scripts/collect_results.py`. It reads the JSON files under `results/` and
writes:

```text
results/tables/diversity.csv
results/tables/runtime.csv
results/tables/temperature_ablation.csv
results/tables/component_ablation.csv
```

The collector never runs generation or evaluation. Missing input files should
produce a clear error listing the missing paths, so reviewers know which
experiment group has not been run.

The first CSV version should prioritize stable, inspectable columns over exact
paper formatting. Paper-specific rounding can be handled later if needed.

## Testing

Tests should avoid loading HuggingFace models or running real generation.
Required coverage:

- Metrics result construction returns the expected dictionary fields and numeric
  metric types.
- Metrics output writing creates parent directories and writes valid JSON.
- Runtime result construction and output writing work without model loading.
- Table collection reads small fixture JSON files and writes CSV files with the
  expected headers and representative values.

The implementation can refactor small pure helper functions out of the CLI
`main()` functions to make these tests straightforward.

## Documentation

Update `experiments/README.md` to describe:

- `results/` as the standard location for evaluation outputs.
- Which experiment script writes which JSON files.
- How to run `uv run python scripts/collect_results.py`.
- That `tee` logs are optional debugging artifacts, not the main reproduction
  output.

## Acceptance Criteria

- Running the diversity evaluation script writes JSON files under
  `results/diversity/`.
- Running the runtime evaluation script writes JSON files under
  `results/runtime/`.
- Temperature and component ablation scripts write JSON files under their
  corresponding `results/` subtrees.
- `scripts/collect_results.py` creates CSV summaries from existing JSON files.
- Existing stdout summaries remain available.
- Unit tests cover the new serialization and collection behavior without
  requiring model downloads.
