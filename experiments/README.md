# Experiments

This directory contains the artifact reproduction scripts for the paper
results. The scripts are organized by experiment group and mapped to the paper
tables. Each experiment has a Linux/macOS shell entrypoint and a Windows
PowerShell entrypoint.

Use the root [README.md](/home/lxk/projects/diverse-dfa-gen/README.md) for
repository setup and API context. Use
[case_study/README.md](/home/lxk/projects/diverse-dfa-gen/experiments/case_study/README.md)
for the isolated Table 9 artifact.

## Before You Start

Complete the repository-level setup in the root
[README.md](/home/lxk/projects/diverse-dfa-gen/README.md) first.

Minimum checklist:

1. Install Python dependencies with `uv sync`.
2. Build the Rust DFA extension from repository root with `maturin develop --release -m regex_dfa_guide/Cargo.toml` (after activating root `.venv`).
3. Download the generation model used in the experiments.
4. Optionally build `wd_kernel` for faster diversity evaluation.

Recommended model defaults used by these scripts:

- Generation model: `Qwen/Qwen2.5-1.5B-Instruct`
- Perplexity model for temperature ablation: `microsoft/Phi-4-mini-instruct`

Shared defaults such as the grammar list and model names live in
[common.sh](/home/lxk/projects/diverse-dfa-gen/experiments/common.sh) and
[common.ps1](/home/lxk/projects/diverse-dfa-gen/experiments/common.ps1).

## Table Mapping

| Experiment | Linux / macOS | Windows | Paper tables |
|-----------|----------------|---------|--------------|
| Sample generation | `1_generation.sh` | `1_generation.ps1` | Produces generated samples used by downstream evaluations |
| Diversity evaluation | `2_diversity_evaluation.sh` | `2_diversity_evaluation.ps1` | Table 1, Table 2, Table 3 |
| Efficiency evaluation | `3_efficiency_evaluation.sh` | `3_efficiency_evaluation.ps1` | Table 4 |
| Temperature ablation | `4_temperature_ablation.sh` | `4_temperature_ablation.ps1` | Table 5, Table 6, Table 7 |
| Component ablation | `5_component_ablation.sh` | `5_component_ablation.ps1` | Table 8 |
| Case study | `case_study/` | `case_study/` | Table 9 |

Note: Table 5 to Table 7 additionally require `microsoft/Phi-4-mini-instruct`
for perplexity evaluation.

## Dependencies Between Experiments

- `1_generation.*` should run before `2_diversity_evaluation.*`, because Table
  1 to Table 3 consume the generated sample files.
- `3_efficiency_evaluation.*` is independent from the saved generation outputs;
  it measures runtime directly.
- `4_temperature_ablation.*` is self-contained for the temperature ablation
  runs and their evaluation.
- `5_component_ablation.*` is self-contained for the component ablation runs
  and their evaluation.
- `case_study/` is intentionally isolated from the repository root environment
  and has its own README and local dependency setup.

## Expected Runtime and Outputs

Approximate runtime on a single GPU:

- `1_generation.*`: about 1 hour
- `2_diversity_evaluation.*`: usually a few minutes
- `3_efficiency_evaluation.*`: about 1 hour
- `4_temperature_ablation.*`: usually a few minutes to moderate runtime,
  depending on model loading and perplexity evaluation
- `5_component_ablation.*`: about 1 hour

Outputs:

- Generated samples are written under `data/diverse/{model}/` or
  `data/baseline/{model}/`.
- Evaluation summaries are primarily printed to the console by the current
  scripts.
- Table 9 outputs are described separately in
  [case_study/README.md](/home/lxk/projects/diverse-dfa-gen/experiments/case_study/README.md).

## Recommended Reproduction Order

If you want to reproduce all experiment groups:

1. Run sample generation.
2. Run diversity evaluation.
3. Run efficiency evaluation.
4. Run temperature ablation.
5. Run component ablation.
6. Run the case study for Table 9.

If you only want one table group, you can usually run just the corresponding
script pair, except that Table 1 to Table 3 depend on the outputs from
`1_generation.*`.

## Minimal Sanity Check

If you want a lightweight artifact check instead of full paper reproduction:

1. Complete the setup steps from the root
   [README.md](/home/lxk/projects/diverse-dfa-gen/README.md).
2. Run `uv run poe test` from the repository root.
3. Run one small generation command such as `uv run poe gen css-color -n 10 --stdout-only`.
4. Run the isolated case study from
   [case_study/README.md](/home/lxk/projects/diverse-dfa-gen/experiments/case_study/README.md).

## Commands

### Linux / macOS

```bash
bash experiments/1_generation.sh
bash experiments/2_diversity_evaluation.sh
bash experiments/3_efficiency_evaluation.sh
bash experiments/4_temperature_ablation.sh
bash experiments/5_component_ablation.sh
```

### Windows PowerShell

```powershell
./experiments/1_generation.ps1
./experiments/2_diversity_evaluation.ps1
./experiments/3_efficiency_evaluation.ps1
./experiments/4_temperature_ablation.ps1
./experiments/5_component_ablation.ps1
```

## How to Save Results

The current reproduction scripts print most evaluation summaries to the
console. For paper artifact submission, it is a good idea to save logs for each
run.

Linux / macOS example:

```bash
bash experiments/2_diversity_evaluation.sh | tee diversity_eval.log
```

Windows PowerShell example:

```powershell
./experiments/2_diversity_evaluation.ps1 | Tee-Object diversity_eval.log
```

You can apply the same pattern to the other experiment scripts. Generated
sample files are written under `data/diverse/{model}/` and
`data/baseline/{model}/`.

## Per-Experiment Notes

### `1_generation.*`

- Runs diverse generation and baseline generation for all seven grammars.
- Uses `1000` samples per grammar.
- Applies `--max-tokens 54` for the `json` grammar.

### `2_diversity_evaluation.*`

- Evaluates both diverse and baseline outputs for all seven grammars.
- Used for the metrics reported in Table 1, Table 2, and Table 3.

### `3_efficiency_evaluation.*`

- Measures runtime for both diverse and baseline generation.
- Used for the throughput comparison in Table 4.

### `4_temperature_ablation.*`

- Regenerates samples at temperature `1.5`.
- Evaluates diversity metrics and perplexity with
  `microsoft/Phi-4-mini-instruct`.
- Used for Table 5, Table 6, and Table 7.

### `5_component_ablation.*`

- Runs ablations for `reward`, `penalty`, and `range_scaling`.
- The shell version uses `timeout 1800`; the PowerShell version mirrors this
  with a job timeout and also respects `TIMEOUT_SECONDS`.
- Used for Table 8.

### `case_study/`

- Independent artifact for the coverage-based case study.
- See [case_study/README.md](/home/lxk/projects/diverse-dfa-gen/experiments/case_study/README.md).
