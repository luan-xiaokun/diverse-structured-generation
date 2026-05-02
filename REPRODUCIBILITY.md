# Reproducibility Guide

This guide is the reviewer-facing entry point for the DiverseGuide artifact and
for reproducing the empirical
results reported in the ICFEM 2025 paper:

> Automata-Based Steering of Large Language Models for Diverse Structured
> Generation

It links the repository setup, experiment scripts, generated artifacts, and
known sources of nondeterminism. The detailed command reference remains in
`experiments/README.md` so that experiment scripts and documentation stay close
to each other.

## Scope

The repository supports three levels of reproduction:

1. Environment and implementation checks that do not load a language model.
2. OSP primary reproduction for the manuscript's main artifact claims.
3. Optional supplementary reproduction for the case study and extended ICFEM
   analyses.

Result-group map:

| Role | Result group | Entry point | Required for OSP primary reproduction |
| --- | --- | --- | --- |
| Primary OSP reproduction | Sample generation | `experiments/1_generation.*` | Yes |
| Primary OSP reproduction | Diversity and DFA coverage | `experiments/2_diversity_evaluation.*` | Yes |
| Primary OSP reproduction | Runtime efficiency | `experiments/3_efficiency_evaluation.*` | Yes |
| Optional automated impact check | Coverage case study | `experiments/case_study/` | No |
| Optional extended ICFEM analysis | Temperature ablation | `experiments/4_temperature_ablation.*` | No |
| Optional extended ICFEM analysis | Component ablation | `experiments/5_component_ablation.*` | No |

The primary OSP reproduction focuses on the CSV summaries in
`artifact-results/v0.2.0/primary/`. Optional extended ICFEM summaries are kept in
`artifact-results/v0.2.0/optional/`.

Use `experiments/README.md` for exact Linux/macOS and Windows commands.

## Environment

Start from the root `README.md`.

Recommended setup:

1. Use Docker when you want the most controlled environment for review.
2. Use `uv sync` for local setup when Docker is not convenient.
3. Download the required Hugging Face models before running full experiments.
4. Build the optional WD-shift native kernel for faster Vendi score evaluation.

Default models used by the reproduction scripts:

| Purpose | Model | Recommended Hugging Face revision for OSP reproduction |
| --- | --- | --- |
| Generation | `Qwen/Qwen2.5-1.5B-Instruct` | `989aa7980e4cf806f80c7fef2b1adb7bc71aa306` |
| Perplexity in temperature ablation | `microsoft/Phi-4-mini-instruct` | `cfbefacb99257ffa30c83adab238a50856ac3083` |

The scripts currently pass model ids to `transformers.from_pretrained` without
an explicit `revision` argument. The revisions above are the local Hugging Face
cache snapshots recorded while preparing this OSP artifact. They should be used
as the reference model revisions for archival reproduction notes. If you
reproduce with newer model snapshots, record the exact revisions with the
resulting `results/` artifacts.

## Quick Checks

These checks are intended to verify that the artifact is installed and wired
correctly without reproducing every paper table.

From the repository root:

```bash
uv run poe test
uv run poe test-rust
uv run poe lint
```

Run the smoke test from the root `README.md` to confirm that the Python package
and Rust DFA extension work together.

For a lightweight model-backed generation check:

```bash
uv run poe gen css-color -n 10 --stdout-only
```

The optional coverage case study is documented in
`experiments/case_study/README.md`. It includes its own environment, pinned
package snapshots, and expected coverage percentages.

## Full Reproduction

The full command reference is documented in `experiments/README.md`.

Recommended OSP primary reproduction order:

1. Run sample generation: `experiments/1_generation.*`.
2. Run diversity evaluation: `experiments/2_diversity_evaluation.*`.
3. Run efficiency evaluation: `experiments/3_efficiency_evaluation.*`.
4. Collect primary CSV summaries:
   - `uv run python scripts/collect_results.py --experiment diversity`
   - `uv run python scripts/collect_results.py --experiment runtime`
5. Compare `results/tables/diversity.csv` and `results/tables/runtime.csv`
   with `artifact-results/v0.2.0/primary/`.

Optional automated impact check:

1. Run the case study under `experiments/case_study/`.
2. Inspect the generated `experiments/case_study/case_study_summary.json`.

Optional extended ICFEM analyses:

1. Run temperature ablation: `experiments/4_temperature_ablation.*`.
2. Run component ablation: `experiments/5_component_ablation.*`.
3. Collect optional CSV summaries:
   - `uv run python scripts/collect_results.py --experiment temperature_ablation`
   - `uv run python scripts/collect_results.py --experiment component_ablation`
4. Compare the optional CSVs under `results/tables/` with
   `artifact-results/v0.2.0/optional/`.

The default efficiency-evaluation baseline is the internal regex-only
masking backend (`--baseline-backend internal`). The optional Outlines backend
is useful for external diagnostic comparisons, but it is not required for the
reported efficiency table and should be written to separate output files.

Approximate runtime on a single GPU is documented in
`experiments/README.md`. The generation and efficiency experiments are the most
expensive primary groups; expect roughly one hour for each of those groups on
the reference single-GPU workflow.

## Output Artifacts

Generated samples are written under:

```text
data/diverse/{model}/
data/baseline/{model}/
```

Structured evaluation results are written under:

```text
results/
  diversity/
  runtime/
  temperature_ablation/
  component_ablation/
  tables/
```

Tracked reference CSV summaries are kept under:

```text
artifact-results/v0.2.0/
  primary/
    diversity.csv
    runtime.csv
  optional/
    temperature_ablation.csv
    component_ablation.csv
```

After running experiment groups, collect table-oriented CSV summaries with:

```bash
uv run python scripts/collect_results.py
```

If only one experiment group was reproduced, pass `--experiment`, for example:

```bash
uv run python scripts/collect_results.py --experiment diversity
```

The JSON result files include reproduction metadata such as timestamp, git
commit, Python version, platform, and where applicable the selected
string-kernel backend.

## Nondeterminism and Tolerance

Generation and runtime scripts do not fix a seed by default. Set `SEED` when
you want same-environment repeatability:

```bash
SEED=42 bash experiments/1_generation.sh
SEED=42 bash experiments/3_efficiency_evaluation.sh
SEED=42 bash experiments/4_temperature_ablation.sh
SEED=42 bash experiments/5_component_ablation.sh
```

PowerShell uses the same environment variable:

```powershell
$env:SEED = "42"
./experiments/1_generation.ps1
./experiments/3_efficiency_evaluation.ps1
```

Even with a seed, exact bit-for-bit outputs may vary across model revisions,
PyTorch and Transformers versions, CUDA versions, GPU hardware, and sampling
implementation details. Reviewers should compare reproduced metrics within a
reasonable empirical tolerance rather than expecting byte-identical samples.

Treat the structured JSON and CSV outputs as the primary reproduction artifacts.
When comparing with the paper tables, record the command line, seed, model
revision, hardware, and software environment used for the reproduction. Runtime
throughput and sampling-based diversity metrics should be compared within a
reasonable empirical tolerance rather than as exact equality checks.

## Reviewer Checklist

For a short artifact check:

1. Build or install the environment from `README.md`.
2. Run `uv run poe test`, `uv run poe test-rust`, and `uv run poe lint`.
3. Run the README smoke test.
4. Run `uv run poe gen css-color -n 10 --stdout-only`.

For OSP primary reproduction:

1. Follow the setup in `README.md`.
2. Run `experiments/1_generation.*`.
3. Run `experiments/2_diversity_evaluation.*`.
4. Run `experiments/3_efficiency_evaluation.*`.
5. Collect primary CSV summaries:
   - `uv run python scripts/collect_results.py --experiment diversity`
   - `uv run python scripts/collect_results.py --experiment runtime`
6. Compare `results/tables/diversity.csv` and `results/tables/runtime.csv`
   with `artifact-results/v0.2.0/primary/`.

For optional supplementary checks:

1. Run the case study under `experiments/case_study/`.
2. Run `experiments/4_temperature_ablation.*` and
   `experiments/5_component_ablation.*` when extended ICFEM analyses are needed.
3. Collect optional CSV summaries:
   - `uv run python scripts/collect_results.py --experiment temperature_ablation`
   - `uv run python scripts/collect_results.py --experiment component_ablation`
4. Compare optional CSV summaries with `artifact-results/v0.2.0/optional/`.
