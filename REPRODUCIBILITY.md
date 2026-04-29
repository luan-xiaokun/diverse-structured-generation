# Reproducibility Guide

This guide is the reviewer-facing entry point for reproducing the empirical
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
2. Lightweight generation and evaluation checks.
3. Full paper-result reproduction for the paper tables.

Paper table mapping:

| Paper result group | Reproduction entry point |
| --- | --- |
| Diversity evaluation results, Tables 1-3 | `experiments/2_diversity_evaluation.*` after `experiments/1_generation.*` |
| Efficiency evaluation results, Table 4 | `experiments/3_efficiency_evaluation.*` |
| Temperature ablation results, Tables 5-7 | `experiments/4_temperature_ablation.*` |
| Component ablation results, Table 8 | `experiments/5_component_ablation.*` |
| Coverage case study results, Table 9 | `experiments/case_study/` |

Use `experiments/README.md` for exact Linux/macOS and Windows commands.

## Environment

Start from the root `README.md`.

Recommended setup:

1. Use Docker when you want the most controlled environment for review.
2. Use `uv sync` for local setup when Docker is not convenient.
3. Download the required Hugging Face models before running full experiments.
4. Build the optional WD-shift native kernel for faster Vendi score evaluation.

Default models used by the reproduction scripts:

| Purpose | Model |
| --- | --- |
| Generation | `Qwen/Qwen2.5-1.5B-Instruct` |
| Perplexity in temperature ablation | `microsoft/Phi-4-mini-instruct` |

The model files are resolved by `transformers.from_pretrained`. For archival
reproduction, record the exact Hugging Face revisions available in your local
cache or execution environment. Model-revision pinning is tracked separately in
the OSP preparation checklist.

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

For the isolated coverage case study, follow
`experiments/case_study/README.md`. It includes its own environment and expected
coverage percentages.

## Full Reproduction

The full reproduction workflow is documented in `experiments/README.md`.

Recommended order:

1. Run sample generation: `experiments/1_generation.*`.
2. Run diversity evaluation: `experiments/2_diversity_evaluation.*`.
3. Run efficiency evaluation: `experiments/3_efficiency_evaluation.*`.
4. Run temperature ablation: `experiments/4_temperature_ablation.*`.
5. Run component ablation: `experiments/5_component_ablation.*`.
6. Run the case study under `experiments/case_study/`.

Approximate runtime on a single GPU is documented in
`experiments/README.md`. The generation and efficiency experiments are the most
expensive groups; expect roughly one hour for each of those groups on the
reference single-GPU workflow.

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

After running experiment groups, collect table-oriented CSV summaries with:

```bash
uv run python scripts/collect_results.py
```

If only one experiment group was reproduced, pass `--experiment`, for example:

```bash
uv run python scripts/collect_results.py --experiment diversity
```

The JSON result files include reproduction metadata such as timestamp, git
commit, Python version, platform, and the selected string-kernel backend.

## Nondeterminism and Tolerance

The generation scripts do not fix a seed by default. Set `SEED` when you want
same-environment repeatability:

```bash
SEED=42 bash experiments/1_generation.sh
SEED=42 bash experiments/4_temperature_ablation.sh
SEED=42 bash experiments/5_component_ablation.sh
```

PowerShell uses the same environment variable:

```powershell
$env:SEED = "42"
./experiments/1_generation.ps1
```

Even with a seed, exact bit-for-bit outputs may vary across model revisions,
PyTorch and Transformers versions, CUDA versions, GPU hardware, and sampling
implementation details. Reviewers should compare reproduced metrics within a
reasonable empirical tolerance rather than expecting byte-identical samples.

An automated tolerance checker for `results/tables/*.csv` is planned as a
separate OSP preparation task. Until that checker exists, treat the structured
JSON and CSV outputs as the primary reproduction artifacts and compare them to
the paper tables manually.

## Reviewer Checklist

For a short artifact check:

1. Build or install the environment from `README.md`.
2. Run `uv run poe test`, `uv run poe test-rust`, and `uv run poe lint`.
3. Run the README smoke test.
4. Run `uv run poe gen css-color -n 10 --stdout-only`.
5. Run the isolated case-study checks in `experiments/case_study/`.

For full reproduction:

1. Follow the setup in `README.md`.
2. Follow the command sequence in `experiments/README.md`.
3. Collect CSV summaries with `scripts/collect_results.py`.
4. Compare `results/tables/*.csv` and case-study outputs with the paper tables.
