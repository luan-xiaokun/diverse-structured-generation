# DiverseGuide v0.2.0 Reference Results

This directory contains author-generated CSV summaries for the DiverseGuide
v0.2.0 OSP artifact snapshot.

The ignored `results/` directory is used for local regenerated outputs. The
tracked files here are reference summaries that reviewers can compare against
newly regenerated `results/tables/*.csv` files.

## Layout

| Directory | Role | Files |
| --- | --- | --- |
| `primary/` | OSP primary reproduction targets | `diversity.csv`, `runtime.csv` |
| `optional/` | Extended ICFEM analyses | `temperature_ablation.csv`, `component_ablation.csv` |

The primary files correspond to the OSP manuscript's main reproducibility
claims: automata-oriented diversity and DFA coverage, plus runtime overhead
relative to the internal regex-only baseline.

The optional files correspond to supplementary ICFEM method analyses. They are
included because the repository keeps the full ICFEM reproduction scripts, but
they are not required for the primary OSP artifact check.

## Regenerating Summaries

After running experiment scripts, collect CSV summaries from repository root:

```bash
uv run python scripts/collect_results.py
```

For one experiment group:

```bash
uv run python scripts/collect_results.py --experiment diversity
uv run python scripts/collect_results.py --experiment runtime
uv run python scripts/collect_results.py --experiment temperature_ablation
uv run python scripts/collect_results.py --experiment component_ablation
```

Compare regenerated files in `results/tables/` with the corresponding files in
this directory.

## Variation

Runtime values are hardware-sensitive and depend on GPU, CUDA, PyTorch,
Transformers, and model-loading behavior.

Sampling-based diversity metrics can vary with seed, model revision, CUDA,
PyTorch, Transformers, and tokenizer/model implementation details. Use the
recorded metadata in the JSON files under `results/` when interpreting
differences.

The optional case study writes `experiments/case_study/case_study_summary.json`
when run. It is documented in `experiments/case_study/README.md` and is not
included here as a tracked reference file.
