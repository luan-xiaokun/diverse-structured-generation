# Technology Stack

**Analysis Date:** 2026-07-23

## Languages

**Primary:**
- Python 3 — the exploratory witness-construction demo in `aiap/cov_test.py`.

**Secondary:**
- Rust — used indirectly by the parent repository's `regex_dfa_guide` extension that exposes DFA operations to Python.

## Runtime

**Environment:**
- Python virtual environment at the repository root (`.venv/`) is present; this demo does not declare a separate environment.
- The demo is executed as a script with Python, from a working directory where `experiments/` is resolvable.

**Package Manager:**
- Parent project uses `pyproject.toml` and `uv.lock`; `aiap/` has no independent package manifest or lockfile.

## Frameworks

**Core:**
- `regex_dfa_guide` / `diverse_guide` — parent-project packages that provide DFA traversal, token constraints, and coverage metrics.
- Hugging Face `transformers` — `AutoTokenizer` loads the tokenizer used to convert token IDs into witness text.

**Testing:**
- No `aiap/` tests or test runner configuration yet.

## Key Dependencies

**Critical:**
- `DiverseGuideDFA` — DFA state, transition, accepted-token, and transition-sequence APIs.
- `DiverseRegexGuide` — builds the DFA from a regular expression and tokenizer.
- `transformers.AutoTokenizer` — loads `Qwen/Qwen2.5-0.5B` in the current demo.

## Configuration

- Input datasets are hard-coded relative paths under `experiments/case_study/data/` in `aiap/cov_test.py`.
- Coverage threshold defaults to one visit in `compute_coverage_metrics` and `get_coverage_obligations`.
- No environment variables, deployment configuration, or user-facing CLI configuration are defined.

## Platform Requirements

**Development:**
- Python environment compatible with the parent repository and its Rust/Python extension.
- Network or cached Hugging Face model artifacts are needed on first tokenizer load.

**Production:**
- This is a research demo, not a deployed service.

---
*Stack analysis: 2026-07-23*
