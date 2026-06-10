# Technology Stack

**Analysis Date:** 2026-05-19

## Languages

**Primary:**
- Python 3.12 - Main package, scripts, examples, and tests in `src/diverse_guide/`, `scripts/`, `examples/`, and `tests/`; required by `pyproject.toml`, `uv.lock`, and `.python-version`.
- Rust 1.71.1+ - DFA engine and Python extension in `regex_dfa_guide/`; declared by `regex_dfa_guide/Cargo.toml`.

**Secondary:**
- C with OpenMP - Optional WD-shift evaluation kernel in `native/wd_kernel/wd_kernel.c`, built by `native/wd_kernel/Makefile` and `scripts/build_wd_kernel.py`.
- Shell and PowerShell - Reproducibility workflows in `experiments/*.sh`, `experiments/*.ps1`, `experiments/common.sh`, `experiments/common.ps1`, `scripts/build_wd_kernel.sh`, and `scripts/build_wd_kernel.ps1`.
- TOML/YAML/Markdown - Python/Rust package metadata in `pyproject.toml`, `regex_dfa_guide/pyproject.toml`, `regex_dfa_guide/Cargo.toml`, CI in `.github/workflows/test.yml`, and user/developer docs in `README.md`, `REPRODUCIBILITY.md`, and `docs/developing.md`.

## Runtime

**Environment:**
- Python `>=3.12`; local version selector is `.python-version` with `3.12`.
- Rust toolchain `>=1.71.1`; `README.md` and `docs/developing.md` document a verified local toolchain of `rustc 1.94.1` / `cargo 1.94.1`.
- CUDA-capable PyTorch runtime for full model-backed experiments; `Dockerfile` defaults to `nvidia/cuda:13.1.1-runtime-ubuntu24.04`.
- CPU-only operation is supported for tests and small runs because `scripts/generate_re.py`, `scripts/eval_runtime.py`, and `scripts/metrics_eval.py` choose `cuda` only when `torch.cuda.is_available()`.

**Package Manager:**
- `uv` - Primary Python environment and lockfile manager; configured by `pyproject.toml`, `uv.lock`, `README.md`, and `docs/developing.md`.
- `poethepoet` 0.42.1 - Task runner resolved in `uv.lock` and configured under `[tool.poe]` in `pyproject.toml`.
- Cargo - Rust dependency/build manager for `regex_dfa_guide/Cargo.toml` and `regex_dfa_guide/Cargo.lock`.
- Make - Native C build entry points in `native/wd_kernel/Makefile`, `regex_dfa_guide/Makefile`, and `experiments/case_study/Makefile`.
- Lockfiles: `uv.lock` present, `regex_dfa_guide/Cargo.lock` present, `experiments/case_study/uv.lock` present.

## Frameworks

**Core:**
- Hugging Face Transformers 5.3.0 - Model/tokenizer interface for generation via `AutoModelForCausalLM`, `AutoTokenizer`, and `LogitsProcessor` in `src/diverse_guide/guide_rust.py`, `src/diverse_guide/guide_python.py`, `scripts/generate_re.py`, and `scripts/eval_runtime.py`.
- PyTorch 2.8.0 - Tensor operations, logits masking, device/dtype selection, and optional CUDA execution in `src/diverse_guide/guide_rust.py`, `src/diverse_guide/guide_python.py`, `src/diverse_guide/evaluation/perplexity.py`, and scripts under `scripts/`.
- PyO3 0.28.x - Rust-to-Python extension bindings in `regex_dfa_guide/src/python_bindings/mod.rs`; declared as `pyo3 = "0.28.2"` in `regex_dfa_guide/Cargo.toml` and resolved to `0.28.3` in `regex_dfa_guide/Cargo.lock`.
- Maturin 1.12.6 - Rust Python-extension build backend for `regex_dfa_guide/pyproject.toml`; root dev dependency declared as `maturin>=1.7,<2.0` in `pyproject.toml`.

**Testing:**
- Pytest 9.0.2 - Python test runner configured in `pyproject.toml` and used by `tests/`.
- pytest-cov 7.1.0 - Branch/statement coverage configured in `pyproject.toml`.
- Cargo test - Rust unit tests for `regex_dfa_guide/src/diverse_guide_dfa.rs`, invoked by `uv run poe test-rust`.
- PowerShell parser checks - CI validates experiment PowerShell scripts in `.github/workflows/test.yml`.

**Build/Dev:**
- Ruff 0.9.10 - Formatter and linter configured by `[tool.ruff]`, `[tool.ruff.lint]`, and `[tool.ruff.format]` in `pyproject.toml`.
- setuptools 77+ and wheel - Root Python build backend declared in `pyproject.toml`.
- Docker - Reproducible CUDA/Python/Rust environment defined by `Dockerfile`.
- cargo-llvm-cov - Optional Rust coverage task `uv run poe cov-rust` in `pyproject.toml` and `latest-cov` Docker target in `Dockerfile`.

## Key Dependencies

**Critical:**
- `regex-dfa-guide` 0.2.0 - Local path dependency from `regex_dfa_guide/`; exposes `DiverseGuideDFA` to `src/diverse_guide/guide_rust.py` and evaluation scripts.
- `numpy` 2.2.3 - Numeric arrays and metric computation in `src/diverse_guide/evaluation/metrics.py`, `src/diverse_guide/evaluation/string_kernel.py`, `src/diverse_guide/evaluation/string_kernel_py.py`, and `scripts/generate_re.py`.
- `scipy` 1.15.2 - Entropy calculation in `src/diverse_guide/evaluation/metrics.py`.
- `torch` 2.8.0 - Core model-generation tensor runtime in `src/diverse_guide/` and `scripts/`.
- `transformers` 5.3.0 - Hugging Face model/tokenizer API in `src/diverse_guide/` and model-backed scripts.
- `tqdm` 4.67.1 - Progress reporting in `scripts/eval_runtime.py`.

**Infrastructure:**
- `outlines[transformers]` 1.2.12 - Optional external regex baseline selected by `--baseline-backend outlines` in `scripts/eval_runtime.py`; dependency group is `outlines` in `pyproject.toml`.
- `bincode` 2.0.1 - Rust DFA serialization/pickle support in `regex_dfa_guide/Cargo.toml` and `regex_dfa_guide/Cargo.lock`.
- `regex-automata` declared 0.4.9, resolved 0.4.14 - DFA construction backend in `regex_dfa_guide/Cargo.toml` and `regex_dfa_guide/Cargo.lock`.
- `rustc-hash` declared 2.1.1, resolved 2.1.2 - Fast hash collections in Rust DFA code.
- `thiserror` declared 2.0.12, resolved 2.0.18 - Rust error definitions in `regex_dfa_guide/src/error.rs`.
- `coverage` 7.13.5, `idna` 3.11, and `pytest` 9.0.2 - Isolated optional case-study dependencies in `experiments/case_study/pyproject.toml` and `experiments/case_study/uv.lock`.

## Configuration

**Environment:**
- Root Python package metadata, dependency groups, Poe tasks, pytest settings, Ruff settings, setuptools package discovery, and uv workspace/path dependency mapping live in `pyproject.toml`.
- Rust extension package metadata and maturin binding settings live in `regex_dfa_guide/pyproject.toml`.
- Rust crate metadata, feature flags, library crate types, release profile, and lints live in `regex_dfa_guide/Cargo.toml`.
- Optional case-study environment is isolated under `experiments/case_study/pyproject.toml`; it is excluded from the root uv workspace by `pyproject.toml`.
- No `.env` files are present at repository root or in the scanned shallow tree; secret-backed configuration is not part of this codebase.

**Build:**
- `Dockerfile` installs system build tools, Rust, uv, Python 3.12, root dev dependencies, the local Rust extension, and the optional WD kernel.
- `.github/workflows/test.yml` runs CI on Ubuntu and macOS with Python 3.12, stable Rust, uv, maturin, pytest coverage, cargo test, Ruff, and PowerShell script checks.
- `native/wd_kernel/Makefile` builds `wd_kernel.so` on Linux/macOS and `wd_kernel.dll` on Windows; `scripts/build_wd_kernel.py` wraps this build from Python.
- `regex_dfa_guide/Makefile` provides extension build/test/dev helper targets, but root workflows prefer Poe tasks from `pyproject.toml`.

## Platform Requirements

**Development:**
- Python 3.12 with `uv sync --group dev` from repository root.
- Rust toolchain 1.71.1+ with Cargo for `regex_dfa_guide/`.
- C compiler with OpenMP for optional WD-shift native kernel: GCC/OpenMP on Linux, Homebrew `libomp` on macOS, or MinGW/MSVC paths on Windows as documented in `README.md`.
- CUDA-capable GPU and compatible NVIDIA drivers are recommended for full model-backed generation and evaluation; tests do not require model downloads or a GPU.

**Production:**
- No service deployment target is defined. The deliverable is a Python package with a local Rust extension, optional C/OpenMP evaluation acceleration, reproducibility scripts, and Docker images from `Dockerfile`.
- Published package metadata points to the GitHub repository and paper DOI in `[project.urls]` of `pyproject.toml`.

---

*Stack analysis: 2026-05-19*
