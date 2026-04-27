# Diverse Structured Generation

![Python coverage](https://img.shields.io/badge/python%20coverage-89%25-yellowgreen)
![Rust coverage](https://img.shields.io/badge/rust%20coverage-97.22%25-brightgreen)

This repository is the artifact for the paper:

> **Automata-Based Steering of Large Language Models for Diverse Structured Generation**
> Xiaokun Luan, Zemin Wei, Yihao Zhang, Meng Sun
> 26th International Conference on Formal Engineering Methods (ICFEM 2025)

It implements a diversity-enhancing method for LLM structured generation constrained by regular expressions.

---

## Table of Contents

- [Diverse Structured Generation](#diverse-structured-generation)
  - [Table of Contents](#table-of-contents)
  - [Overview](#overview)
  - [Installation](#installation)
    - [Path A: Run with Docker (recommended for reproducibility)](#path-a-run-with-docker-recommended-for-reproducibility)
    - [Path B: Local environment setup](#path-b-local-environment-setup)
      - [Step 1 - Python dependencies](#step-1---python-dependencies)
      - [Step 2 - Rust DFA extension](#step-2---rust-dfa-extension)
      - [Step 3 - WD-shift native extension *(evaluation only, optional)*](#step-3---wd-shift-native-extension-evaluation-only-optional)
      - [Step 4 - Download models](#step-4---download-models)
      - [Non-uv setup note](#non-uv-setup-note)
  - [Minimal Smoke Test](#minimal-smoke-test)
  - [Quick Start](#quick-start)
  - [API Reference](#api-reference)
    - [`diverse_regex(model, tokenizer, regex_str, gamma=0.5, beta=3.0, **generation_kwargs)`](#diverse_regexmodel-tokenizer-regex_str-gamma05-beta30-generation_kwargs)
    - [`baseline_regex(model, tokenizer, regex_str, **generation_kwargs)`](#baseline_regexmodel-tokenizer-regex_str-generation_kwargs)
    - [`StatefulSequenceGeneratorAdapter`](#statefulsequencegeneratoradapter)
    - [Parameters](#parameters)
  - [Running Experiments](#running-experiments)
    - [Reproducing paper results](#reproducing-paper-results)
  - [Evaluation Backend](#evaluation-backend)
  - [Tests](#tests)
  - [Project Structure](#project-structure)
  - [Acknowledgements](#acknowledgements)
  - [License](#license)

---

## Overview

This project adds regex-constrained diverse generation on top of standard Hugging Face Transformers workflows.
You keep using `AutoModelForCausalLM` and `AutoTokenizer`, then create a guided generator via:

- `diverse_regex(model, tokenizer, regex_str, ...)` for diversity-enhanced constrained generation
- `baseline_regex(model, tokenizer, regex_str, ...)` for constrained generation without diversity adjustment

Typical usage flow:

1. Load model/tokenizer with `transformers.from_pretrained`.
2. Create a generator with a target regex.
3. Generate one sample (`__call__`) or multiple samples (`generate_batch`).

Implementation split:

- **Rust (`regex_dfa_guide/`)**: builds and minimizes DFA, compiles token-level transitions, and maintains counter/state logic efficiently.
- **Python (`src/diverse_guide/`)**: integrates with Transformers generation and exposes user-facing APIs.
- **Optional C/OpenMP (`native/wd_kernel/`)**: accelerates WD-shift kernel computation for evaluation metrics only.

---

## Installation

For reproducibility, this project supports two setup paths:

- **Path A (Docker)**: easiest way to reproduce the full environment in a container
- **Path B (Local)**: native setup on your host machine

### Path A: Run with Docker (recommended for reproducibility)

The repository provides a unified `Dockerfile` with multi-stage targets.

> **Prerequisites**: NVIDIA GPU with CUDA support, NVIDIA drivers installed, and [nvidia-container-toolkit](https://github.com/NVIDIA/nvidia-container-toolkit) configured.

Build targets:

| Target | Purpose |
|--------|---------|
| `latest` (default) | GPU image for generation, evaluation, and tests |
| `latest-cov` | Same as `latest`, plus Rust coverage tooling (`cargo-llvm-cov`) |

Build images:

```bash
# Default image
docker build -t diverse-guide:latest .

# Image with coverage tools
docker build --target latest-cov -t diverse-guide:latest-cov .
```

Run image:

```bash
docker run --rm -it --gpus all diverse-guide:latest
```

After entering the container, use the same commands shown in [Tests](#tests), [Running Experiments](#running-experiments), and [Minimal Smoke Test](#minimal-smoke-test).

### Path B: Local environment setup

#### Step 1 - Python dependencies

We recommend [uv](https://docs.astral.sh/uv/) for dependency management.
This project keeps build/test tools in the `dev` dependency group.
The root project also declares the Rust DFA extension as a local path dependency,
so `uv sync` builds and installs it automatically.

```bash
uv sync
source .venv/bin/activate          # Linux / macOS
# .venv\Scripts\Activate.ps1       # Windows (PowerShell)
```

#### Step 2 - Rust DFA extension

Requires **Rust >= 1.75** (install via [rustup](https://rustup.rs/)).

If you use the recommended `uv sync` setup above, no separate command is needed:
`regex-dfa-guide` is installed from the local `regex_dfa_guide/` directory.

If you change Rust sources and want to rebuild the extension explicitly, run the
build command from repository root **after activating the root `.venv`**.
Do not use `uv run` inside `regex_dfa_guide/`, otherwise `uv` may resolve to that
subdirectory's own environment.

```bash
maturin develop --release -m regex_dfa_guide/Cargo.toml
```

This builds `regex_dfa_guide` and installs it into the currently active root virtual environment.
Re-run after any change to `regex_dfa_guide/src/`.

#### Step 3 - WD-shift native extension *(evaluation only, optional)*

The native extension is used only for computing the Vendi diversity metric during evaluation.
It is **not required for generation**.
If it is not built, a pure Python/NumPy fallback is used automatically (see [Evaluation Backend](#evaluation-backend)).

Quick build (recommended):

```bash
uv run python scripts/build_wd_kernel.py
```

The script compiles from `native/wd_kernel/` and places artifacts in `build/native/wd_kernel/`.

Platform wrappers:

```bash
# Linux / macOS
./scripts/build_wd_kernel.sh
```

```powershell
# Windows PowerShell
./scripts/build_wd_kernel.ps1
```

<details>
<summary><b>Linux</b></summary>

Requires `gcc` (>= 9) with OpenMP support (standard on most distributions).

```bash
cd native/wd_kernel && make && cd ../..
```

</details>

<details>
<summary><b>macOS</b></summary>

Apple's default `clang` does not include OpenMP. Install it via Homebrew first:

```bash
brew install libomp
```

Then build:

```bash
cd native/wd_kernel && make && cd ../..
```

The `Makefile` automatically passes the correct `-Xpreprocessor -fopenmp -lomp` flags
and links against Homebrew's `libomp`.

</details>

<details>
<summary><b>Windows - WSL2 (recommended)</b></summary>

Run everything inside a WSL2 Ubuntu shell. All Linux instructions apply.

```bash
# inside WSL2
cd native/wd_kernel && make && cd ../..
```

</details>

<details>
<summary><b>Windows - native (MinGW-w64)</b></summary>

Install [MSYS2](https://www.msys2.org/) and its MinGW-w64 GCC toolchain:

```bash
# in MSYS2 MinGW64 shell
pacman -S mingw-w64-x86_64-gcc mingw-w64-x86_64-openmp
```

Then build (the `Makefile` detects Windows and writes `wd_kernel.dll` under `build/native/wd_kernel/`):

```bash
cd native/wd_kernel
make
cd ../..
```

> **Note**: the `.dll` will be loaded by `diverse_guide.evaluation.string_kernel`
> automatically on Windows.

Alternatively, with MSVC (Visual Studio 2019+):

```cmd
cl.exe /O2 /openmp /LD native\wd_kernel\wd_kernel.c /Febuild\native\wd_kernel\wd_kernel.dll
```

</details>

#### Step 4 - Download models

Models are loaded via `transformers.from_pretrained`, so you can use any compatible model.
The default model used in experiments is `Qwen/Qwen2.5-1.5B-Instruct`.

If you want to reproduce the paper's **temperature-ablation experiment results (Table 5, Table 6, Table 7)**,
you should also prepare `microsoft/Phi-4-mini-instruct` for perplexity evaluation.

```bash
python -c "
from transformers import AutoModelForCausalLM, AutoTokenizer
model = 'Qwen/Qwen2.5-1.5B-Instruct'
AutoTokenizer.from_pretrained(model)
AutoModelForCausalLM.from_pretrained(model)
"
```

Optional additional download for the paper's temperature-ablation experiment results (Table 5, Table 6, Table 7):

```bash
python -c "
from transformers import AutoModelForCausalLM, AutoTokenizer
model = 'microsoft/Phi-4-mini-instruct'
AutoTokenizer.from_pretrained(model)
AutoModelForCausalLM.from_pretrained(model)
"
```

#### Non-uv setup note

`uv` reads the repository's local path dependency configuration and installs
`regex-dfa-guide` from `regex_dfa_guide/` automatically.
Other installers may not understand this uv-specific source mapping.
If you use `pip` directly, install the Rust extension first, then the root project:

```bash
pip install -e regex_dfa_guide
pip install -e .
```

Alternatively, after installing `maturin`, build the Rust extension into the
active environment before installing or running the root project:

```bash
maturin develop --release -m regex_dfa_guide/Cargo.toml
pip install -e .
```

---

## Minimal Smoke Test

After finishing setup, run this quick check from repository root.
It does not load a large language model;
it only verifies that the Python package and Rust extension are installed and working together.

```bash
python - <<'PY'
from regex_dfa_guide import DiverseGuideDFA
from diverse_guide import diverse_regex, baseline_regex

# Import check for public API
assert callable(diverse_regex)
assert callable(baseline_regex)

# Rust extension check
vocab = {0: "<eos>", 1: "a", 2: "b", 3: "ab"}
dfa = DiverseGuideDFA(r"(?:ab)$", 0, vocab)
s0 = dfa.get_initial_state()
allowed = set(dfa.get_allowed_token_ids(s0))

assert 1 in allowed and 3 in allowed  # "a" and "ab" should be valid from start
print("Smoke test passed: imports and DFA extension are working.")
PY
```

If this script prints `Smoke test passed`, your environment setup is correct.

---

## Quick Start

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from diverse_guide import diverse_regex, baseline_regex

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-1.5B-Instruct", dtype=dtype
).to(device)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")

# regex for IPv4 addresses
ipv4_regex = r"(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)$"

generator = diverse_regex(model, tokenizer, ipv4_regex)

# generate a single sample
sample = generator("Give me an IPv4 address.", max_tokens=20)

# generate a diverse batch (updates path counters between samples)
samples = generator.generate_batch("Give me an IPv4 address.", n=10, max_tokens=20)
```

See [examples/](examples/) for more complete usage patterns.

---

## API Reference

### `diverse_regex(model, tokenizer, regex_str, gamma=0.5, beta=3.0, **generation_kwargs)`

Returns a `StatefulSequenceGeneratorAdapter` configured for diverse generation.

### `baseline_regex(model, tokenizer, regex_str, **generation_kwargs)`

Returns a `StatefulSequenceGeneratorAdapter` with `gamma=0`
(pure constrained generation, no diversity adjustment). Used as the comparison baseline.

### `StatefulSequenceGeneratorAdapter`

| Method | Description |
|--------|-------------|
| `__call__(prompt, max_tokens)` | Generate one sequence |
| `generate_batch(prompt, n, max_tokens)` | Generate `n` sequences; updates path counters after each |
| `update_generated_content(text)` | Manually update the path counter with an externally generated string |

### Parameters

| Parameter | Default | Effect |
|-----------|---------|--------|
| `gamma` | `0.5` | Reward scale. Higher values more aggressively boost under-explored paths. `gamma=0` disables diversity (baseline). |
| `beta` | `3.0` | Penalty scale. Higher values more strongly suppress tokens reused within the current batch. |

---

## Running Experiments

```bash
# Generate samples for a grammar (grammars: no-bomb, threefold, ipv4, ipv6, email, css-color, json)
uv run poe gen css-color
uv run poe gen css-color --baseline        # baseline (non-diverse)
uv run poe gen css-color --model Qwen/Qwen2.5-0.5B-Instruct -n 100

# Batch-way generation: same prompt, multiple samples per model call
uv run poe gen css-color -n 100 --batch-size 16

# Output control
uv run poe gen css-color --output /tmp/css-color.json
uv run poe gen css-color --output /tmp/gen_runs
uv run poe gen css-color --stdout-only

# Evaluate diversity metrics on generated samples
uv run poe eval css-color

# Measure generation throughput
uv run poe eval-runtime css-color

# See all options
uv run poe gen --help
uv run poe eval --help
```

By default, generated samples are saved to `data/diverse/{model}/` (diverse)
or `data/baseline/{model}/` (baseline).
Use `--output` to override the output path, or `--stdout-only` to skip writing JSON files.

### Reproducing paper results

Reproduction scripts live in [experiments/README.md](experiments/README.md).
The document maps scripts to paper tables and provides Linux / macOS and Windows entrypoints.

Table mapping summary:

- paper's diversity-evaluation experiment results (Table 1, Table 2, Table 3): `2_diversity_evaluation.*`
- paper's efficiency-evaluation experiment results (Table 4): `3_efficiency_evaluation.*`
- paper's temperature-ablation experiment results (Table 5, Table 6, Table 7): `4_temperature_ablation.*`
- paper's component-ablation experiment results (Table 8): `5_component_ablation.*`
- paper's case-study experiment results (Table 9): `experiments/case_study/`

If you want a lightweight artifact check instead of full reproduction,
start with `Minimal Sanity Check` in [experiments/README.md](experiments/README.md).

---

## Evaluation Backend

The Vendi score metric requires computing a pairwise WD-shift kernel matrix.
Two backends are supported, selected automatically at import time:

| Backend | Speed (n=500) | Build required | Platform |
|---------|---------------|----------------|----------|
| C + OpenMP (`native/wd_kernel/` source, `build/native/wd_kernel/` artifact) | ~0.02-0.06 s | `uv run python scripts/build_wd_kernel.py` | Linux, macOS, Windows (MinGW/MSVC) |
| Pure Python/NumPy (fallback) | ~1-2 s (parallel) | none | all platforms |

The fallback uses `concurrent.futures.ProcessPoolExecutor` to parallelize across all available CPU cores.
For n=1000 on an 8-core machine, it takes roughly 10-20 s; the C extension takes < 0.5 s.

To check which backend is active:

```python
from diverse_guide.evaluation.string_kernel import STRING_KERNEL_BACKEND  # "c" or "python"
```

---

## Tests

The test suite does **not** require a language model.
All tests use a small mock tokenizer and synthetic DFA inputs.

```bash
# Run all tests with branch + statement coverage
uv run poe test

# Or directly:
pytest --cov=diverse_guide --cov-report=term-missing --cov-branch
```

Coverage is reported for the `diverse_guide` package (the core deliverable).
The Rust extension (`DiverseGuideDFA`) is tested via its Python bindings in [tests/test_dfa.py](tests/test_dfa.py).

Current test and coverage status (latest local run):

- Python test suite: `142 passed`
- Python package coverage (`diverse_guide`): `89%`
- Rust unit tests (`regex_dfa_guide`): `11 passed`
- Rust coverage: `Regions 96.37%`, `Lines 97.22%`, `Functions Executed 90.28%`

Rust native tests:

```bash
uv run poe test-rust
```

Rust coverage (optional):

```bash
# one-time install (requires rustc >= 1.87)
cargo install cargo-llvm-cov

# coverage summary for regex_dfa_guide
uv run poe cov-rust
```

Docker note:

- If you use Docker and need Rust coverage, either build `diverse-guide:latest-cov`,
  or install `cargo-llvm-cov` (and required LLVM tooling) manually inside `diverse-guide:latest`.

---

## Project Structure

```text
src/
  diverse_guide/          # Core Python library (public API and logits processor)
    __init__.py
    guide_python.py
    guide_rust.py
    vocab.py
    evaluation/           # Artifact reproduction and evaluation utilities
      metrics.py          # Evaluation metric wrappers
      paths.py            # Output directory path helpers
      perplexity.py       # Perplexity calculation utility
      string_kernel.py    # WD-shift kernel selector (C or NumPy backend)
      string_kernel_py.py # Pure NumPy WD-shift kernel fallback

regex_dfa_guide/          # Rust extension package (DFA construction + counters)
  src/
    diverse_guide_dfa.rs
    error.rs
    lib.rs
    python_bindings/

native/wd_kernel/         # Native C/OpenMP WD-shift kernel source

scripts/
  generate_re.py          # Generate constrained samples
  metrics_eval.py         # Compute diversity metrics
  eval_runtime.py         # Measure generation throughput
  build_wd_kernel.py      # Build/check WD-kernel extension
  benchmark_wd_kernel.py  # Benchmark C vs NumPy kernel backend

examples/                 # Standalone usage examples
tests/                    # Pytest suite
experiments/              # Paper reproduction scripts and case study

Dockerfile                # Main container build
.dockerignore             # Docker build-context exclusions
```

---

## Acknowledgements

[uthash](https://github.com/troydhanson/uthash) is used in the native C kernel implementation.

---

## License

MIT. See [LICENSE](LICENSE).
