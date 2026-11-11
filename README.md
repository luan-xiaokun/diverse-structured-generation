# DiverseGuide

![Version](https://img.shields.io/badge/version-v0.3.0-blue)
![License](https://img.shields.io/badge/license-MIT-green)

**OSP title:** DiverseGuide: Automata-Based Steering for Diverse Structured Generation

This repository is the DiverseGuide software artifact for the paper:

> **Automata-Based Steering of Large Language Models for Diverse Structured Generation**
> Xiaokun Luan, Zeming Wei, Yihao Zhang, Meng Sun
> 26th International Conference on Formal Engineering Methods (ICFEM 2025)

It implements a diversity-enhancing method for LLM structured generation constrained by regular expressions.

The canonical software name and Python distribution name are `DiverseGuide` and
`diverse-guide`. The repository remains
`luan-xiaokun/diverse-structured-generation` for continuity with the ICFEM 2025
paper link.

---

## Table of Contents

- [Overview](#overview)
- [Prerequisites](#prerequisites)
  - [Hardware](#hardware)
  - [Software](#software)
  - [Running without an NVIDIA GPU](#running-without-an-nvidia-gpu)
- [Project Structure](#project-structure)
- [Installation](#installation)
  - [Path A: Run with Docker (NVIDIA GPU)](#path-a-run-with-docker-nvidia-gpu)
  - [Path B: Local environment setup](#path-b-local-environment-setup)
- [Minimal Smoke Test](#minimal-smoke-test)
- [Quick Start](#quick-start)
- [API Reference](#api-reference)
- [Running Experiments](#running-experiments)
- [Evaluation Backend](#evaluation-backend)
- [Tests](#tests)
- [Acknowledgements](#acknowledgements)
- [License](#license)

---

## Overview

`DiverseGuide` adds regex-constrained diverse generation on top of standard Hugging Face Transformers workflows.
You keep using `AutoModelForCausalLM` and `AutoTokenizer`, then create a guided generator:

- `DiverseGuide(model, tokenizer, regex_str, ...)` (a class) for diversity-enhanced constrained generation
- `baseline_regex(model, tokenizer, regex_str, ...)` (a factory function) for constrained generation without diversity adjustment

Typical usage flow:

1. Load model/tokenizer with `transformers.from_pretrained`.
2. Create a generator with a target regex.
3. Generate one sample (`__call__`) or multiple samples (`generate_batch`).

Implementation split:

- **Rust (`regex_dfa_guide/`)**: builds and minimizes DFA, compiles token-level transitions, and maintains counter/state logic efficiently.
- **Python (`src/diverse_guide/`)**: integrates with Transformers generation and exposes user-facing APIs.
- **Optional C/OpenMP (`native/wd_kernel/`)**: accelerates WD-shift kernel computation for evaluation metrics only.

---

## Prerequisites

### Hardware

| Use | Requirement |
| --- | --- |
| Full experiments (`experiments/1_generation.*` to `5_component_ablation.*`) | NVIDIA GPU with CUDA support and an NVIDIA driver. The reported results used a GeForce RTX 3060 Ti (8 GB). |
| Docker setup (Path A) | Additionally the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html). The image is built on `nvidia/cuda:13.1.1-runtime-ubuntu24.04`, so the host driver must support CUDA 13.1. |
| Tests, smoke test, small generation runs | Any machine supported by PyTorch; no GPU needed (see [Running without an NVIDIA GPU](#running-without-an-nvidia-gpu)). On a CPU, the manuscript example needs about 4 GB of memory and the experiment scripts about 10 GB (they load the model in float32). |
| Disk | About 3 GB for `Qwen/Qwen2.5-1.5B-Instruct`; about 7 GB more for `microsoft/Phi-4-mini-instruct` (only for the optional temperature ablation), plus the Python environment. |

### Software

| Tool | Version | Needed for | Installation |
| --- | --- | --- | --- |
| Python | >= 3.12 | everything | [python.org](https://www.python.org/downloads/), or `uv python install 3.12` |
| uv | >= 0.11.5 | dependency management and all commands below | [uv installation guide](https://docs.astral.sh/uv/getting-started/installation/) |
| Rust toolchain | >= 1.85 | building the `regex_dfa_guide` extension during `uv sync` | [rustup](https://rustup.rs/) |
| C compiler with OpenMP | e.g. GCC >= 9 | optional: fast WD-shift kernel for Vendi evaluation ([Step 3](#step-3---wd-shift-native-extension-evaluation-only-optional)) | GCC on Linux, `brew install libomp` on macOS, [MSYS2](https://www.msys2.org/) MinGW-w64 or MSVC on Windows |
| Docker | recent release | optional: Path A | [Get Docker](https://docs.docker.com/get-started/get-docker/) |
| Git | any | cloning the repository | [git-scm.com](https://git-scm.com/downloads) |

Supported operating systems are Linux and Windows; every experiment script has a
Bash and a PowerShell entry point. Models are downloaded from the
[Hugging Face Hub](https://huggingface.co/) on first use, so the first run needs
network access (or a pre-populated Hugging Face cache).

### Running without an NVIDIA GPU

Without an NVIDIA GPU, use the local setup (Path B); the Docker image targets
NVIDIA GPUs only. The following run on a CPU:

| Command | What it checks | Model needed | Approximate CPU time |
| --- | --- | --- | --- |
| `uv run poe test` | Python test suite with coverage | no (mock tokenizer) | seconds |
| `uv run poe test-rust` | Rust unit tests of the DFA backend | no | under a minute (first build longer) |
| `uv run poe lint` | Ruff checks | no | seconds |
| [Minimal Smoke Test](#minimal-smoke-test) | Python package and Rust extension work together | no | seconds |
| `uv run python examples/05_paper_example.py` | the illustrative example of the OSP manuscript: 10 DiverseGuide and 10 baseline samples | `Qwen/Qwen2.5-1.5B-Instruct` | under a minute after the download |
| `uv run poe gen css-color -n 10 --stdout-only` | the experiment generation script on a small sample | `Qwen/Qwen2.5-1.5B-Instruct` | under a minute after the download |
| `experiments/case_study/` | the optional coverage case study, which uses tracked samples | no | see its [README](experiments/case_study/README.md) |

Generation on a CPU works but is much slower than on a GPU, so the full
experiments (thousands of samples per grammar) are not practical there.

---

## Project Structure

```text
src/
  diverse_guide/          # Core Python library (public API and logits processors)
    __init__.py
    guide_rust.py         # DiverseGuide, baseline_regex, processors (Rust-backed)
    guide_python.py       # Pure-Python reference implementation (used by tests)
    vocab.py              # Tokenizer vocabulary mapping
    evaluation/           # Artifact reproduction and evaluation utilities
      metrics.py          # DFA coverage, distinct n-grams, Vendi score
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
  collect_results.py      # Collect JSON results into CSV tables
  repro_results.py        # Helpers for structured result files
  build_wd_kernel.py      # Build/check WD-kernel extension
  benchmark_wd_kernel.py  # Benchmark C vs NumPy kernel backend
  benchmark_runtime_latency.py  # Developer tool: throughput and token latency
  profile_regex_processor.py    # Developer tool: logits-processor overhead
  profile_runtime_processor.py  # Developer tool: processor microbenchmark

examples/                 # Standalone usage examples (05 = OSP manuscript example)
tests/                    # Pytest suite (no language model needed)
experiments/              # Reproduction scripts (Bash and PowerShell) and case study
artifact-results/         # Tracked reference CSV summaries
docs/                     # Architecture and developer documentation

Dockerfile                # Main container build
REPRODUCIBILITY.md        # Reproduction protocol
```

For implementation architecture, see [docs/architecture.md](docs/architecture.md).
For contributor and developer workflows, see [docs/developing.md](docs/developing.md).

---

## Installation

For reproducibility, this project supports two setup paths:

- **Path A (Docker)**: reproduces the full GPU environment in a container
- **Path B (Local)**: native setup on your host machine; also the CPU-only path

### Path A: Run with Docker (NVIDIA GPU)

The repository provides a unified `Dockerfile` with multi-stage targets.
It requires an NVIDIA GPU, a CUDA 13.1-capable driver, and the NVIDIA Container
Toolkit (see [Prerequisites](#prerequisites)).

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

Install [uv](https://docs.astral.sh/uv/) (>= 0.11.5) and a Rust toolchain
(>= 1.85) first. This project keeps build/test tools in the `dev` dependency
group. The root project also declares the Rust DFA extension as a local path
dependency, so `uv sync` builds and installs it automatically.

```bash
uv sync
source .venv/bin/activate          # Linux / macOS
# .venv\Scripts\Activate.ps1       # Windows (PowerShell)
```

#### Step 2 - Rust DFA extension

Requires **Rust >= 1.85**, matching `rust-version` in
`regex_dfa_guide/Cargo.toml` (install via [rustup](https://rustup.rs/)).

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

If you want to run the optional extended ICFEM temperature-ablation analysis,
you should also prepare `microsoft/Phi-4-mini-instruct` for perplexity evaluation.

```bash
python -c "
from transformers import AutoModelForCausalLM, AutoTokenizer
model = 'Qwen/Qwen2.5-1.5B-Instruct'
AutoTokenizer.from_pretrained(model)
AutoModelForCausalLM.from_pretrained(model)
"
```

Optional additional download for the optional extended ICFEM temperature-ablation analysis:

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
from diverse_guide import DiverseGuide, baseline_regex

# Import check for public API
assert callable(DiverseGuide)
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

This is the illustrative example of the OSP manuscript
([examples/05_paper_example.py](examples/05_paper_example.py)); it also runs on
a CPU.

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from diverse_guide import DiverseGuide, baseline_regex

model_id = "Qwen/Qwen2.5-1.5B-Instruct"
device = "cuda" if torch.cuda.is_available() else "cpu"

model = AutoModelForCausalLM.from_pretrained(model_id).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_id)
regex = r"#[0-9a-fA-F]{6}"
prompt = "Give me a CSS color code."

generator = DiverseGuide(model, tokenizer, regex)
samples = generator.generate_batch(prompt, n=10, max_tokens=20)

baseline = baseline_regex(model, tokenizer, regex)
base_samples = baseline.generate_batch(prompt, n=10, max_tokens=20)
```

Within one `generate_batch` call, the parallel sequences share the local state
counter; afterwards, completed samples that match the regex are added to the
global path counter, which steers later calls.

See [examples/](examples/) for more usage patterns, including single-sample
generation with `__call__`.

---

## API Reference

The package exposes one class and two factory functions. All three produce
generator objects with the same interface (the methods of
`StatefulSequenceGeneratorAdapter`).

| Entry point | Kind | Returns |
| --- | --- | --- |
| `DiverseGuide(model, tokenizer, regex_str, gamma=0.5, beta=3.0, ablation_component=None, **generation_kwargs)` | class (subclass of `StatefulSequenceGeneratorAdapter`) | a diversity-steered generator |
| `diverse_regex(model, tokenizer, regex_str, gamma=0.5, beta=3.0, ablation_component=None, **generation_kwargs)` | factory function (compatibility helper) | a `DiverseGuide` instance |
| `baseline_regex(model, tokenizer, regex_str, **generation_kwargs)` | factory function | a `StatefulSequenceGeneratorAdapter` with the mask-only `RegexMaskLogitsProcessor` (no diversity adjustment); this is the internal comparison baseline |

### Generator interface

| Method | Description |
|--------|-------------|
| `__call__(prompt, max_tokens=None, **kwargs)` | Generate one sequence. Does not update the global path counter; call `update_generated_content` afterwards if later calls should take the sample into account. |
| `generate_batch(prompt, n, max_tokens=None, **kwargs)` | Generate `n` sequences in one `model.generate` call; afterwards adds each completed sample that matches the regex to the global path counter. |
| `update_generated_content(text)` | Add an externally generated string to the global path counter; only full regex matches are counted. |

### Parameters

| Parameter | Default | Effect |
|-----------|---------|--------|
| `gamma` | `0.5` | Reward scale. Higher values more aggressively boost under-explored paths. |
| `beta` | `3.0` | Penalty scale. Higher values more strongly suppress tokens reused within the current batch. |
| `ablation_component` | `None` | For ablation studies: `"reward"`, `"penalty"`, or `"range_scaling"` disables that component of the logit adjustment. Other values raise `ValueError`. |
| `**generation_kwargs` | see below | Default arguments forwarded to Hugging Face `model.generate` on every call (for example `temperature`, `top_p`). Per-call `**kwargs` override them. |

The logit adjustment is documented in the docstring of
`DiverseRegexLogitsProcessor` and in [docs/architecture.md](docs/architecture.md).

### Sampling defaults

Generators sample with `do_sample=True` from the full next-token distribution
restricted to valid tokens: unless you pass them explicitly, the sampling
arguments are fixed to `temperature=1.0`, `top_k=0`, `top_p=1.0`, and
`repetition_penalty=1.0` (exported as `diverse_guide.PURE_SAMPLING_DEFAULTS`).
They are **not** taken from the model's `generation_config.json`. Pass any of
these arguments to the constructor, or to a single call, to change them. Other
generation settings, such as the end-of-sequence token, still come from the
model's generation config.

---

## Running Experiments

The experiment entry points live in [experiments/](experiments/); each has a
Bash (`.sh`) and a PowerShell (`.ps1`) version. They need an NVIDIA GPU in
practice. Times are approximate for a single consumer GPU.

| Entry point | What it does | Output | Approx. time |
| --- | --- | --- | --- |
| `experiments/1_generation.*` | Generates 1,000 DiverseGuide and 1,000 baseline samples for each of the four grammars (email, css-color, json, no-bomb) with `Qwen/Qwen2.5-1.5B-Instruct`. | `data/diverse/{model}/`, `data/baseline/{model}/` | about 1 hour |
| `experiments/2_diversity_evaluation.*` | Computes DFA state, transition, and path coverage, distinct n-grams, and the Vendi score of the samples from step 1. | `results/diversity/` | a few minutes |
| `experiments/3_efficiency_evaluation.*` | Measures generation throughput (tokens per second) of DiverseGuide and the baseline for each grammar. | `results/runtime/` | about 1 hour |
| `experiments/4_temperature_ablation.*` | Optional: regenerates the samples of step 1 at temperature 1.5 and evaluates diversity and perplexity (with `microsoft/Phi-4-mini-instruct`). | `data/{diverse,baseline}/{model}-temperature_1.5/`, `results/temperature_ablation/` | about 1 hour plus perplexity evaluation |
| `experiments/5_component_ablation.*` | Optional: generates css-color samples with the reward, penalty, or range scaling disabled and evaluates them. | `results/component_ablation/` | about 1 hour |
| `experiments/case_study/` | Optional: uses tracked baseline and DiverseGuide samples as test inputs for `email_validator` and `webcolors` and measures their code coverage; has its own environment ([README](experiments/case_study/README.md)) and needs no GPU. | `experiments/case_study/case_study_summary.json` | short |
| `uv run python scripts/collect_results.py` | Collects the JSON results of the steps above into CSV tables; use `--experiment <name>` for a single group. | `results/tables/*.csv` | seconds |

The same scripts can be run directly through `poe` tasks for individual
grammars:

```bash
# Generate samples for one grammar (grammars: no-bomb, email, css-color, json)
uv run poe gen css-color                   # DiverseGuide samples
uv run poe gen css-color --baseline        # baseline (mask-only) samples
uv run poe gen css-color -n 100 --batch-size 16

# Output control: write to a file or directory, or print only
uv run poe gen css-color --output /tmp/css-color.json
uv run poe gen css-color --stdout-only

# Evaluate diversity metrics of generated samples
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
The generation and runtime scripts record the sampling arguments they used in
their JSON output.

### Reproducing OSP artifact results

For the reviewer-facing reproducibility protocol, start with
[REPRODUCIBILITY.md](REPRODUCIBILITY.md).

Reference CSV summaries are tracked under
[artifact-results/v0.3.0/](artifact-results/v0.3.0/). Regenerated local outputs
are written under ignored `results/`; primary results can be summarized with
`uv run python scripts/collect_results.py --experiment diversity` and
`uv run python scripts/collect_results.py --experiment runtime`.

The OSP primary reproduction path is:

1. Generate baseline and DiverseGuide samples with `experiments/1_generation.*`.
2. Evaluate diversity and DFA coverage with `experiments/2_diversity_evaluation.*`.
3. Measure runtime with `experiments/3_efficiency_evaluation.*`.
4. Run `uv run python scripts/collect_results.py --experiment diversity` and
   `uv run python scripts/collect_results.py --experiment runtime`.
5. Compare `results/tables/diversity.csv` and `results/tables/runtime.csv`
   with `artifact-results/v0.3.0/primary/`.

The optional temperature and component ablations can be compared with
`artifact-results/v0.3.0/optional/temperature_ablation.csv` and
`artifact-results/v0.3.0/optional/component_ablation.csv`.

The diversity results should be interpreted through automata-oriented coverage
metrics first: DFA state coverage, transition coverage, and path coverage.
String-diversity metrics such as Vendi score are task-sensitive and are not
expected to improve for every grammar.

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

The test suite does **not** require a language model or a GPU.
All tests use a small mock tokenizer and synthetic DFA inputs.

```bash
# Run all tests with branch + statement coverage
uv run poe test

# Or directly:
pytest --cov=diverse_guide --cov-report=term-missing --cov-branch
```

Coverage is reported for the `diverse_guide` package (the core deliverable).
The Rust extension (`DiverseGuideDFA`) is tested via its Python bindings in [tests/test_dfa.py](tests/test_dfa.py)
and by its own Rust unit tests.

Verification status for the `v0.3.0` release:

- Python test suite: `83 passed` (statement and branch coverage of `diverse_guide`: 90%)
- Rust unit tests (`regex_dfa_guide`): `15 passed`
- Python linting with Ruff: passed
- `CITATION.cff` validation with `cffconvert`: passed

The release checklist in [docs/developing.md](docs/developing.md) lists the
checks to run before an artifact snapshot.

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

## Acknowledgements

[uthash](https://github.com/troydhanson/uthash) is used in the native C kernel implementation.

---

## License

MIT. See [LICENSE](LICENSE).
