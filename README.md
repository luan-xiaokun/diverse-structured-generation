# Diverse Structured Generation

![Python coverage](https://img.shields.io/badge/python%20coverage-96%25-brightgreen)
![Rust coverage](https://img.shields.io/badge/rust%20coverage-97.22%25-brightgreen)

This repository is the artifact for the paper:

> **Automata-Based Steering of Large Language Models for Diverse Structured Generation**
> Xiaokun Luan, Zemin Wei, Yihao Zhang,  Meng Sun 
> 26th International Conference on Formal Engineering Methods (ICFEM 2025)

It implements a diversity-enhancing method for LLM structured generation constrained by regular expressions.
Some code is adapted from [Outlines](https://github.com/dottxt-ai/outlines);
[uthash](https://github.com/troydhanson/uthash) is used in the native C kernel implementation.

---

## Algorithm

Standard regex-constrained generation (e.g., Outlines) builds a DFA from the regex and, at each generation step, masks all tokens that would lead to a dead state. This enforces validity but does not promote diversity: the model tends to produce the same high-probability outputs repeatedly.

**This work** adds a logit adjustment on top of the masking step. For each allowed token at state $s$, the adjustment is:

$$\Delta_i = \gamma \cdot \text{logits\_range} \cdot \frac{\log(1 + \sum_j c^{\text{path}}_j)}{1 + c^{\text{path}}_i} \cdot \frac{1}{\beta \cdot (c^{\text{local}}_i)^2}$$

where:
- $c^{\text{path}}_i$ — minimum path-counter along token $i$'s byte-state sequence (global reward: tokens traversing less-visited paths are boosted)
- $c^{\text{local}}_i$ — maximum local-state-counter along token $i$'s byte-state sequence (per-batch penalty: tokens that have already been used heavily in the current batch are suppressed)
- $\gamma$, $\beta$ — reward and penalty scale hyperparameters

The path counter is updated after each complete sequence is generated, so later generations within a session are steered away from already-explored DFA paths.

The DFA is built by [regex-automata](https://github.com/BurntSushi/regex-automata) (Rust), minimized, and compiled into token-level transitions once per `(regex, tokenizer)` pair.

---

## Installation

### Step 1 — Python dependencies

We recommend [uv](https://docs.astral.sh/uv/) for dependency management.
This project keeps build/test tools (including `maturin`) in the `dev` dependency group,
so run `uv sync` first.

```bash
uv sync
source .venv/bin/activate          # Linux / macOS
# .venv\Scripts\Activate.ps1       # Windows (PowerShell)
```

### Step 2 — Rust DFA extension

Requires **Rust ≥ 1.75** (install via [rustup](https://rustup.rs/)).

Run the build command from the repository root **after activating the root `.venv`**.
Do not use `uv run` inside `regex_dfa_guide/`, otherwise `uv` may resolve to that
subdirectory's own environment.

```bash
maturin develop --release -m regex_dfa_guide/Cargo.toml
```

This builds `regex_dfa_guide` and installs it into the currently active (root) virtual environment.
Re-run after any change to `regex_dfa_guide/src/`.

### Step 3 — WD-shift native extension *(evaluation only, optional)*

The native extension is used only for computing the Vendi diversity metric during evaluation.
It is **not required for generation**. If it is not built, a pure Python/NumPy fallback
is used automatically (see [Evaluation backend](#evaluation-backend)).

Quick build (recommended):

```bash
# Linux / macOS
python3 scripts/build_wd_kernel.py
```

```powershell
# Windows PowerShell
python scripts/build_wd_kernel.py
```

The script compiles from `native/wd_kernel/` and places artifacts in
`build/native/wd_kernel/`.

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

Requires `gcc` (≥ 9) with OpenMP support (standard on most distributions).

```bash
cd native/wd_kernel && make && cd ../..
```

</details>

<details>
<summary><b>macOS</b></summary>

Apple's default `clang` does not include OpenMP.  Install it via Homebrew first:

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
<summary><b>Windows — WSL2 (recommended)</b></summary>

Run everything inside a WSL2 Ubuntu shell.  All Linux instructions apply.

```bash
# inside WSL2
cd native/wd_kernel && make && cd ../..
```

</details>

<details>
<summary><b>Windows — native (MinGW-w64)</b></summary>

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

> **Note**: the `.dll` will be loaded by `string_kernel.py` automatically on Windows.

Alternatively, with MSVC (Visual Studio 2019+):
```cmd
cl.exe /O2 /openmp /LD native\wd_kernel\wd_kernel.c /Febuild\native\wd_kernel\wd_kernel.dll
```

</details>

### Step 4 — Download models

Models are loaded via `transformers.from_pretrained`, so you can use any compatible model.
The default model used in experiments is `Qwen/Qwen2.5-1.5B-Instruct`.
If you want to reproduce the temperature-ablation results (Table 5 to Table 7),
you should also prepare `microsoft/Phi-4-mini-instruct` for perplexity
evaluation.

```bash
python -c "
from transformers import AutoModelForCausalLM, AutoTokenizer
model = 'Qwen/Qwen2.5-1.5B-Instruct'
AutoTokenizer.from_pretrained(model)
AutoModelForCausalLM.from_pretrained(model)
"
```

Optional additional download for Table 5 to Table 7:

```bash
python -c "
from transformers import AutoModelForCausalLM, AutoTokenizer
model = 'microsoft/Phi-4-mini-instruct'
AutoTokenizer.from_pretrained(model)
AutoModelForCausalLM.from_pretrained(model)
"
```

---

## Minimal Smoke Test

After finishing setup, run this quick check from repository root. It does not load
a large language model; it only verifies that the Python package and Rust extension
are installed and working together.

```bash
PYTHONPATH=src python - <<'PY'
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

See [`examples/`](examples/) for more complete usage patterns.

---

## API Reference

### `diverse_regex(model, tokenizer, regex_str, gamma=0.5, beta=3.0, **generation_kwargs)`

Returns a `StatefulSequenceGeneratorAdapter` configured for diverse generation.

### `baseline_regex(model, tokenizer, regex_str, **generation_kwargs)`

Returns a `StatefulSequenceGeneratorAdapter` with `gamma=0` (pure constrained generation, no diversity adjustment). Used as the comparison baseline.

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
or `data/baseline/{model}/` (baseline).  Use `--output` to override the output path,
or `--stdout-only` to skip writing JSON files.

### Reproducing paper results

Paper-result reproduction scripts live under
[experiments/README.md](/home/lxk/projects/diverse-dfa-gen/experiments/README.md).
That document maps each script to the corresponding paper tables and provides
separate Linux/macOS and Windows entrypoints, setup prerequisites, experiment
dependencies, expected outputs, a minimal sanity-check path, and a recommended
execution order. The Table 9 case study is documented separately in
[experiments/case_study/README.md](/home/lxk/projects/diverse-dfa-gen/experiments/case_study/README.md).
If you want a lightweight artifact check instead of full reproduction, start
with the `Minimal Sanity Check` section in
[experiments/README.md](/home/lxk/projects/diverse-dfa-gen/experiments/README.md).

---

## Evaluation backend

The Vendi score metric requires computing a pairwise WD-shift kernel matrix.  Two
backends are supported, selected automatically at import time:

| Backend | Speed (n=500) | Build required | Platform |
|---------|---------------|----------------|----------|
| C + OpenMP (`native/wd_kernel/` source, `build/native/wd_kernel/` artifact) | ~0.02–0.06 s | `python scripts/build_wd_kernel.py` | Linux, macOS, Windows (MinGW/MSVC) |
| Pure Python/NumPy (fallback) | ~1–2 s (parallel) | none | all platforms |

The fallback uses `concurrent.futures.ProcessPoolExecutor` to parallelize across all
available CPU cores.  For n=1000 on an 8-core machine, it takes roughly 10–20 s;
the C extension takes < 0.5 s.

To check which backend is active:

```python
from string_kernel import STRING_KERNEL_BACKEND  # "c" or "python"
```

---

## Tests

The test suite does **not** require a language model. All tests use a small mock tokenizer and synthetic DFA inputs.

```bash
# Run all tests with branch + statement coverage
uv run poe test

# Or directly:
pytest --cov=diverse_guide --cov-report=term-missing --cov-branch
```

Coverage is reported for the `diverse_guide` package (the core deliverable).
The Rust extension (`DiverseGuideDFA`) is tested via its Python bindings in [`tests/test_dfa.py`](tests/test_dfa.py).

Current test and coverage status (latest local run):

- Python test suite: `140 passed`
- Python core coverage (`diverse_guide`): `96%`
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

---

## Project Structure

```
src/
  diverse_guide/          # Core Python library (the deliverable)
    __init__.py           # Public API: diverse_regex, baseline_regex, ...
    guide_rust.py         # DiverseRegexLogitsProcessor, StatefulSequenceGeneratorAdapter
    vocab.py              # Tokenizer vocabulary decoding utilities
  string_kernel.py        # Evaluation utility: WD-shift kernel (C or NumPy backend)
  string_kernel_py.py     # Evaluation utility: pure NumPy WD-shift kernel (fallback)
  paths.py                # Evaluation utility: output directory path helpers
  perplexity.py           # Evaluation utility: model perplexity calculation
regex_dfa_guide/          # Rust extension (DFA construction + token mapping + counters)
  src/
    diverse_guide_dfa.rs  # DfaIndex, DiverseGuideDFA, counter methods
    python_bindings/      # PyO3 bindings exposing DiverseGuideDFA to Python
native/wd_kernel/         # Native C extension source (evaluation metric only)
build/native/wd_kernel/   # Built native artifacts (.so/.dll), generated by script
scripts/
  generate_re.py          # Run generation for a predefined task
  metrics_eval.py         # Compute diversity metrics on generated samples
  eval_runtime.py         # Measure generation throughput
  benchmark_wd_kernel.py  # Benchmark C vs NumPy kernel backends
examples/                 # Standalone usage examples (no LLM required for DFA examples)
tests/                    # Pytest suite (no LLM required)
experiments/              # Reproduction scripts and case study for the paper tables
deprecated/               # Archived earlier implementation (kept for reference)
```

---

## License

MIT. See [LICENSE](LICENSE).
