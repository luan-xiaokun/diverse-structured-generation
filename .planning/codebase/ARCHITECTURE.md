<!-- refreshed: 2026-05-19 -->
# Architecture

**Analysis Date:** 2026-05-19

## System Overview

```text
┌─────────────────────────────────────────────────────────────┐
│                    User-Facing Python API                    │
├──────────────────┬──────────────────┬───────────────────────┤
│  Public exports  │  HF generation   │ Vocabulary conversion  │
│ `src/diverse_...`│ `guide_rust.py`  │       `vocab.py`       │
└────────┬─────────┴────────┬─────────┴──────────┬────────────┘
         │                  │                     │
         ▼                  ▼                     ▼
┌─────────────────────────────────────────────────────────────┐
│                      Rust DFA Backend                        │
│ `regex_dfa_guide/src/diverse_guide_dfa.rs`                  │
│ `regex_dfa_guide/src/python_bindings/mod.rs`                │
└────────────────────────────┬────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────┐
│              Evaluation, Reproduction, and Outputs           │
│ `src/diverse_guide/evaluation/`, `scripts/`, `experiments/` │
│ `data/`, `results/`, `artifact-results/`                    │
└─────────────────────────────────────────────────────────────┘
```

The repository is a layered research software artifact. The runtime package is `diverse-guide`, implemented in Python under `src/diverse_guide/`, with regex/DFA performance-critical work delegated to the local Rust extension package `regex-dfa-guide` under `regex_dfa_guide/`. Evaluation and reproduction code is kept outside the package in `scripts/`, `experiments/`, `data/`, `results/`, and `artifact-results/`.

## Component Responsibilities

| Component | Responsibility | File |
|-----------|----------------|------|
| Public Python package namespace | Export the supported API surface and keep callers off internal module paths | `src/diverse_guide/__init__.py` |
| Rust-backed generation API | Integrate Hugging Face `model.generate()` with regex masking, diversity reward, local penalty, and stateful counter updates | `src/diverse_guide/guide_rust.py` |
| Pure-Python reference guide | Reference implementation for equivalence checks and debugging; do not make it the default runtime path | `src/diverse_guide/guide_python.py` |
| Tokenizer vocabulary conversion | Convert Hugging Face tokenizer vocabulary into `token_id -> decoded string` inputs for DFA construction | `src/diverse_guide/vocab.py` |
| Rust DFA index and counters | Build minimized byte DFA, compile token transitions, track global path counters and per-call local counters | `regex_dfa_guide/src/diverse_guide_dfa.rs` |
| PyO3 DFA bindings | Expose `DiverseGuideDFA` to Python behind `Arc<RwLock<...>>`, including pickle/deepcopy support | `regex_dfa_guide/src/python_bindings/mod.rs` |
| Rust extension package shim | Re-export the compiled `regex_dfa_guide_rs.DiverseGuideDFA` object for Python imports | `regex_dfa_guide/python/regex_dfa_guide/__init__.py` |
| Diversity metrics | Compute DFA coverage, path coverage, Vendi score, and n-gram metrics from generated sample strings | `src/diverse_guide/evaluation/metrics.py` |
| String-kernel selector | Dispatch WD-shift kernel calls to C/OpenMP library when available, otherwise NumPy/Python fallback | `src/diverse_guide/evaluation/string_kernel.py` |
| Pure string-kernel fallback | Implement WD-shift kernel without native build requirements | `src/diverse_guide/evaluation/string_kernel_py.py` |
| Perplexity metric | Compute optional model-backed perplexity for generated strings | `src/diverse_guide/evaluation/perplexity.py` |
| Output path helper | Build consistent generated-data paths from experiment arguments | `src/diverse_guide/evaluation/paths.py` |
| Sample generation CLI | Load model/tokenizer, generate regex-constrained samples, write JSON datasets | `scripts/generate_re.py` |
| Diversity metrics CLI | Load generated samples, build DFA for grammar regex, compute structured metric JSON | `scripts/metrics_eval.py` |
| Runtime metrics CLI | Measure generation throughput for diverse and baseline generators | `scripts/eval_runtime.py` |
| Results collector | Convert structured JSON result groups into CSV tables | `scripts/collect_results.py` |
| Reproduction metadata | Attach git commit, Python version, platform, and timestamps to JSON results | `scripts/repro_results.py` |
| Native WD-shift build helper | Compile optional C/OpenMP metric backend and place artifacts under `build/native/wd_kernel/` | `scripts/build_wd_kernel.py` |
| Optional native metric backend | C/OpenMP implementation of WD-shift kernel for evaluation speed | `native/wd_kernel/wd_kernel.c` |
| Experiment wrappers | Cross-platform command orchestration for generation, diversity, runtime, ablation, and case-study runs | `experiments/*.sh`, `experiments/*.ps1`, `experiments/case_study/` |
| Root test suite | Verify package behavior, bindings, metrics, scripts, packaging, and reference outputs | `tests/` |

## Pattern Overview

**Overall:** Layered Python package with Rust extension backend and script-driven experiment orchestration.

**Key Characteristics:**
- The public package API is intentionally small: import from `diverse_guide` through `src/diverse_guide/__init__.py`.
- The default generation implementation is Rust-backed through `src/diverse_guide/guide_rust.py`; `src/diverse_guide/guide_python.py` is a reference implementation.
- The DFA backend separates immutable DFA/token-transition data (`DfaIndex`) from mutable diversity counters (`DiverseGuideDFA`) in `regex_dfa_guide/src/diverse_guide_dfa.rs`.
- Evaluation code is pure where possible under `src/diverse_guide/evaluation/`, while CLI scripts in `scripts/` own argument parsing, filesystem output, and model loading.
- Reproduction wrappers in `experiments/` call the Python scripts and write results under `data/` and `results/`; reference outputs live under `artifact-results/`.

## Layers

**Public Python API:**
- Purpose: Provide the stable user-facing generator and helper functions.
- Location: `src/diverse_guide/__init__.py`, `src/diverse_guide/guide_rust.py`
- Contains: `DiverseGuide`, `diverse_regex`, `baseline_regex`, `DiverseRegexLogitsProcessor`, `RegexMaskLogitsProcessor`, `StatefulSequenceGeneratorAdapter`
- Depends on: Hugging Face Transformers, PyTorch, `regex_dfa_guide`, `src/diverse_guide/vocab.py`
- Used by: Examples in `examples/`, generation scripts in `scripts/generate_re.py`, runtime scripts in `scripts/eval_runtime.py`, tests in `tests/test_guide.py`

**Vocabulary Conversion:**
- Purpose: Normalize tokenizer vocabulary into the byte/string representation expected by the Rust DFA.
- Location: `src/diverse_guide/vocab.py`
- Contains: `byte_symbol`, GPT-2 byte/unicode helpers, `build_reduced_vocab`, `build_token_id_map`
- Depends on: Hugging Face tokenizer API
- Used by: `DiverseRegexGuide` in `src/diverse_guide/guide_rust.py` and reference implementation in `src/diverse_guide/guide_python.py`

**Rust DFA Core:**
- Purpose: Construct minimized regex DFA, compile token-level transitions, expose transition queries, and maintain reward/penalty counters.
- Location: `regex_dfa_guide/src/diverse_guide_dfa.rs`
- Contains: `DfaIndex`, `DiverseGuideDFA`, `TokenCounts`, state/transition query methods, counter update methods
- Depends on: `regex-automata`, `rustc-hash`, `bincode`, crate errors in `regex_dfa_guide/src/error.rs`
- Used by: PyO3 bindings in `regex_dfa_guide/src/python_bindings/mod.rs`, Rust unit tests in `regex_dfa_guide/src/diverse_guide_dfa.rs`

**Python Binding Layer:**
- Purpose: Make the Rust DFA usable from Python with thread-safe shared access and serialization support.
- Location: `regex_dfa_guide/src/python_bindings/mod.rs`, `regex_dfa_guide/python/regex_dfa_guide/__init__.py`, `regex_dfa_guide/python/regex_dfa_guide/diverse_guide_dfa_rs.pyi`
- Contains: `PyDiverseGuideDFA(Arc<RwLock<DiverseGuideDFA>>)` and Python-visible methods mirroring Rust queries and counters
- Depends on: PyO3 and bincode
- Used by: `src/diverse_guide/guide_rust.py`, `src/diverse_guide/evaluation/metrics.py`, `scripts/metrics_eval.py`, `tests/test_dfa.py`

**Evaluation Library:**
- Purpose: Provide reusable metric functions without argument parsing or experiment filesystem policy.
- Location: `src/diverse_guide/evaluation/`
- Contains: DFA coverage metrics in `metrics.py`, path construction in `paths.py`, perplexity in `perplexity.py`, string-kernel dispatcher in `string_kernel.py`
- Depends on: NumPy, SciPy, PyTorch/Transformers for perplexity, optional C/OpenMP library via `ctypes`
- Used by: `scripts/metrics_eval.py`, `scripts/generate_re.py`, tests in `tests/test_metrics.py`, `tests/test_string_kernel.py`, `tests/test_perplexity.py`

**CLI/Reproduction Scripts:**
- Purpose: Own model loading, argument parsing, filesystem reads/writes, structured result schemas, and table collection.
- Location: `scripts/`
- Contains: `generate_re.py`, `metrics_eval.py`, `eval_runtime.py`, `collect_results.py`, `repro_results.py`, build/profiling helpers
- Depends on: Package API, evaluation library, Hugging Face Transformers, PyTorch, filesystem data under `data/` and `results/`
- Used by: Poe tasks in `pyproject.toml`, shell/PowerShell wrappers in `experiments/`, tests under `tests/test_*_output.py`

**Experiment Orchestration:**
- Purpose: Provide cross-platform reproducibility entry points for the artifact.
- Location: `experiments/`, `experiments/case_study/`
- Contains: numbered shell/PowerShell workflows, shared helpers, isolated case-study project
- Depends on: Root scripts, root package environment, case-study-local `experiments/case_study/pyproject.toml`
- Used by: Reviewers and reproduction workflows documented in `REPRODUCIBILITY.md` and `experiments/README.md`

**Artifacts and Generated Outputs:**
- Purpose: Store generated samples, structured results, reference CSVs, and release artifacts.
- Location: `data/`, `results/`, `artifact-results/`
- Contains: generated JSON samples, result JSON, collected CSV tables, tracked reference CSVs
- Depends on: Script output schemas in `scripts/generate_re.py`, `scripts/metrics_eval.py`, `scripts/eval_runtime.py`, `scripts/collect_results.py`
- Used by: Reproduction comparison flows in `REPRODUCIBILITY.md`

## Data Flow

### Primary Generation Path

1. Caller imports the public API from `src/diverse_guide/__init__.py:1`.
2. Caller constructs `DiverseGuide(model, tokenizer, regex_str, ...)` or `baseline_regex(...)` in `src/diverse_guide/guide_rust.py:436` and `src/diverse_guide/guide_rust.py:518`.
3. `DiverseRegexGuide` converts tokenizer vocabulary with `build_token_id_map()` and builds `DiverseGuideDFA` in `src/diverse_guide/guide_rust.py:25`.
4. Rust `DfaIndex::new()` builds a minimized byte DFA, token transitions, token paths, and allowed-token tables in `regex_dfa_guide/src/diverse_guide_dfa.rs:34`.
5. `StatefulSequenceGeneratorAdapter.__call__()` or `generate_batch()` resets local counters and invokes `model.generate()` with a `LogitsProcessorList` in `src/diverse_guide/guide_rust.py:359` and `src/diverse_guide/guide_rust.py:387`.
6. At each decoding step, `DiverseRegexLogitsProcessor.__call__()` advances DFA states, masks invalid logits, computes reward/penalty counts, and writes adjusted scores in `src/diverse_guide/guide_rust.py:246`.
7. After batch generation, `generate_batch()` decodes strings and updates the global path counter with `update_path_counter()` in `src/diverse_guide/guide_rust.py:416`.
8. For single calls, caller code must update global history through `StatefulSequenceGeneratorAdapter.update_generated_content()` in `src/diverse_guide/guide_rust.py:425`.

### Baseline Generation Path

1. Caller constructs `baseline_regex()` in `src/diverse_guide/guide_rust.py:518`.
2. The helper builds `RegexMaskLogitsProcessor` in `src/diverse_guide/guide_rust.py:68`.
3. `RegexMaskLogitsProcessor.__call__()` advances DFA state and preserves only allowed token logits in `src/diverse_guide/guide_rust.py:111`.
4. No diversity reward, local penalty, or global path counter adjustment is applied by this path.

### Sample Generation CLI Flow

1. `scripts/generate_re.py:153` parses grammar/model/sampling arguments.
2. Built-in grammar regexes and prompts come from `GRAMMAR_REGEX` and `GRAMMAR_PROMPT` in `scripts/generate_re.py:19`.
3. The script loads `AutoModelForCausalLM` and `AutoTokenizer` in `scripts/generate_re.py:173`.
4. It selects `baseline_regex()` or `DiverseGuide()` in `scripts/generate_re.py:187`.
5. It generates batches with `generator.generate_batch()` and validates samples against the Python regex in `scripts/generate_re.py:229`.
6. It writes `GeneratedData.to_dict()` JSON under an explicit output path or `get_data_dir_path(args) / <grammar>.json` in `scripts/generate_re.py:210`.

### Diversity Evaluation Flow

1. `scripts/metrics_eval.py:247` reads generated sample JSON from `get_data_dir_path(args) / <grammar>.json`.
2. It constructs a regex-only `DiverseGuideDFA(regex, 2**32 - 1, {})` for coverage metrics in `scripts/metrics_eval.py:255`.
3. `build_metrics_result()` combines metadata and computed metrics in `scripts/metrics_eval.py:129`.
4. Metric functions in `src/diverse_guide/evaluation/metrics.py` query DFA states/transitions and call the active string-kernel backend from `src/diverse_guide/evaluation/string_kernel.py`.
5. Structured results are written only when `--output` is provided, through `write_metrics_result()` in `scripts/metrics_eval.py:194`.

### Runtime Evaluation Flow

1. `scripts/eval_runtime.py:175` parses grammar/model/baseline arguments.
2. The script can select internal mask-only baseline or optional Outlines baseline in `scripts/eval_runtime.py:201`.
3. Generation calls are timed in a loop until target token count is reached in `scripts/eval_runtime.py:219`.
4. `build_runtime_result()` writes schema-versioned runtime metrics and metadata in `scripts/eval_runtime.py:80`.

### Native String-Kernel Flow

1. `src/diverse_guide/evaluation/string_kernel.py:32` derives platform-specific library names and candidate paths.
2. It tries package-native, build-native, and legacy native paths in `src/diverse_guide/evaluation/string_kernel.py:37`.
3. If a C/OpenMP library loads, public functions dispatch to `_wd_shift_kernel_c()`, `_compute_wd_kernel_matrix_c()`, and `_compute_wd_kernel_multiple_c()` in `src/diverse_guide/evaluation/string_kernel.py:105`.
4. Otherwise, public names are imported from `src/diverse_guide/evaluation/string_kernel_py.py` in `src/diverse_guide/evaluation/string_kernel.py:196`.

**State Management:**
- Global diversity history is stored in Rust `DiverseGuideDFA.path_counter` in `regex_dfa_guide/src/diverse_guide_dfa.rs:379` and updated after completed generations.
- Per-generation-call penalty state is stored in Rust `DiverseGuideDFA.local_state_counter` in `regex_dfa_guide/src/diverse_guide_dfa.rs:380` and reset before each `model.generate()`.
- Python logits processors track per-row DFA state and how many generated tokens have already been processed in `_guide_states_by_row` and `_processed_lengths_by_row` in `src/diverse_guide/guide_rust.py:203`.
- PyO3 wraps the Rust object with `Arc<RwLock<DiverseGuideDFA>>` in `regex_dfa_guide/src/python_bindings/mod.rs:19`.

## Key Abstractions

**`DiverseGuide`:**
- Purpose: Canonical public generator for diverse regex-constrained generation.
- Examples: `src/diverse_guide/guide_rust.py`, `examples/01_quickstart.py`, `scripts/generate_re.py`
- Pattern: Thin public subclass of `StatefulSequenceGeneratorAdapter` that wires a `DiverseRegexLogitsProcessor`.

**`StatefulSequenceGeneratorAdapter`:**
- Purpose: Wrap a Hugging Face model/tokenizer/logits processor and expose single/batch generation methods.
- Examples: `src/diverse_guide/guide_rust.py:310`
- Pattern: Adapter around `model.generate()` with reset-before-call behavior and explicit global-counter updates.

**`DiverseRegexLogitsProcessor`:**
- Purpose: Enforce regex constraints and apply diversity adjustments during token generation.
- Examples: `src/diverse_guide/guide_rust.py:138`
- Pattern: Hugging Face `LogitsProcessor` implementation with mutable per-row state.

**`RegexMaskLogitsProcessor`:**
- Purpose: Enforce the same regex constraint without diversity adjustment for baseline comparisons.
- Examples: `src/diverse_guide/guide_rust.py:68`
- Pattern: Mask-only `LogitsProcessor`, preserving baseline runtime comparability.

**`DfaIndex`:**
- Purpose: Immutable regex DFA, byte transitions, token transitions, token-state paths, and allowed token tables.
- Examples: `regex_dfa_guide/src/diverse_guide_dfa.rs:23`
- Pattern: Expensive construction data behind `Arc` sharing.

**`DiverseGuideDFA`:**
- Purpose: Runtime DFA object combining shared `DfaIndex` with mutable diversity counters.
- Examples: `regex_dfa_guide/src/diverse_guide_dfa.rs:377`, `regex_dfa_guide/src/python_bindings/mod.rs:19`
- Pattern: Shared immutable index plus mutable counters; `fork()` creates fresh counters while sharing the index.

**`GeneratedData`:**
- Purpose: JSON schema for generated sample files.
- Examples: `scripts/generate_re.py:34`
- Pattern: Dataclass with `to_dict()` and `from_dict()` helpers for script/test symmetry.

**Structured Result Dictionaries:**
- Purpose: JSON schema for diversity and runtime outputs.
- Examples: `scripts/metrics_eval.py:129`, `scripts/eval_runtime.py:80`, `scripts/collect_results.py:80`
- Pattern: Plain dictionaries with `schema_version`, `experiment`, `parameters`, `metadata`, and domain-specific result sections.

## Entry Points

**Python package import:**
- Location: `src/diverse_guide/__init__.py`
- Triggers: `from diverse_guide import DiverseGuide, baseline_regex`
- Responsibilities: Export only stable package names.

**Rust extension import:**
- Location: `regex_dfa_guide/python/regex_dfa_guide/__init__.py`
- Triggers: `from regex_dfa_guide import DiverseGuideDFA`
- Responsibilities: Re-export the PyO3 module class.

**Sample generation CLI:**
- Location: `scripts/generate_re.py`
- Triggers: `uv run poe gen ...` or `uv run python scripts/generate_re.py ...`
- Responsibilities: Generate and persist sample JSON.

**Diversity evaluation CLI:**
- Location: `scripts/metrics_eval.py`
- Triggers: `uv run poe eval ...` or experiment wrappers.
- Responsibilities: Compute metrics from generated sample JSON and optionally write structured result JSON.

**Runtime evaluation CLI:**
- Location: `scripts/eval_runtime.py`
- Triggers: `uv run poe eval-runtime ...` or experiment wrappers.
- Responsibilities: Measure token throughput for diverse and baseline generation.

**Results collection CLI:**
- Location: `scripts/collect_results.py`
- Triggers: `uv run python scripts/collect_results.py`
- Responsibilities: Validate result inputs and write CSV tables under `results/tables/`.

**Native kernel build CLI:**
- Location: `scripts/build_wd_kernel.py`
- Triggers: `uv run python scripts/build_wd_kernel.py`
- Responsibilities: Build/check/copy optional WD-shift C/OpenMP library.

**Experiment wrappers:**
- Location: `experiments/1_generation.sh`, `experiments/2_diversity_evaluation.sh`, `experiments/3_efficiency_evaluation.sh`, `experiments/4_temperature_ablation.sh`, `experiments/5_component_ablation.sh`, and matching `.ps1` files
- Triggers: Direct shell/PowerShell execution.
- Responsibilities: Cross-platform reproduction orchestration.

**CI:**
- Location: `.github/workflows/test.yml`
- Triggers: Push and pull request to `master`/`main`.
- Responsibilities: Build Rust extension, run Python tests, run Rust tests, lint, and parse PowerShell experiment scripts.

## Architectural Constraints

- **Python version:** Runtime package targets Python 3.12+ in `pyproject.toml`.
- **Rust extension:** The `regex-dfa-guide` path dependency must be buildable with maturin; source lives in `regex_dfa_guide/` and Python imports expect `regex_dfa_guide.regex_dfa_guide_rs`.
- **Threading:** Python generation follows Hugging Face/PyTorch execution. The Rust Python binding uses `Arc<RwLock<DiverseGuideDFA>>`; read methods take shared locks and counter mutations take write locks in `regex_dfa_guide/src/python_bindings/mod.rs`.
- **Global state:** `src/diverse_guide/evaluation/string_kernel.py` selects the active kernel backend at import time. `src/diverse_guide/vocab.py` caches tokenizer conversions with `@lru_cache`.
- **Mutable generation state:** Reuse `DiverseGuide` objects deliberately; cross-call diversity depends on retained `path_counter`. Callers using `__call__()` directly must call `update_generated_content()` after successful generation.
- **Model I/O:** CLI scripts load Hugging Face models inside `main()` functions. Keep tests model-free unless specifically testing model-backed behavior.
- **Filesystem outputs:** Generated data goes under `data/`; result JSON and collected CSVs go under `results/`; tracked reference CSVs go under `artifact-results/`.
- **Case study isolation:** `experiments/case_study/` has its own `pyproject.toml`, `uv.lock`, test harness, and fetched package snapshots. Treat it as an isolated reproduction subproject.
- **Circular imports:** No package-level circular imports are evident in `src/diverse_guide/`; scripts use fallback imports to work both as direct scripts and package-imported test targets.

## Anti-Patterns

### Importing Implementation Internals From User Code

**What happens:** Caller code imports directly from `src/diverse_guide/guide_rust.py` or `regex_dfa_guide.regex_dfa_guide_rs`.
**Why it's wrong:** It bypasses the stable public namespace and couples user code to implementation details that tests treat as internal.
**Do this instead:** Import supported API names from `src/diverse_guide/__init__.py`, for example `from diverse_guide import DiverseGuide, baseline_regex`.

### Adding Filesystem Logic To Metric Functions

**What happens:** New metrics read or write files directly inside `src/diverse_guide/evaluation/metrics.py`.
**Why it's wrong:** The metrics module is pure and is tested directly; filesystem policy belongs in CLI scripts.
**Do this instead:** Put pure computation in `src/diverse_guide/evaluation/metrics.py` and route file loading/output through `scripts/metrics_eval.py`.

### Making The Python Reference Path The Default

**What happens:** New public API code uses `src/diverse_guide/guide_python.py` for normal generation.
**Why it's wrong:** The default implementation is Rust-backed for performance and is what packaging, examples, and experiments exercise.
**Do this instead:** Extend `src/diverse_guide/guide_rust.py` and keep `src/diverse_guide/guide_python.py` aligned only where reference/equivalence tests require it.

### Writing Reproduction Results Outside Standard Trees

**What happens:** New scripts write generated samples or metrics to ad hoc directories.
**Why it's wrong:** `scripts/collect_results.py`, `experiments/README.md`, and reference artifact comparison assume standard `data/`, `results/`, and `artifact-results/` layouts.
**Do this instead:** Use `src/diverse_guide/evaluation/paths.py` for generated data paths and mirror the result schema patterns in `scripts/metrics_eval.py` and `scripts/eval_runtime.py`.

## Error Handling

**Strategy:** Validate user-facing arguments early, surface Rust DFA errors as Python exceptions through PyO3, and keep script failures explicit with raised exceptions or non-zero exit codes.

**Patterns:**
- Rust DFA construction and transition queries return `Result<T>` with domain errors from `regex_dfa_guide/src/error.rs`.
- PyO3 bindings convert Rust errors into Python errors through `.map_err(Into::into)` and explicit `PyValueError`/`PyRuntimeError` mapping in `regex_dfa_guide/src/python_bindings/mod.rs`.
- Python CLI scripts validate invalid combinations and raise `ValueError` or `parser.error()` in `scripts/generate_re.py`, `scripts/metrics_eval.py`, and `scripts/eval_runtime.py`.
- Native C kernel wrappers translate non-zero statuses into `ValueError`, `MemoryError`, or `OSError` in `src/diverse_guide/evaluation/string_kernel.py`.
- Result collection reports missing inputs through `MissingResultInputsError` and exits with status 1 in `scripts/collect_results.py`.

## Cross-Cutting Concerns

**Logging:** The code uses `print()` for CLI progress and status in `scripts/`; the library modules avoid logging frameworks.

**Validation:** Regex/sample validation is split between Python regex checks in `scripts/generate_re.py` and DFA transition validation in `regex_dfa_guide/src/diverse_guide_dfa.rs`.

**Authentication:** Not applicable for the package. Hugging Face model access uses the user environment outside this codebase.

**Packaging:** The root package is configured in `pyproject.toml`; the Rust extension package is configured in `regex_dfa_guide/pyproject.toml` and `regex_dfa_guide/Cargo.toml`.

**Reproducibility Metadata:** Structured result JSON includes git commit, Python version, platform, and timestamp via `scripts/repro_results.py`.

**Native Optionality:** The WD-shift C/OpenMP kernel is optional; generation does not depend on it, and evaluation falls back to Python/NumPy automatically.

---

*Architecture analysis: 2026-05-19*
