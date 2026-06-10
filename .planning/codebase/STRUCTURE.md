# Codebase Structure

**Analysis Date:** 2026-05-19

## Directory Layout

```text
diverse-dfa-gen/
├── src/diverse_guide/              # Main Python package
│   ├── __init__.py                 # Public API exports
│   ├── guide_rust.py               # Default Rust-backed generation path
│   ├── guide_python.py             # Pure-Python reference implementation
│   ├── vocab.py                    # Tokenizer vocabulary conversion
│   ├── evaluation/                 # Reusable metric/perplexity/path helpers
│   └── _native/                    # Optional packaged native metric library target
├── regex_dfa_guide/                # Local Rust/PyO3 extension package
│   ├── src/                        # Rust DFA core and Python bindings
│   ├── python/regex_dfa_guide/     # Python shim and type marker/stub
│   ├── Cargo.toml                  # Rust crate config
│   └── pyproject.toml              # maturin Python package config
├── native/wd_kernel/               # Optional C/OpenMP WD-shift kernel source
├── scripts/                        # Python CLIs and build/profiling helpers
├── experiments/                    # Shell/PowerShell reproduction wrappers
│   └── case_study/                 # Isolated optional case-study project
├── tests/                          # Root Python test suite
├── examples/                       # User-facing API examples
├── docs/                           # Developer and architecture documentation
├── data/                           # Generated sample datasets
├── results/                        # Generated structured result JSON and CSVs
├── artifact-results/               # Tracked reference CSV artifacts
├── archive/                        # Deprecated or historical material
├── .github/workflows/              # CI workflows
├── pyproject.toml                  # Root package, tasks, lint, and test config
├── uv.lock                         # Root dependency lockfile
├── README.md                       # User setup and API guide
└── REPRODUCIBILITY.md              # Reviewer-facing reproduction guide
```

## Directory Purposes

**`src/diverse_guide/`:**
- Purpose: Main installable Python package for DiverseGuide.
- Contains: Public API exports, generation adapters/processors, tokenizer vocabulary utilities, reusable evaluation helpers.
- Key files: `src/diverse_guide/__init__.py`, `src/diverse_guide/guide_rust.py`, `src/diverse_guide/guide_python.py`, `src/diverse_guide/vocab.py`

**`src/diverse_guide/evaluation/`:**
- Purpose: Pure or mostly pure reusable evaluation utilities.
- Contains: DFA metrics, WD-shift string kernel dispatch, NumPy fallback kernel, perplexity calculation, output path construction.
- Key files: `src/diverse_guide/evaluation/metrics.py`, `src/diverse_guide/evaluation/string_kernel.py`, `src/diverse_guide/evaluation/string_kernel_py.py`, `src/diverse_guide/evaluation/perplexity.py`, `src/diverse_guide/evaluation/paths.py`

**`src/diverse_guide/_native/`:**
- Purpose: Optional package-native destination for compiled WD-shift kernel binaries.
- Contains: `.gitkeep`; build helper can copy `wd_kernel.so` or `wd_kernel.dll` here.
- Key files: `src/diverse_guide/_native/.gitkeep`

**`regex_dfa_guide/`:**
- Purpose: Local Rust extension package that provides DFA construction, token-transition compilation, and diversity counter logic.
- Contains: Rust crate files, maturin config, Python shim package, typed stub.
- Key files: `regex_dfa_guide/Cargo.toml`, `regex_dfa_guide/pyproject.toml`, `regex_dfa_guide/src/lib.rs`, `regex_dfa_guide/src/diverse_guide_dfa.rs`, `regex_dfa_guide/src/python_bindings/mod.rs`

**`regex_dfa_guide/src/`:**
- Purpose: Rust implementation for the DFA backend.
- Contains: Crate root, error types, DFA core, PyO3 bindings.
- Key files: `regex_dfa_guide/src/lib.rs`, `regex_dfa_guide/src/error.rs`, `regex_dfa_guide/src/diverse_guide_dfa.rs`, `regex_dfa_guide/src/python_bindings/mod.rs`

**`regex_dfa_guide/python/regex_dfa_guide/`:**
- Purpose: Python package shim for the compiled PyO3 module.
- Contains: Import/export wrapper, `py.typed`, and `.pyi` type stub.
- Key files: `regex_dfa_guide/python/regex_dfa_guide/__init__.py`, `regex_dfa_guide/python/regex_dfa_guide/diverse_guide_dfa_rs.pyi`, `regex_dfa_guide/python/regex_dfa_guide/py.typed`

**`native/wd_kernel/`:**
- Purpose: Source for optional C/OpenMP acceleration of evaluation string-kernel metrics.
- Contains: C source, Makefile, vendored hash header.
- Key files: `native/wd_kernel/wd_kernel.c`, `native/wd_kernel/Makefile`, `native/wd_kernel/uthash.h`

**`scripts/`:**
- Purpose: Python CLI entry points, result conversion, metadata, native build helpers, and profiling/benchmarking utilities.
- Contains: Generation, metric evaluation, runtime evaluation, result collection, reproducibility metadata, WD-kernel build scripts.
- Key files: `scripts/generate_re.py`, `scripts/metrics_eval.py`, `scripts/eval_runtime.py`, `scripts/collect_results.py`, `scripts/repro_results.py`, `scripts/build_wd_kernel.py`, `scripts/benchmark_wd_kernel.py`, `scripts/profile_runtime_processor.py`

**`experiments/`:**
- Purpose: Cross-platform artifact reproduction workflows.
- Contains: Numbered shell scripts and matching PowerShell scripts for generation, diversity evaluation, efficiency evaluation, temperature ablation, and component ablation.
- Key files: `experiments/README.md`, `experiments/common.sh`, `experiments/common.ps1`, `experiments/1_generation.sh`, `experiments/1_generation.ps1`, `experiments/2_diversity_evaluation.sh`, `experiments/3_efficiency_evaluation.sh`, `experiments/4_temperature_ablation.sh`, `experiments/5_component_ablation.sh`

**`experiments/case_study/`:**
- Purpose: Isolated optional impact case study with its own environment and test harness.
- Contains: Case-study package config, data inputs, tests, package-fetch scripts, report generator.
- Key files: `experiments/case_study/pyproject.toml`, `experiments/case_study/uv.lock`, `experiments/case_study/README.md`, `experiments/case_study/run_case_study.sh`, `experiments/case_study/run_case_study.ps1`, `experiments/case_study/report_case_study.py`, `experiments/case_study/tests/`

**`tests/`:**
- Purpose: Root Python verification suite.
- Contains: Unit and integration-style tests for public API, Rust bindings, metrics, scripts, packaging, release metadata, and reference results.
- Key files: `tests/conftest.py`, `tests/test_guide.py`, `tests/test_guide_equivalence.py`, `tests/test_dfa.py`, `tests/test_vocab.py`, `tests/test_metrics.py`, `tests/test_string_kernel.py`, `tests/test_collect_results.py`, `tests/test_packaging_imports.py`

**`examples/`:**
- Purpose: Small user-facing examples for API usage and DFA inspection.
- Contains: Quickstart, batch generation, custom regex, DFA inspection.
- Key files: `examples/01_quickstart.py`, `examples/02_batch_generation.py`, `examples/03_custom_regex.py`, `examples/04_dfa_inspection.py`

**`docs/`:**
- Purpose: Project documentation beyond README/reproducibility guides.
- Contains: Architecture and development notes plus planning/spec artifacts under `docs/superpowers/`.
- Key files: `docs/architecture.md`, `docs/developing.md`, `docs/superpowers/specs/`, `docs/superpowers/plans/`

**`data/`:**
- Purpose: Generated sample JSON data consumed by metric evaluation.
- Contains: Baseline and diverse generated samples grouped by model/setting.
- Key files: `data/diverse/`, `data/baseline/`

**`results/`:**
- Purpose: Generated structured result JSON and collected CSV tables.
- Contains: Diversity, runtime, temperature ablation, component ablation, and table output trees.
- Key files: `results/diversity/`, `results/runtime/`, `results/temperature_ablation/`, `results/component_ablation/`, `results/tables/`

**`artifact-results/`:**
- Purpose: Tracked reference CSV outputs for the v0.2.0 artifact.
- Contains: Primary and optional reproduction result summaries.
- Key files: `artifact-results/v0.2.0/README.md`, `artifact-results/v0.2.0/primary/diversity.csv`, `artifact-results/v0.2.0/primary/runtime.csv`, `artifact-results/v0.2.0/optional/temperature_ablation.csv`, `artifact-results/v0.2.0/optional/component_ablation.csv`

**`archive/`:**
- Purpose: Deprecated and historical material retained outside active package/test paths.
- Contains: Old implementations, data archives, result notes.
- Key files: `archive/NOTE.md`, `archive/deprecated/`

**`.github/workflows/`:**
- Purpose: GitHub Actions CI configuration.
- Contains: Python/Rust/lint workflow and PowerShell script parsing checks.
- Key files: `.github/workflows/test.yml`

## Key File Locations

**Entry Points:**
- `src/diverse_guide/__init__.py`: Public import namespace for package users.
- `src/diverse_guide/guide_rust.py`: Default implementation of `DiverseGuide`, `baseline_regex`, logits processors, and generation adapter.
- `regex_dfa_guide/python/regex_dfa_guide/__init__.py`: Python import shim for the compiled Rust extension.
- `scripts/generate_re.py`: CLI for generating sample data.
- `scripts/metrics_eval.py`: CLI for diversity/coverage/perplexity evaluation.
- `scripts/eval_runtime.py`: CLI for runtime throughput evaluation.
- `scripts/collect_results.py`: CLI for collecting structured result JSON into CSV tables.
- `scripts/build_wd_kernel.py`: CLI for optional native WD-kernel build/check/copy.
- `experiments/*.sh` and `experiments/*.ps1`: Cross-platform reproduction workflows.

**Configuration:**
- `pyproject.toml`: Root Python package metadata, dependencies, `poe` tasks, pytest config, Ruff config, setuptools package discovery, uv workspace config.
- `uv.lock`: Locked root Python dependency resolution.
- `regex_dfa_guide/Cargo.toml`: Rust crate dependencies, features, crate type, release profile, Rust version.
- `regex_dfa_guide/pyproject.toml`: maturin build configuration for the Python extension package.
- `.github/workflows/test.yml`: CI workflow for Python tests, Rust tests, linting, extension build, and PowerShell parsing.
- `experiments/case_study/pyproject.toml`: Isolated case-study environment.

**Core Logic:**
- `src/diverse_guide/guide_rust.py`: Runtime generation and diversity adjustment.
- `src/diverse_guide/vocab.py`: Tokenizer vocabulary normalization.
- `regex_dfa_guide/src/diverse_guide_dfa.rs`: DFA construction, transition tables, counters, count computation.
- `regex_dfa_guide/src/python_bindings/mod.rs`: Python binding layer.
- `src/diverse_guide/evaluation/metrics.py`: DFA and string diversity metrics.
- `src/diverse_guide/evaluation/string_kernel.py`: Native/fallback WD-kernel selection.

**Testing:**
- `tests/test_guide.py`: Logits processors and generation adapter behavior.
- `tests/test_guide_equivalence.py`: Pure-Python and Rust-backed guide equivalence.
- `tests/test_dfa.py`: Rust DFA behavior through Python bindings.
- `tests/test_vocab.py`: Tokenizer vocabulary conversion.
- `tests/test_metrics.py`: Metric functions.
- `tests/test_string_kernel.py`: C/Python string-kernel backend behavior.
- `tests/test_metrics_eval_output.py`: Structured diversity result output.
- `tests/test_eval_runtime_output.py`: Structured runtime result output.
- `tests/test_collect_results.py`: CSV collection behavior.
- `regex_dfa_guide/src/diverse_guide_dfa.rs`: Rust unit tests for DFA internals.

**Documentation:**
- `README.md`: User setup, API, examples, tests, and project overview.
- `REPRODUCIBILITY.md`: Reviewer-facing reproduction protocol.
- `experiments/README.md`: Detailed experiment command reference.
- `docs/architecture.md`: Human architecture overview.
- `docs/developing.md`: Development guidance.

## Naming Conventions

**Files:**
- Python package modules use lowercase snake_case: `src/diverse_guide/guide_rust.py`, `src/diverse_guide/evaluation/string_kernel_py.py`.
- Script files use lowercase snake_case with action-oriented names: `scripts/generate_re.py`, `scripts/collect_results.py`, `scripts/build_wd_kernel.py`.
- Tests use `test_<subject>.py`: `tests/test_guide.py`, `tests/test_metrics_eval_output.py`.
- Experiment wrappers use numeric ordering plus descriptive names: `experiments/1_generation.sh`, `experiments/4_temperature_ablation.ps1`.
- Rust modules use snake_case: `regex_dfa_guide/src/diverse_guide_dfa.rs`, `regex_dfa_guide/src/python_bindings/mod.rs`.
- Reference result files are lowercase CSVs under release/version folders: `artifact-results/v0.2.0/primary/runtime.csv`.

**Directories:**
- Installable Python package code lives under `src/<package_name>/`, here `src/diverse_guide/`.
- Rust extension package owns its own root directory, here `regex_dfa_guide/`.
- Reproduction orchestration uses top-level `experiments/`, with isolated subprojects nested under it.
- Generated sample data uses `data/<setting>/<model-or-setting>/`.
- Structured result outputs use `results/<experiment>/<setting>/...`.

**Classes and Types:**
- Public Python API classes use PascalCase: `DiverseGuide`, `DiverseRegexLogitsProcessor`, `StatefulSequenceGeneratorAdapter`.
- Rust public structs use PascalCase: `DfaIndex`, `DiverseGuideDFA`, `TokenCounts`.
- Python dataclasses use PascalCase: `GeneratedData`.

**Functions and Variables:**
- Python functions use snake_case: `baseline_regex`, `build_token_id_map`, `compute_metrics`, `collect_all`.
- Rust functions/methods use snake_case: `get_allowed_token_ids`, `update_path_counter`, `compute_counts`.
- Constants use uppercase snake_case: `GRAMMAR_REGEX`, `GRAMMAR_PROMPT`, `METRIC_FIELDS`, `STRING_KERNEL_BACKEND`.

## Where to Add New Code

**New Public Generation Feature:**
- Primary code: `src/diverse_guide/guide_rust.py`
- Public export: `src/diverse_guide/__init__.py`
- Reference/equivalence code when needed: `src/diverse_guide/guide_python.py`
- Tests: `tests/test_guide.py` and `tests/test_guide_equivalence.py`
- Use this when adding new logits processor behavior, generation adapter options, or public generator helpers.

**New DFA Backend Capability:**
- Rust implementation: `regex_dfa_guide/src/diverse_guide_dfa.rs`
- Rust error additions: `regex_dfa_guide/src/error.rs`
- Python binding: `regex_dfa_guide/src/python_bindings/mod.rs`
- Python type stub: `regex_dfa_guide/python/regex_dfa_guide/diverse_guide_dfa_rs.pyi`
- Tests: Rust unit tests in `regex_dfa_guide/src/diverse_guide_dfa.rs` and Python binding tests in `tests/test_dfa.py`

**New Tokenizer Vocabulary Handling:**
- Implementation: `src/diverse_guide/vocab.py`
- Tests: `tests/test_vocab.py`, plus generation/guide tests if behavior affects valid token transitions.
- Keep vocabulary changes cached and compatible with Hugging Face tokenizer APIs.

**New Evaluation Metric:**
- Pure computation: `src/diverse_guide/evaluation/metrics.py`
- CLI wiring/output schema: `scripts/metrics_eval.py`
- CSV collection fields: `scripts/collect_results.py`
- Tests: `tests/test_metrics.py`, `tests/test_metrics_eval_output.py`, `tests/test_collect_results.py`

**New String-Kernel Backend Behavior:**
- Dispatcher logic: `src/diverse_guide/evaluation/string_kernel.py`
- Python fallback changes: `src/diverse_guide/evaluation/string_kernel_py.py`
- Native C changes: `native/wd_kernel/wd_kernel.c`
- Build helper updates: `scripts/build_wd_kernel.py`
- Tests: `tests/test_string_kernel.py`

**New CLI Script:**
- Implementation: `scripts/<action_name>.py`
- Shared metadata: reuse `scripts/repro_results.py`
- Shared data path policy: reuse or extend `src/diverse_guide/evaluation/paths.py`
- Tests: add `tests/test_<action_name>.py` or `tests/test_<action_name>_output.py`
- Poe task: add to `[tool.poe.tasks]` in `pyproject.toml` if it is a common command.

**New Experiment Workflow:**
- Linux/macOS wrapper: `experiments/<n>_<name>.sh`
- Windows wrapper: `experiments/<n>_<name>.ps1`
- Shared shell helpers: `experiments/common.sh`
- Shared PowerShell helpers: `experiments/common.ps1`
- Documentation: `experiments/README.md` and `REPRODUCIBILITY.md`
- CI parse coverage: `.github/workflows/test.yml` already parses `experiments/**/*.ps1`

**New Example:**
- Implementation: `examples/<nn>_<topic>.py`
- Documentation reference: `README.md`
- Keep examples importing from `diverse_guide`, not from implementation modules.

**New Case-Study Code:**
- Implementation: `experiments/case_study/`
- Tests: `experiments/case_study/tests/`
- Documentation: `experiments/case_study/README.md`
- Keep dependencies in `experiments/case_study/pyproject.toml`, not the root package, unless the dependency is needed by the main package.

**New Reference Artifact:**
- Primary reference CSV: `artifact-results/v0.2.0/primary/`
- Optional reference CSV: `artifact-results/v0.2.0/optional/`
- Documentation: `artifact-results/v0.2.0/README.md`, `REPRODUCIBILITY.md`, `experiments/README.md`

**Utilities:**
- Reusable package helper: place under `src/diverse_guide/` or `src/diverse_guide/evaluation/` based on whether it is runtime generation or evaluation-specific.
- Script-only helper: place under `scripts/`.
- Test-only helper: place in `tests/conftest.py` or the relevant `tests/test_*.py` file.

## Special Directories

**`build/`:**
- Purpose: Local build output for the optional native WD-kernel and Python build artifacts.
- Generated: Yes
- Committed: No for normal generated artifacts; treat contents as disposable.

**`dist/`:**
- Purpose: Python distribution artifacts.
- Generated: Yes
- Committed: No for normal generated artifacts.

**`.venv/`:**
- Purpose: Local Python virtual environment.
- Generated: Yes
- Committed: No.

**`.pytest_cache/` and `.ruff_cache/`:**
- Purpose: Tool caches for pytest and Ruff.
- Generated: Yes
- Committed: No.

**`regex_dfa_guide/target/`:**
- Purpose: Rust build output for the extension crate.
- Generated: Yes
- Committed: No.

**`regex_dfa_guide/build/` and `regex_dfa_guide/python/minimal_dfa.egg-info/`:**
- Purpose: Build/package byproducts inside the Rust extension package.
- Generated: Yes
- Committed: No for normal build products.

**`.worktrees/`:**
- Purpose: Local auxiliary worktrees.
- Generated: Yes
- Committed: No.

**`data/`:**
- Purpose: Generated sample data used by evaluation scripts.
- Generated: Yes, but current repository contains sample/result data trees.
- Committed: Mixed; avoid adding large generated datasets unless they are intended artifacts.

**`results/`:**
- Purpose: Generated evaluation output and collected tables.
- Generated: Yes, but current repository contains result trees.
- Committed: Mixed; prefer `artifact-results/` for curated reference outputs.

**`artifact-results/`:**
- Purpose: Curated, tracked release reference artifacts.
- Generated: From reproduction pipeline, then curated.
- Committed: Yes.

**`archive/`:**
- Purpose: Historical/deprecated code and compressed legacy data.
- Generated: No for active workflows.
- Committed: Yes; do not add new active code here.

**`docs/superpowers/`:**
- Purpose: Planning/spec documentation artifacts.
- Generated: No for runtime package behavior.
- Committed: Yes.

---

*Structure analysis: 2026-05-19*
