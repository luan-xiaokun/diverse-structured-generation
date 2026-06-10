# Architecture

This document explains how the software is organized internally. For setup and
user-facing commands, see `../README.md`. For paper-result reproduction, see
`../REPRODUCIBILITY.md`.

## System Overview

The project has four implementation layers:

| Layer | Location | Responsibility |
| --- | --- | --- |
| Python generation API | `src/diverse_guide/` | Public API, Hugging Face integration, logits processing, vocabulary conversion |
| Rust DFA backend | `regex_dfa_guide/` | Regex-to-DFA construction, token transition tables, path/local counters |
| Evaluation utilities | `src/diverse_guide/evaluation/`, `scripts/` | Diversity metrics, runtime measurement, result collection |
| Optional native metric kernel | `native/wd_kernel/` | C/OpenMP acceleration for the WD-shift string kernel used by Vendi score |

The generation path depends on Python, Hugging Face Transformers, and the Rust
DFA extension. The optional C/OpenMP kernel is only used for evaluation metrics;
it is not required to generate constrained samples.

## Generation Path

The public entry points are exported from `src/diverse_guide/__init__.py`:

- `DiverseGuide(model, tokenizer, regex_str, gamma=0.5, beta=3.0, **kwargs)`
- `diverse_regex(model, tokenizer, regex_str, gamma=0.5, beta=3.0, **kwargs)` as a compatibility helper
- `baseline_regex(model, tokenizer, regex_str, **kwargs)`
- `StatefulSequenceGeneratorAdapter`
- `DiverseRegexLogitsProcessor`

The default implementation is Rust-backed:

```text
user code
  -> DiverseGuide(...) or baseline_regex(...)
  -> DiverseRegexLogitsProcessor
  -> DiverseRegexGuide
  -> regex_dfa_guide.DiverseGuideDFA
  -> StatefulSequenceGeneratorAdapter
  -> model.generate(..., logits_processor=[...])
```

`StatefulSequenceGeneratorAdapter` owns the Hugging Face model, tokenizer, and
logits processor. Its `__call__` method generates one sample. Its
`generate_batch` method generates multiple samples for the same prompt and
updates the global path counter for each completed output.

During `model.generate`, `DiverseRegexLogitsProcessor.__call__` runs at every
generation step. It:

1. Tracks the current DFA state for each generated row.
2. Asks the Rust DFA backend which token ids are valid from each state.
3. Masks invalid token logits to `-inf`.
4. Computes diversity adjustments for valid tokens.
5. Adds the adjustments back to the Hugging Face score tensor.

`baseline_regex` uses `RegexMaskLogitsProcessor`, a mask-only processor that
enforces the same regex constraint without computing the diversity reward or
local penalty.

## DFA Backend

`regex_dfa_guide/src/diverse_guide_dfa.rs` is the core Rust module. It defines:

- `DfaIndex`: immutable DFA structure, byte transitions, token transitions, and
  token-to-byte paths.
- `DiverseGuideDFA`: shared DFA index plus mutable diversity counters.
- `TokenCounts`: per-token count arrays returned to Python for logit adjustment.

The backend builds a DFA from an anchored regex and compiles tokenizer
vocabulary entries into token-level transitions. This lets the Python logits
processor work at token granularity while the automaton remains byte-based.

The Python extension binding lives in `regex_dfa_guide/src/python_bindings/`.
The typed Python stub lives in
`regex_dfa_guide/python/regex_dfa_guide/diverse_guide_dfa_rs.pyi`.

Counter responsibilities:

- Path counters record byte-transition usage across completed generated
  samples.
- Local state counters record token-path usage within one `model.generate`
  call.
- `compute_counts(state)` returns valid token ids plus reward and penalty
  count signals for the current DFA state.

`DiverseGuideDFA.fork()` creates a new DFA object that shares the immutable
index while resetting counters. Pickle support uses bincode serialization.

## Diversity Mechanism

The diversity logic is implemented in `src/diverse_guide/guide_rust.py`.

At a high level, each step combines three operations:

1. **Constraint masking**: invalid token ids are set to `-inf`.
2. **Global reward**: tokens traversing less-used DFA paths receive a larger
   boost.
3. **Local penalty**: tokens overused within the current generation call receive
   a larger penalty.

The `gamma` parameter scales the reward signal in `DiverseRegexLogitsProcessor`.
The `beta` parameter scales the local penalty. The default internal baseline is
implemented separately by `RegexMaskLogitsProcessor` so runtime comparisons do
not include diversity-count computation in the baseline path.

The ablation options in `DiverseGuide(..., ablation_component=...)` disable
individual parts of this adjustment for experiment scripts:

- `reward`
- `penalty`
- `range_scaling`

`src/diverse_guide/guide_python.py` is a pure-Python reference implementation.
It is useful for debugging and equivalence tests, but the package exports the
Rust-backed implementation by default.

## Vocabulary Conversion

`src/diverse_guide/vocab.py` converts Hugging Face tokenizer vocabularies into
the byte/string forms expected by the DFA backend.

The DFA backend needs a `token_id -> decoded string` mapping. Token decoding is
nontrivial because tokenizers may use byte-level encodings, special tokens, or
model-specific string conventions. Keep vocabulary conversion changes covered by
`tests/test_vocab.py` and guide tests.

## Evaluation Path

Experiment scripts are thin command-line wrappers around the package API.

```text
scripts/generate_re.py
  -> diverse_guide.DiverseGuide / baseline_regex
  -> data/{diverse,baseline}/{model}/{grammar}.json

scripts/metrics_eval.py
  -> diverse_guide.evaluation.metrics
  -> diverse_guide.evaluation.string_kernel
  -> results/<experiment>/.../*.json

scripts/eval_runtime.py
  -> generation API
  -> results/runtime/.../*.json

scripts/collect_results.py
  -> results/**/*.json
  -> results/tables/*.csv
```

`scripts/repro_results.py` centralizes result metadata such as timestamp, git
commit, Python version, and platform. Evaluation scripts include this metadata
in structured JSON outputs.

## WD-Shift Kernel Backend

The Vendi score metric uses a pairwise WD-shift string kernel. The selector in
`src/diverse_guide/evaluation/string_kernel.py` chooses between:

- C/OpenMP backend from `native/wd_kernel/`, loaded from
  `build/native/wd_kernel/`.
- Pure Python/NumPy fallback in
  `src/diverse_guide/evaluation/string_kernel_py.py`.

The fallback is always available. Building the C/OpenMP backend improves metric
runtime but does not change the generation API.

## Case Study

`experiments/case_study/` is intentionally isolated from the root package. It
has its own `pyproject.toml`, lock file, test harness, data files, package-fetch
scripts, and README. It measures how baseline and diverse generated datasets
affect coverage in pinned snapshots of third-party Python libraries.

## Testing Strategy

The root tests do not download language models. They use mock tokenizers and
synthetic DFA inputs.

Important test groups:

- `tests/test_dfa.py`: Rust DFA behavior through Python bindings.
- `tests/test_guide.py`: logits processor and generation adapter behavior.
- `tests/test_guide_equivalence.py`: pure-Python and Rust-backed guide
  equivalence checks.
- `tests/test_string_kernel.py`: C/Python string-kernel backend behavior and
  fallback handling.
- `tests/test_metrics*.py`, `tests/test_collect_results.py`,
  `tests/test_eval_runtime_output.py`: structured evaluation output behavior.
- `tests/test_packaging_imports.py`: public package namespace boundaries.

CI runs Python tests with coverage, Rust tests, Ruff linting, and PowerShell
experiment-script checks.
