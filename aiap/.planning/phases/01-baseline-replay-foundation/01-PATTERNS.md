# Phase 1: Baseline Replay Foundation - Pattern Map

**Mapped:** 2026-07-23  
**Files analyzed:** 14 planned files/edits  
**Analogs found:** 11 / 14

## File Classification

| New/Modified File | Role | Data Flow | Closest Analog | Match Quality |
|---|---|---|---|---|
| `pyproject.toml` | config | package/build | `../pyproject.toml` | partial |
| `aiap/__init__.py` | package API | import | `../src/diverse_guide/__init__.py` | role-match |
| `aiap/__main__.py` | entry point | request-response | `../scripts/metrics_eval.py` | partial |
| `aiap/case_io.py` | service/utility | file-I/O | `../scripts/metrics_eval.py` | role-match |
| `aiap/obligations.py` | model/utility | transform | `cov_test.py` | exact |
| `aiap/baseline.py` | service | graph traversal | `cov_test.py` | exact |
| `aiap/replay.py` | service | streaming transform | `cov_test.py`; `../regex_dfa_guide/src/diverse_guide_dfa.rs` | role-match |
| `aiap/reporting.py` | service | file-I/O/streaming | `../scripts/metrics_eval.py` | partial |
| `aiap/cli.py` | controller | request-response/file-I/O | `../scripts/metrics_eval.py` | role-match |
| `cov_test.py` | compatibility runner | request-response | `../scripts/metrics_eval.py` | partial |
| `tests/conftest.py` | test fixture | transform | `../tests/conftest.py` | exact |
| `tests/test_replay.py` | test | transform | `../tests/test_dfa.py`; `../tests/test_metrics.py` | role-match |
| `tests/test_baseline.py` | test | graph traversal | `../tests/conftest.py`; `cov_test.py` | role-match |
| `tests/test_case_parity.py` | integration test | file-I/O/transform | `../tests/test_metrics.py`; `../tests/test_metrics_eval_output.py` | role-match |

## Pattern Assignments

### `aiap/obligations.py` and `aiap/baseline.py` (model/utility and graph-traversal service)

**Primary analog:** `cov_test.py`

**Imports and typed-domain pattern** ([`cov_test.py:1-14`](/home/lxk/projects/diverse-structured-generation/aiap/cov_test.py:1)):

```python
from collections import Counter, deque
from dataclasses import dataclass
from typing import Literal

from regex_dfa_guide import DiverseGuideDFA
```

**Obligation representation and matching rule** ([`cov_test.py:22-67`](/home/lxk/projects/diverse-structured-generation/aiap/cov_test.py:22)):

```python
@dataclass
class CoverageObligation:
    obligation_type: Literal["state", "transition", "path"]
    obligation: int | tuple[int, int] | tuple[int, int, int]
    ...

    def is_met_by_transition(self, transition: tuple[int, int, int]) -> bool:
        prev_state, _, next_state = transition
        if self.is_transition_obligation():
            return self.transition == transition
        elif self.is_path_obligation():
            return self.path == (prev_state, next_state)
        elif self.is_state_obligation():
            return self.state == next_state
```

Keep this semantic matching but give catalog items stable IDs (`state:<n>`, `transition:<from>:<byte>:<to>`, `path:<from>:<to>`) and JSON-safe raw payloads. Build the universe from `dfa.get_states()` and `dfa.get_transitions()` as the existing implementation does at [`cov_test.py:70-98`](/home/lxk/projects/diverse-structured-generation/aiap/cov_test.py:70).

**Token-level BFS with byte-level obligation detection** ([`cov_test.py:119-157`](/home/lxk/projects/diverse-structured-generation/aiap/cov_test.py:119)):

```python
queue = deque([(initial_state, [])])
visited_states = {initial_state}
...
for token_id in dfa.get_allowed_token_ids(state):
    try:
        transitions = dfa.get_byte_transition_sequence_from_state(state, token_id)
    except ValueError:
        next_state = dfa.get_next_token_state(state, token_id)
        transitions = []
    else:
        next_state = transitions[-1][1]

    previous_state = state
    for byte, byte_next_state in transitions[1:]:
        if obligation.is_met_by_transition((previous_state, byte, byte_next_state)):
            ...
        previous_state = byte_next_state
```

Preserve `transitions[1:]`: the first item is a sentinel, not coverage. Preserve `None` for a reachable but non-completable obligation, as in the suffix BFS at [`cov_test.py:175-192`](/home/lxk/projects/diverse-structured-generation/aiap/cov_test.py:175).

### `aiap/replay.py` (canonical replay service, streaming transform)

**Primary analog:** `cov_test.py` byte inspection, backed by `DiverseGuideDFA` byte-sequence contract.

**DFA byte sequence contract** ([`../regex_dfa_guide/src/diverse_guide_dfa.rs:348-383`](/home/lxk/projects/diverse-structured-generation/regex_dfa_guide/src/diverse_guide_dfa.rs:348)):

```rust
let mut seq: Vec<(Byte, StateId)> = vec![(0, state)];
...
if token_id == self.eos_token_id {
    self.get_next_token_state(state, token_id)?;
    return Ok(vec![(0, state)]);
}
```

For every proposed token, first record `from_state`, resolve `to_state` with `get_next_token_state`, then expand `get_byte_transition_sequence_from_state(from_state, token_id)`. Convert only entries after the sentinel into `ByteStep(byte, from_state, to_state)`. An EOS action must have `byte_steps=[]` and must add no state/transition/path event, while the final DFA state remains the token-level destination.

**Legacy counting semantics to duplicate independently** ([`../src/diverse_guide/evaluation/metrics.py:63-77`](/home/lxk/projects/diverse-structured-generation/src/diverse_guide/evaluation/metrics.py:63), [`:94-137`](/home/lxk/projects/diverse-structured-generation/src/diverse_guide/evaluation/metrics.py:94)):

```python
for state in dfa.get_state_sequence(s):
    state_count[state] += 1

_, prev_state = trans_seq[0]
for char, state in trans_seq[1:]:
    transition_count[(prev_state, char, state)] += 1
    path_count[(prev_state, state)] += 1
    prev_state = state
```

Initialize each accepted certificate with exactly one initial state event. Aggregate ordered replay events with multiplicity, not merely its deduplicated covered ID list. Invalid, rejected, or trace-mismatched certificates must set `aggregate_eligible=False`, contain no credited coverage IDs, and retain a categorized diagnostic.

### `aiap/case_io.py` and `aiap/reporting.py` (file-I/O services)

**Primary analog:** `../scripts/metrics_eval.py`

**JSON input and structured artifact pattern** ([`../scripts/metrics_eval.py:247-265`](/home/lxk/projects/diverse-structured-generation/scripts/metrics_eval.py:247), [`:298-300`](/home/lxk/projects/diverse-structured-generation/scripts/metrics_eval.py:298)):

```python
with open(json_path) as f:
    gen_data = json.load(f)
...
if args.output:
    write_metrics_result(args.output, result)
    print(f"Structured results saved to: {args.output}")
```

Use `Path` and explicit paths (no demo-relative fallback). Validate the case's required `regex`, `samples`, and tokenizer/model provenance before constructing the DFA. Reporting should use stdlib `json` and write one serialized certificate per JSONL line; write the single summary JSON only when its explicit path was requested. Parent artifact tests verify by reopening the file and parsing JSON ([`../tests/test_metrics_eval_output.py:157-172`](/home/lxk/projects/diverse-structured-generation/tests/test_metrics_eval_output.py:157)).

### `aiap/cli.py`, `aiap/__main__.py`, and compatibility `cov_test.py` (controller/entry points)

**Primary analog:** `../scripts/metrics_eval.py`

**Argparse and thin orchestration pattern** ([`../scripts/metrics_eval.py:198-244`](/home/lxk/projects/diverse-structured-generation/scripts/metrics_eval.py:198)):

```python
parser = argparse.ArgumentParser(
    description="Evaluate diversity metrics of generated samples."
)
parser.add_argument("grammar", type=str, help="Grammar name")
parser.add_argument("--output", type=str, default=None, help="JSON output path.")
args = parser.parse_args()
```

**Human-readable summary pattern** ([`../scripts/metrics_eval.py:178-191`](/home/lxk/projects/diverse-structured-generation/scripts/metrics_eval.py:178)):

```python
print(f"- Number of samples: {result['sample_count']}")
print(f"- Number of states: {result['dfa']['state_count']}")
print(f"- State Coverage: {100 * metrics['state_coverage']:.2f}%")
```

`cli.main(argv: list[str] | None = None) -> int` should own parser construction and call pure/package modules. `__main__.py` should contain only `from .cli import main` and `raise SystemExit(main())`. The old `cov_test.py` is a compatibility runner only: import and delegate to the package instead of retaining BFS, input, or reporting logic.

### `tests/conftest.py`, `tests/test_replay.py`, and `tests/test_baseline.py` (fixture and unit tests)

**Primary analog:** `../tests/conftest.py`

**Minimal tokenizer/DFA fixture pattern** ([`../tests/conftest.py:6-53`](/home/lxk/projects/diverse-structured-generation/tests/conftest.py:6)):

```python
class MockTokenizer:
    eos_token_id = 99
    ...
    def get_vocab(self):
        return {"a": 10, "b": 20, "<eos>": 99}

@pytest.fixture
def vocab_ab():
    return {10: "a", 20: "b"}

@pytest.fixture
def dfa_single(vocab_ab):
    return DiverseGuideDFA("(?:[ab])$", 99, vocab_ab)
```

Extend this local fixture family with a multi-byte token (for example one token mapping to `"ab"`) and a DFA that accepts empty input. Keep tests deterministic and network-free. Assert explicit replay record fields for: initial-state credit, accepted empty input, an interior byte hit, EOS zero-byte action, rejected/illegal token failure, and an obligation that BFS reaches but cannot complete.

**Assertion style for DFA semantics** ([`../tests/test_dfa.py:96-125`](/home/lxk/projects/diverse-structured-generation/tests/test_dfa.py:96)):

```python
seq = dfa_multi.get_state_sequence("a")
assert seq[0] == initial
assert dfa_multi.is_final_state(seq[-1])

trans = dfa_multi.get_transition_sequence("a")
_, last_state = trans[-1]
assert dfa_multi.is_final_state(last_state)
```

### `tests/test_case_parity.py` (real-case integration test)

**Primary analog:** `../tests/test_metrics.py`

**Coverage-ratio assertion pattern** ([`../tests/test_metrics.py:50-59`](/home/lxk/projects/diverse-structured-generation/tests/test_metrics.py:50), [`:87-121`](/home/lxk/projects/diverse-structured-generation/tests/test_metrics.py:87)):

```python
cov = state_coverage(dfa_single, ["a"])
assert cov == 1.0

cov = transition_coverage(dfa_single, ["a"])
assert cov == pytest.approx(0.5)

cov = path_coverage(dfa_single, ["a"])
assert cov == pytest.approx(1.0)
```

For CSS color and email case inputs, parameterize per item (sample and BFS witness) and compare: acceptance, ordered state events, byte-transition events, state-pair events, count maps, and each of the three legacy ratios. Each comparison must expose the defined parity category (`acceptance`, `state_events`, `transition_events`, `path_events`, `ratio`, `none`) so a mismatch is diagnostic. Mark/cache-gate tests that require unavailable Hugging Face artifacts rather than weakening deterministic fixture tests.

### `pyproject.toml`, `aiap/__init__.py` (new isolated package boundary)

**Primary analog:** parent packaging configuration at [`../pyproject.toml:1-16`](/home/lxk/projects/diverse-structured-generation/pyproject.toml:1) and package-boundary tests at [`../tests/test_packaging_imports.py:1-33`](/home/lxk/projects/diverse-structured-generation/tests/test_packaging_imports.py:1).

Copy the standard PEP 621/setuptools shape but keep it local to `aiap/` and list no duplicate parent dependencies. Package discovery must find nested `aiap/`; do not change the parent root's `package-dir = "src"` configuration. Add a small import/entry-point test using `importlib.util.find_spec("aiap")`, matching the parent packaging test's namespace assertion strategy.

## Shared Patterns

### Imports, naming, and typed boundaries

**Source:** [`cov_test.py:1-14`](/home/lxk/projects/diverse-structured-generation/aiap/cov_test.py:1)

Use standard-library imports, then third-party (`transformers`), then parent/local package imports separated by blank lines. Use `snake_case` module/function names, `PascalCase` dataclasses, explicit return annotations, `Literal` for fixed categories, and modern `X | None` unions.

### DFA authority and token vocabulary

**Source:** [`../src/diverse_guide/guide_rust.py:25-42`](/home/lxk/projects/diverse-structured-generation/src/diverse_guide/guide_rust.py:25)

```python
token_id_to_token = build_token_id_map(tokenizer)
self.dfa = DiverseGuideDFA(
    "(?:" + regex_string + ")$", tokenizer.eos_token_id, token_id_to_token
)
```

Construct the production DFA through `DiverseRegexGuide`; do not recreate tokenizer vocabulary decoding or DFA semantics. Token IDs, rather than display text, are replay truth.

### Error representation

**Source:** [`cov_test.py:135-142`](/home/lxk/projects/diverse-structured-generation/aiap/cov_test.py:135) and [`../regex_dfa_guide/src/python_bindings/mod.rs:93-115`](/home/lxk/projects/diverse-structured-generation/regex_dfa_guide/src/python_bindings/mod.rs:93)

Parent DFA calls can raise `ValueError` for invalid state/token/byte transitions. Replay should catch and turn those into structured failure categories; baseline search may return `None` for no constructible accepted witness. Do not treat EOS's valid zero-byte action as an error, and do not silently suppress any other mismatch.

## No Analog Found

| File/Concern | Role | Data Flow | Reason |
|---|---|---|---|
| `aiap/replay.py` certificate records | service/model | streaming transform | No existing component persists token-level DFA byte traces or enforces aggregate eligibility. Use the Phase 1 research contract plus parent DFA APIs. |
| `aiap/reporting.py` JSONL stream | service | file-I/O | Parent code writes one JSON result, not a candidate-per-line evidence stream. |
| `aiap/__main__.py` | entry point | request-response | The parent uses script guards rather than a `python -m` package entry point. |

## Metadata

**Analog search scope:** `aiap/`, `../src/diverse_guide/`, `../scripts/`, `../tests/`, and `../regex_dfa_guide/`  
**Files scanned:** 12  
**Pattern extraction date:** 2026-07-23
