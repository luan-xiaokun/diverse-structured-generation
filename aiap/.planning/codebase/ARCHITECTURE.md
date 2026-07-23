# Architecture

**Analysis Date:** 2026-07-23

## Pattern Overview

**Overall:** Single-file exploratory research script layered over an existing DFA-guided generation library.

**Key Characteristics:**
- Defines coverage-obligation data and derives uncovered state, transition, and path obligations.
- Uses breadth-first traversal over token-level DFA transitions to build one accepted witness for one obligation.
- Runs fixed case studies and prints results; there is no reusable public API, experiment orchestrator, or persistence layer yet.

## Layers

**Coverage analysis:**
- Location: `aiap/cov_test.py` (`compute_coverage_metrics`, `get_coverage_obligations`).
- Purpose: derive uncovered state, byte-transition, and state-pair path obligations from current samples.
- Depends on: DFA inspection and metrics from the parent packages.

**Witness construction:**
- Location: `aiap/cov_test.py` (`construct_one_witness`).
- Purpose: find a token prefix that traverses one obligation, then a suffix to an accepting DFA state.
- Depends on: DFA allowed-token and state-transition APIs.

**Experiment runner:**
- Location: `aiap/cov_test.py` under `if __name__ == "__main__":`.
- Purpose: load two datasets and a tokenizer, then demonstrate one witness per case study.

## Data Flow

**Single-obligation witness generation:**

1. Load a regex and generated samples from a case-study JSON file.
2. Build `DiverseRegexGuide`, then obtain its DFA.
3. Count traversed states, byte transitions, and state-pair paths for the samples.
4. Materialize each under-covered item as a `CoverageObligation`.
5. BFS over allowed tokens until one token's byte sequence satisfies the selected obligation.
6. BFS from the matched DFA state to an accepting state.
7. Decode token IDs and print an accepted witness.

**State Management:**
- In-memory `Counter`, `set`, queue, and token-ID lists only.
- Dataset and model cache are external; no output artifact is written.

## Key Abstractions

**CoverageObligation:**
- Represents exactly one state, byte transition, or state-pair path obligation.
- Uses `is_met_by_transition` to match a byte-level DFA step.
- Current design intentionally models a single obligation, which is the principal limitation for the planned multi-obligation method.

**DFA traversal API:**
- Accessed through `DiverseGuideDFA` methods such as `get_allowed_token_ids`, `get_byte_transition_sequence_from_state`, and `is_final_state`.
- The witness code treats tokens as paths of one or more byte transitions.

## Entry Points

**Demo script:**
- Location: `aiap/cov_test.py`.
- Trigger: `python aiap/cov_test.py` from the parent repository root.
- Responsibilities: load tokenizer/data, construct guide, identify obligations, and print one witness.

## Error Handling

- `ValueError` from byte transition lookup is treated as an EOS-related token-level transition and skipped for byte-obligation matching.
- A missing witness is represented by `None`.
- File, tokenizer, and schema errors otherwise propagate without contextual handling.

---
*Architecture analysis: 2026-07-23*
