# Codebase Concerns

**Analysis Date:** 2026-07-23

## Tech Debt

**Single-file prototype:**
- Issue: coverage accounting, DFS/BFS-style search, data loading, and output formatting all live in `aiap/cov_test.py`.
- Why: it is explicitly an exploratory demo.
- Impact: multi-obligation search, LLM integration, and suite selection will be difficult to test or compare independently.
- Fix approach: extract stable domain objects and algorithm modules before adding the full experiment pipeline.

**Parent-project coupling:**
- Issue: the demo imports local packages and reads parent-relative case-study files without a manifest or installation boundary.
- Impact: running from the wrong directory or an environment without the compiled extension fails.
- Fix approach: document an explicit invocation command and later add a package/experiment configuration for AIAP.

## Known Limitations

**One obligation per witness:**
- Current behavior: `construct_one_witness` receives only one `CoverageObligation`.
- Consequence: it cannot optimize coverage gain across several missing states, transitions, or paths.
- Project relevance: this is the primary research gap that the AIAP work is intended to address.

**Shortest traversal is not natural language quality:**
- Current behavior: BFS minimizes token count under DFA constraints.
- Consequence: decoded text can be short and valid but unnatural.
- Project relevance: LLM-guided beam search/reranking needs a separate quality objective and a reproducible evaluation protocol.

## Fragile Areas

**Byte-token alignment:**
- Why fragile: coverage is defined over byte-level DFA transitions but search expands token-level actions.
- Common failures: missing an obligation inside a multi-byte token or incorrectly treating EOS as a byte transition.
- Current mitigation: the script iterates `transitions[1:]` and catches `ValueError` for the EOS case.
- Safe modification: preserve tests for in-token and EOS behavior before changing DFA API usage.

**Visited-state pruning:**
- Why fragile: `visited_states` keeps only one token path per DFA state.
- Consequence: correct for a shortest reachability witness but inadequate when search state must retain already-covered obligation sets, quality scores, or beam diversity.
- Safe modification: extend search state to include an obligation-coverage representation and objective values; do not reuse the current pruning rule unchanged.

## Performance Bottlenecks

**Token branching:**
- Problem: each BFS expansion iterates every allowed token and may inspect each token's byte transition sequence.
- Measurement: none recorded.
- Improvement path: precompute graph annotations, compact equivalent actions, limit/group obligation sets, and benchmark separately from LLM scoring.

## Test Coverage Gaps

- `aiap/cov_test.py` has no dedicated tests.
- No assertions verify that its decoded witnesses actually cover their intended obligations.
- No baseline experiment artifacts record coverage gain, naturalness, or selection-set size.

---
*Concerns audit: 2026-07-23*
