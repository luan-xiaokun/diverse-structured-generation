# Coding Conventions

**Analysis Date:** 2026-07-23

## Naming Patterns

**Files:**
- Python modules use `snake_case`; the current experiment is `aiap/cov_test.py`.
- No package/module export convention exists yet because the demo is a single script.

**Functions:**
- `snake_case` descriptive names, for example `get_coverage_obligations` and `construct_one_witness`.
- Predicate helpers begin with `is_`, for example `is_path_obligation` and `is_met_by_transition`.

**Variables:**
- `snake_case` for local values and collection names.
- Type-specific names such as `state_count`, `transition_count`, and `prefix_token_ids` make traversal data explicit.

**Types:**
- `PascalCase` for data classes (`CoverageObligation`).
- `typing.Literal` specifies the finite obligation kinds.

## Code Style

**Formatting:**
- Four-space indentation and double-quoted literals are used in `aiap/cov_test.py`.
- Imports are grouped as standard library, third-party packages, then parent-project imports, with blank lines between groups.
- No formatter or linter configuration is local to `aiap/`.

**Typing:**
- Function parameters and result types are annotated throughout the demo.
- Modern union syntax (`list[int] | None`) is used.
- Tuple shapes are encoded directly in type annotations.

## Error Handling

**Patterns:**
- Invalid obligation kinds raise `ValueError` in `CoverageObligation` methods.
- One known traversal exception is handled locally: unavailable byte transitions are treated as EOS-like token behavior.
- Search failure returns `None` instead of raising, allowing the caller to distinguish impossible/uncompletable obligations.

## Logging and Comments

- The demo uses `print` for progress and outputs.
- The detailed docstring in `construct_one_witness` explains the prefix/obligation/suffix idea.
- Inline comments explain DFA/token edge cases rather than restating obvious code.

## Function and Module Design

- Helpers are currently pure or nearly pure: inputs are DFA/sample collections and outputs are values, not files.
- BFS queues carry `(state, token_ids)` pairs and `visited_states` intentionally avoid repeated expansion.
- As the prototype grows, separate obligation selection, graph search, LLM scoring, and suite selection into modules; preserve the existing typed data model at boundaries.

---
*Convention analysis: 2026-07-23*
