<!-- GSD:project-start source:PROJECT.md -->

## Project

**AIAP Multi-Obligation Witness Generation**

一个面向 AISoLA 2026 AI Assisted Programming（AIAP）track 的研究型原型项目，扩展已发表于 ICFEM 2025 的 diverse structured generation 工作。项目以直接遍历 DFA 为基础，生成能够覆盖既有样本未覆盖 DFA states、byte-level transitions 与 state-pair paths 的有效 witness；进一步将其发展为可在一次生成中覆盖多项 obligation、并由语言模型提升自然度的混合方法。

该项目也将把新 witness 与 ICFEM 2025 方法产生的样本结合，通过近似/启发式选择较小而高覆盖的互补测试套件，并在 case studies 上评估覆盖、文本质量与测试效用。当前 `aiap/cov_test.py` 是单-obligation BFS baseline 的探索性 demo。

**Core Value:** 在保持 DFA 约束正确性的前提下，以少量、自然且互补的测试样本显著提高结构化生成输出的 DFA 覆盖。

### Constraints

- **Correctness**: 所有交付的 witness 必须被目标 DFA 接受，且其实际 transition trace 可验证地覆盖所声称的 obligations — 形式约束是方法可信度的基础。
- **Research scope**: 算法必须同时处理 multi-obligation grouping、自然度引导与互补套件选择 — 三者共同构成相对于 ICFEM 2025 baseline 的贡献。
- **Reproducibility**: case studies、tokenizer/model 标识、随机种子、解码/beam 参数与评价脚本须记录 — 论文结论必须可复现。
- **Isolation**: 新实现和工件默认置于 `aiap/` — 避免与 ICFEM 2025 主实现的在途修改交叉污染。
- **Optimization**: 允许启发式或次优选择，不以严格全局最优为 v1 目标 — 将研究资源投入可测量的覆盖/质量改进。

<!-- GSD:project-end -->

<!-- GSD:stack-start source:codebase/STACK.md -->

## Technology Stack

## Languages

- Python 3 — the exploratory witness-construction demo in `aiap/cov_test.py`.
- Rust — used indirectly by the parent repository's `regex_dfa_guide` extension that exposes DFA operations to Python.

## Runtime

- Python virtual environment at the repository root (`.venv/`) is present; this demo does not declare a separate environment.
- The demo is executed as a script with Python, from a working directory where `experiments/` is resolvable.
- Parent project uses `pyproject.toml` and `uv.lock`; `aiap/` has no independent package manifest or lockfile.

## Frameworks

- `regex_dfa_guide` / `diverse_guide` — parent-project packages that provide DFA traversal, token constraints, and coverage metrics.
- Hugging Face `transformers` — `AutoTokenizer` loads the tokenizer used to convert token IDs into witness text.
- No `aiap/` tests or test runner configuration yet.

## Key Dependencies

- `DiverseGuideDFA` — DFA state, transition, accepted-token, and transition-sequence APIs.
- `DiverseRegexGuide` — builds the DFA from a regular expression and tokenizer.
- `transformers.AutoTokenizer` — loads `Qwen/Qwen2.5-0.5B` in the current demo.

## Configuration

- Input datasets are hard-coded relative paths under `experiments/case_study/data/` in `aiap/cov_test.py`.
- Coverage threshold defaults to one visit in `compute_coverage_metrics` and `get_coverage_obligations`.
- No environment variables, deployment configuration, or user-facing CLI configuration are defined.

## Platform Requirements

- Python environment compatible with the parent repository and its Rust/Python extension.
- Network or cached Hugging Face model artifacts are needed on first tokenizer load.
- This is a research demo, not a deployed service.

<!-- GSD:stack-end -->

<!-- GSD:conventions-start source:CONVENTIONS.md -->

## Conventions

## Naming Patterns

- Python modules use `snake_case`; the current experiment is `aiap/cov_test.py`.
- No package/module export convention exists yet because the demo is a single script.
- `snake_case` descriptive names, for example `get_coverage_obligations` and `construct_one_witness`.
- Predicate helpers begin with `is_`, for example `is_path_obligation` and `is_met_by_transition`.
- `snake_case` for local values and collection names.
- Type-specific names such as `state_count`, `transition_count`, and `prefix_token_ids` make traversal data explicit.
- `PascalCase` for data classes (`CoverageObligation`).
- `typing.Literal` specifies the finite obligation kinds.

## Code Style

- Four-space indentation and double-quoted literals are used in `aiap/cov_test.py`.
- Imports are grouped as standard library, third-party packages, then parent-project imports, with blank lines between groups.
- No formatter or linter configuration is local to `aiap/`.
- Function parameters and result types are annotated throughout the demo.
- Modern union syntax (`list[int] | None`) is used.
- Tuple shapes are encoded directly in type annotations.

## Error Handling

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

<!-- GSD:conventions-end -->

<!-- GSD:architecture-start source:ARCHITECTURE.md -->

## Architecture

## Pattern Overview

- Defines coverage-obligation data and derives uncovered state, transition, and path obligations.
- Uses breadth-first traversal over token-level DFA transitions to build one accepted witness for one obligation.
- Runs fixed case studies and prints results; there is no reusable public API, experiment orchestrator, or persistence layer yet.

## Layers

- Location: `aiap/cov_test.py` (`compute_coverage_metrics`, `get_coverage_obligations`).
- Purpose: derive uncovered state, byte-transition, and state-pair path obligations from current samples.
- Depends on: DFA inspection and metrics from the parent packages.
- Location: `aiap/cov_test.py` (`construct_one_witness`).
- Purpose: find a token prefix that traverses one obligation, then a suffix to an accepting DFA state.
- Depends on: DFA allowed-token and state-transition APIs.
- Location: `aiap/cov_test.py` under `if __name__ == "__main__":`.
- Purpose: load two datasets and a tokenizer, then demonstrate one witness per case study.

## Data Flow

- In-memory `Counter`, `set`, queue, and token-ID lists only.
- Dataset and model cache are external; no output artifact is written.

## Key Abstractions

- Represents exactly one state, byte transition, or state-pair path obligation.
- Uses `is_met_by_transition` to match a byte-level DFA step.
- Current design intentionally models a single obligation, which is the principal limitation for the planned multi-obligation method.
- Accessed through `DiverseGuideDFA` methods such as `get_allowed_token_ids`, `get_byte_transition_sequence_from_state`, and `is_final_state`.
- The witness code treats tokens as paths of one or more byte transitions.

## Entry Points

- Location: `aiap/cov_test.py`.
- Trigger: `python aiap/cov_test.py` from the parent repository root.
- Responsibilities: load tokenizer/data, construct guide, identify obligations, and print one witness.

## Error Handling

- `ValueError` from byte transition lookup is treated as an EOS-related token-level transition and skipped for byte-obligation matching.
- A missing witness is represented by `None`.
- File, tokenizer, and schema errors otherwise propagate without contextual handling.

<!-- GSD:architecture-end -->

<!-- GSD:skills-start source:skills/ -->

## Project Skills

No project skills found. Add skills to any of: `.claude/skills/`, `.agents/skills/`, `.cursor/skills/`, `.github/skills/`, or `.codex/skills/` with a `SKILL.md` index file.
<!-- GSD:skills-end -->

<!-- GSD:workflow-start source:GSD defaults -->

## GSD Workflow Enforcement

Before using Edit, Write, or other file-changing tools, start work through a GSD command so planning artifacts and execution context stay in sync.

Use these entry points:

- `/gsd-quick` for small fixes, doc updates, and ad-hoc tasks
- `/gsd-debug` for investigation and bug fixing
- `/gsd-execute-phase` for planned phase work

Do not make direct repo edits outside a GSD workflow unless the user explicitly asks to bypass it.
<!-- GSD:workflow-end -->

<!-- GSD:profile-start -->

## Developer Profile

> Profile not yet configured. Run `/gsd-profile-user` to generate your developer profile.
> This section is managed by `generate-claude-profile` -- do not edit manually.
<!-- GSD:profile-end -->
