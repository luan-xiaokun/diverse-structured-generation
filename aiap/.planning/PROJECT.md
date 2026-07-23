# AIAP Multi-Obligation Witness Generation

## What This Is

一个面向 AISoLA 2026 AI Assisted Programming（AIAP）track 的研究型原型项目，扩展已发表于 ICFEM 2025 的 diverse structured generation 工作。项目以直接遍历 DFA 为基础，生成能够覆盖既有样本未覆盖 DFA states、byte-level transitions 与 state-pair paths 的有效 witness；进一步将其发展为可在一次生成中覆盖多项 obligation、并由语言模型提升自然度的混合方法。

该项目也将把新 witness 与 ICFEM 2025 方法产生的样本结合，通过近似/启发式选择较小而高覆盖的互补测试套件，并在 case studies 上评估覆盖、文本质量与测试效用。当前 `aiap/cov_test.py` 是单-obligation BFS baseline 的探索性 demo。

## Core Value

在保持 DFA 约束正确性的前提下，以少量、自然且互补的测试样本显著提高结构化生成输出的 DFA 覆盖。

## Requirements

### Validated

- ✓ 单一 state、transition 或 path obligation 可由 token-level DFA BFS 构造一个接受的 witness — 现有 `aiap/cov_test.py` demo
- ✓ 现有样本可计算 state、transition 与 path 覆盖及其未满足 obligations — 现有 `aiap/cov_test.py` demo

### Active

- [ ] 设计多-obligation witness 生成算法，包括从大量未满足 obligations 中构造可联合覆盖的目标组。
- [ ] 设计图遍历与语言模型结合的候选生成/搜索方法，在 DFA 有效性不被破坏的前提下提升 witness 自然度。
- [ ] 设计新 witness 与 ICFEM 2025 diverse structured generation 样本的互补测试套件选择算法，权衡覆盖水平与集合大小。
- [ ] 在可复现 case studies 上实现基线、消融与评估，量化覆盖、效率、自然度及测试质量。
- [ ] 形成支撑 AIAP 投稿的研究叙事、实验数据与可复现实验工件。

### Out of Scope

- 严格求解全局最优最小测试套件 — 组合优化难度高；v1 采用可解释的启发式或带界近似策略。
- 从头训练或微调大语言模型 — 研究重点是 DFA-约束搜索和推理时 LLM 引导，而非模型训练。
- 将原型产品化为通用在线服务或 IDE 插件 — 当前目标是研究投稿与可复现实验。
- 更改 ICFEM 2025 主库的既有行为 — AIAP 原型默认隔离在 `aiap/`，仅复用其 DFA 与数据资产。

## Context

- 父仓库是 ICFEM 2025 diverse structured generation 工作的实现，提供 `regex_dfa_guide` / `diverse_guide`、case-study JSON 数据与覆盖度量。
- `aiap/cov_test.py` 当前将每个 missing state/transition/path 表示为 `CoverageObligation`，并用 BFS 构造一个最短 token 路径的 accepted witness。
- 最短 DFA witness 的局限是每次只针对一项 obligation，且其 token 最短性不能保证文本自然度。
- AIAP track 明确关注 LLM 与 formal specification、verification 和 testing 的结合；本工作应突出符号 DFA 保证与 LLM 引导之间的互补关系。
- 项目根目录固定为 `aiap/`；它处于父仓库 Git worktree 内，父项目已有未提交的 Rust/Python 修改必须保持不变。

## Constraints

- **Correctness**: 所有交付的 witness 必须被目标 DFA 接受，且其实际 transition trace 可验证地覆盖所声称的 obligations — 形式约束是方法可信度的基础。
- **Research scope**: 算法必须同时处理 multi-obligation grouping、自然度引导与互补套件选择 — 三者共同构成相对于 ICFEM 2025 baseline 的贡献。
- **Reproducibility**: case studies、tokenizer/model 标识、随机种子、解码/beam 参数与评价脚本须记录 — 论文结论必须可复现。
- **Isolation**: 新实现和工件默认置于 `aiap/` — 避免与 ICFEM 2025 主实现的在途修改交叉污染。
- **Optimization**: 允许启发式或次优选择，不以严格全局最优为 v1 目标 — 将研究资源投入可测量的覆盖/质量改进。

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| 将 `aiap/` 作为项目根目录 | 保持 AIAP 投稿原型与 ICFEM 2025 主实现隔离，同时可复用 DFA 资产 | — Pending |
| 以 DFA traversal 为正确性骨架 | 每个候选都能由接受性与 trace 覆盖进行机械验证 | — Pending |
| 将 LLM 用于搜索/排序的自然度提升，而非取代 DFA 约束 | 保留形式保证，同时探索自然语言质量的设计空间 | — Pending |
| 采用多-obligation 与互补套件选择的分层研究问题 | 既提高单样本覆盖收益，也控制最终测试集规模 | — Pending |
| 将严格最优集合覆盖留在范围外 | 该问题很难；可解释的启发式足以支持本投稿的实证目标 | — Pending |

## Evolution

This document evolves at phase transitions and milestone boundaries.

**After each phase transition** (via `$gsd-transition`):
1. Requirements invalidated? → Move to Out of Scope with reason
2. Requirements validated? → Move to Validated with phase reference
3. New requirements emerged? → Add to Active
4. Decisions to log? → Add to Key Decisions
5. "What This Is" still accurate? → Update if drifted

**After each milestone** (via `$gsd-complete-milestone`):
1. Full review of all sections
2. Core Value check — still the right priority?
3. Audit Out of Scope — reasons still valid?
4. Update Context with current state

---
*Last updated: 2026-07-23 after initialization*
