# Requirements: AIAP Multi-Obligation Witness Generation

**Defined:** 2026-07-23
**Core Value:** 在保持 DFA 约束正确性的前提下，以少量、自然且互补的测试样本显著提高结构化生成输出的 DFA 覆盖。

## v1 Requirements

### Baseline Demo

- [ ] **BASE-01**: 研究者可将现有单-obligation BFS demo 整理为可导入的基线模块，并在固定 case-study 输入上复现其 accepted witness 与覆盖结果。
- [ ] **BASE-02**: 研究者可用同一 DFA trace replay 明确验证每个基线或新 witness 的接受性及其实际覆盖的 state、byte-transition 与 ordered state-pair path obligations。

### Multi-Obligation Search

- [ ] **MULTI-01**: 研究者可从未覆盖且处于 initial-to-final accepting trace 上的 obligations 中建立具有稳定 ID 的候选目标集合，并记录不可达或不可联合满足的项。
- [ ] **MULTI-02**: 研究者可构造并验证可联合覆盖多个 obligations 的目标组，而非每次仅选择一个 obligation。
- [ ] **MULTI-03**: 研究者可运行 coverage-aware dominance search，在相同 DFA state 保留不同已覆盖 obligation sets，并输出每个 accepted witness 的 replayed coverage certificate。
- [ ] **MULTI-04**: 研究者可在小型 DFA/candidate pool 上将 multi-obligation search 与穷举或受限精确 oracle 对比，量化 cap/pruning 的效果而不宣称全局最优。

### Suite Selection

- [ ] **SUIT-01**: 研究者可将 ICFEM 2025 samples 与新 witness 统一 replay 为带来源、成本和 obligation-incidence 的候选池。
- [ ] **SUIT-02**: 研究者可用确定性的贪心边际增益算法从候选池选择小型互补测试套件，并记录 tie-break、边际收益和来源构成。
- [ ] **SUIT-03**: 研究者可在相同验证候选池上比较 ICFEM-only、AIAP-only、union、random 与贪心选择的 coverage@k / target-coverage suite size。

### LLM Quality Guidance

- [ ] **QUAL-01**: 研究者可在 DFA 合法 token actions 内运行 beam/search 或 reranking，使 LLM 不具有接受性或覆盖判定权。
- [ ] **QUAL-02**: 研究者可对每个 LLM-influenced witness 重新 replay，并报告 DFA validity、coverage retention、长度与生成成本。
- [ ] **QUAL-03**: 研究者可在 coverage 与长度匹配的有效 witnesses 上比较 quality-aware 与 quality-unaware 方法，报告固定自动自然度 proxy 以及盲测 naturalness pilot 的结果和协议。

### Evaluation and Evidence

- [ ] **EVAL-01**: 研究者可通过版本化 manifest 固定 case studies、初始样本、DFA/tokenizer/model revisions、seeds、budgets 和解码/搜索参数。
- [ ] **EVAL-02**: 研究者可在固定的 development/held-out case-study protocol 下，以相同资源预算运行 B0/B1、分组、选择与 LLM 消融。
- [ ] **EVAL-03**: 研究者可按 case、seed 和覆盖准则分别报告绝对/相对覆盖、suite size、效率、失败类别与不确定性，而非只给单一 aggregate 数字。
- [ ] **EVAL-04**: 研究者可从保存的原始 artifacts 复跑核心实验，并将论文结论限定为可测量的 DFA coverage、witness quality 与 suite complementarity。

## v2 Requirements

### Extended Research

- **EXT-01**: 研究者可为所有 case studies 提供带独立 executable oracle 的 fault/mutation-detection transfer 评估。
- **EXT-02**: 研究者可研究训练或微调针对特定形式语言的自然度模型。
- **EXT-03**: 研究者可提供严格全局最小测试套件的可扩展求解器。
- **EXT-04**: 研究者可将原型扩展为 IDE 插件、在线服务或交互式可视化产品。

## Out of Scope

| Feature | Reason |
|---------|--------|
| 严格全局最优最小套件的生产求解 | 组合优化代价高；v1 的确定性贪心与小规模 oracle 足以刻画实际权衡。 |
| LLM 训练或微调 | 增加数据/成本/归因变量，不服务于本项目 DFA-引导推理时研究问题。 |
| unconstrained generate-then-repair | 无法保持直接 DFA traversal 的形式保证，也会混淆有效性来源。 |
| 通用产品、IDE 或 Web UI | 探索性论文原型优先需要算法、对照和可复现实验。 |
| 仅用 parser success 或 LLM judge 认定正确/自然 | DFA replay 是唯一正确性 oracle；自然度需独立且披露的评价协议。 |
| 只报告单一 aggregate coverage | 会掩盖 state、transition 与 path 覆盖及分母不一致的问题。 |

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| BASE-01 | Phase 1 | Pending |
| BASE-02 | Phase 1 | Pending |
| MULTI-01 | Phase 2 | Pending |
| MULTI-02 | Phase 2 | Pending |
| MULTI-03 | Phase 2 | Pending |
| MULTI-04 | Phase 2 | Pending |
| SUIT-01 | Phase 3 | Pending |
| SUIT-02 | Phase 3 | Pending |
| SUIT-03 | Phase 3 | Pending |
| QUAL-01 | Phase 4 | Pending |
| QUAL-02 | Phase 4 | Pending |
| QUAL-03 | Phase 4 | Pending |
| EVAL-01 | Phase 5 | Pending |
| EVAL-02 | Phase 5 | Pending |
| EVAL-03 | Phase 5 | Pending |
| EVAL-04 | Phase 5 | Pending |

**Coverage:**
- v1 requirements: 16 total
- Mapped to phases: 16
- Unmapped: 0 ✓
- Duplicate phase mappings: 0 ✓

---
*Requirements defined: 2026-07-23*
*Last updated: 2026-07-23 after interactive v1 scoping*
