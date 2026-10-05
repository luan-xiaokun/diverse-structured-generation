# AIAP 项目交接

更新：2026-10-05。项目根目录：`aiap/`；Git 仓库根目录是其父目录。本交接依据规划文档、代码和提交记录整理；本次没有重新运行产品测试。最近产品代码提交为 `d5618da`。

## 研究目标

基于已发表于 ICFEM 2025 的 diverse structured generation，建立探索性研究 demo，原始投稿目标为 AISoLA 2026 AIAP track。会议当前安排及投稿可行性本次未核验。

既有样本没有覆盖全部 DFA 结构，项目用图搜索补充有效 witness，并研究：

1. 如何选择、组合多个缺失目标，让一次生成覆盖多项 obligations。
2. 如何结合 DFA traversal 与 LLM 搜索/排序，提升 witness 自然度。
3. 如何从原始 samples 与新 witnesses 中选出小而高覆盖的互补测试套件。
4. 如何在 case studies 上评估覆盖、效率、自然度与测试效用。

允许启发式和次优解；不以严格全局最优、模型训练或产品化为目标。任何有效性与覆盖声明都必须以实际 DFA replay 为依据。

## 当前进展

| 阶段 | 状态 | 交付/剩余工作 |
|---|---|---|
| 1：Baseline Replay Foundation | 完成，3/3 plans | 模块化单-obligation BFS、canonical replay、coverage certificates、CLI 与报告 |
| 01.1：Vocab/byte DFA contract | 完成，7/7 plans | bytes-only API、0.3.0、严格 self-contained artifact schema、reachability/outcomes、离线 parity 门控 |
| 2：Feasible Multi-Obligation Search | 待讨论与规划 | 联合目标分组、coverage-aware search、dominance/pruning、搜索预算、小规模精确 oracle |
| 3：Complementary Suite Selection | 未开始 | 共享候选池、greedy marginal-gain、固定候选池对比 |
| 4：DFA-Masked LLM Quality Guidance | 未开始 | DFA 合法动作上的 LLM beam/search/reranking、自然度对比 |
| 5：Locked Evaluation Evidence | 未开始 | 固定数据划分、模型/预算/种子、消融实验与论文证据 |

两个已完成阶段、十个已完成 phase plans。Phase 2–5 没有计划文件。研究核心的新算法尚未实现。

## 必须保持的实验语义

- **Coverage obligation 仅指选定 baseline samples 尚未 visited 的 DFA 结构。** 已访问结构不叫 obligation。内部结构 catalog 用于稳定 ID 和 replay，不等于缺失目标集合。
- `--sample-num N` 使用输入数据前 N 个原始 samples 进行 replay，计算 missing obligations。
- 当前覆盖类型为 state、byte-level transition `(source, byte, target)`、相邻有序 state-pair path `(source, target)`。这里的 path 不是任意长路径。
- 顺序固定为 baseline replay → missing obligations → reachability 分类 → 选择 reachable targets → witness 尝试 → canonical replay 验证。
- `--obligation-num N`（别名 `--ob-num`）至多选择 N 个 reachable missing targets。目前每次搜索针对一个目标；实际 witness 可以顺带覆盖更多结构。
- `byte_level_unreachable` 仍然是缺失 obligation，留在汇总 JSON 的诊断中，不占生成名额。
- JSONL 仅写实际 witness 尝试的 certificates（包括失败尝试），不包含原始 samples 或未尝试的 unreachable 条目。
- Outcomes：`success`、`byte_level_unreachable`、`search_limit`、`invalid_witness`、`internal_error`。成功须同时满足 replay 接受性和目标 ID 的真实覆盖。
- `--max-expansions N` 是每个目标的 BFS expansion 上限；默认不设上限。资源上限耗尽不能等同于结构不可达。

## 已解决的问题与历史验证

原先 tokenizer vocabulary 的文本转换破坏了 token 的真实 byte 值，引起找不到接受后缀等问题。Phase 01.1 将 Python/Rust 契约统一为 `dict[int, bytes]`，拒绝字符串及旧 artifact；EOS 保留为零 byte 动作。artifact 使用显式 magic/schema，self-contained restore 不依赖 tokenizer 或网络。错误区分 `legacy_artifact`、`unsupported_schema` 和 `corrupt_artifact`。

2026-07-24 独立阶段验证记录：12/12 must-haves；Rust 15/15；hermetic pytest 89 passed、3 deselected；锁定 Qwen 的 CSS/email/U+1680 完整 byte/trace parity 通过。非阻塞问题是 `real_case_parity` pytest marker 未注册。

阶段完成后真实 CLI 曾长时间卡在 reachability：每个 missing obligation 重复扫描所有 accepting token actions。`d5618da` 改为对 supplied missing meanings 建索引、一次扫描 action 计数，保持分类及每个 action 至多贡献一次的语义。修复后下面的 CSS 命令历史耗时约 18.7 秒，1000 baseline + 10 witnesses 全部接受；合并覆盖 state=0.633、transition=0.252、path=0.426，success=10。该结果是 smoke run，不能作为正式研究结论或当前机器的性能保证。

## 可执行入口

从父仓库根目录进入 `aiap/`，使用父仓库 `.venv`；需要安装当前 native 0.3.0 扩展和本地锁定 tokenizer 快照。

```bash
cd aiap
HF_HUB_OFFLINE=1 ../.venv/bin/python -m aiap \
  ../experiments/case_study/data/diverse-css-color.json \
  --sample-num 1000 \
  --obligation-num 10 \
  --json /tmp/css-small.json \
  --jsonl /tmp/css-witnesses.jsonl
```

Tokenizer 锁定为 `Qwen/Qwen2.5-1.5B-Instruct@989aa7980e4cf806f80c7fef2b1adb7bc71aa306`。缺失快照应严格失败，不改 revision、跳过测试或回退在线加载。

从 `aiap/` 执行普通测试和真实离线 parity：

```bash
../.venv/bin/python -m pytest -m 'not real_case_parity'
HF_HUB_OFFLINE=1 ../.venv/bin/python -m pytest \
  tests/test_case_parity.py -m real_case_parity
```

## 下一步与未决问题

推荐从 `aiap/` 执行 `$gsd-discuss-phase 2`，明确分组算法、联合可行性判断、搜索 label 的 coverage 信息、dominance 关系、预算和失败分类，然后 `$gsd-plan-phase 2 research first`。Phase 2 当前未规划，不应直接把它当作待执行阶段。

后续需要确定 LLM/model lock、长度匹配的自然度评价、development/held-out case 划分，以及等预算 baseline/消融协议。当前没有正式测试质量结论。

`PROJECT.md` 和 `AGENTS.md` 中自动生成的 codebase 描述仍有初始化时期内容，如“没有 package/CLI/tests”。恢复工作时以实际代码和完成的阶段证据为准；遵守 AGENTS.md 的 workflow 要求。

本次开始时工作区有未跟踪的 `regex_dfa_guide/uv.lock`；它不属于本交接任务，已保留。

## 恢复阅读与代码导航

- [STATE.md](.planning/STATE.md)：当前阶段和会话恢复点。
- [ROADMAP.md](.planning/ROADMAP.md)：各阶段目标与 success criteria。
- [PROJECT.md](.planning/PROJECT.md)：最初研究范围。
- [01.1-CONTEXT.md](.planning/phases/01.1-repair-tokenizer-vocabulary-and-byte-dfa-contract/01.1-CONTEXT.md)：已锁定的详细契约。
- [01.1-VERIFICATION.md](.planning/phases/01.1-repair-tokenizer-vocabulary-and-byte-dfa-contract/01.1-VERIFICATION.md)：历史独立验证证据。
- [cli.py](aiap/cli.py)：实验 pipeline 与输入输出人口。
- [baseline.py](aiap/baseline.py)：单-target BFS。
- [reachability.py](aiap/reachability.py)：accepting token-action graph 分类与索引计数。
- [replay.py](aiap/replay.py)：canonical certificates 与 outcomes。
- [obligations.py](aiap/obligations.py)：结构 ID 和覆盖含义。
