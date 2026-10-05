# AIAP 研究路线

更新：2026-10-05。此文件记录研究方向；实施通过直接讨论、代码修改与适当验证推进。

| 工作 | 状态 | 完成标准 |
|---|---|---|
| 单-obligation baseline 与 replay | 已完成（历史 Phase 1） | 单-target BFS 可运行；实际 replay 验证接受性与覆盖；输出 certificates |
| Vocab/byte DFA 契约 | 已完成（历史 Phase 01.1） | bytes-only 0.3.0；严格 self-contained artifacts；trace parity；missing/classify/select/attempt 语义 |
| Multi-obligation grouping/search | 下一项，待设计 | 联合可行的目标组；replay-certified joint witnesses；明确 dominance、pruning、caps；小规模 oracle 衡量搜索近似效果 |
| 互补测试套件选择 | 未开始 | 共享候选池；确定性 greedy marginal gain；固定池比较 coverage@k、目标覆盖所需 suite size 与来源组成 |
| LLM 自然度引导 | 未开始 | DFA 合法动作上的 search/beam/reranking；重新 replay；按覆盖与长度匹配比较质量、成本 |
| Case studies 与论文证据 | 未开始 | 锁定 case/model/tokenizer/seeds/budgets；development/held-out 划分；等预算 baseline/消融；逐 case 结果与原始工件 |

所有覆盖声明以 canonical DFA replay 为准。允许启发式；全局最优最小套件和 LLM 训练不在当前目标中。

研究动机见 [HANDOFF.md](HANDOFF.md)。详细历史 success criteria 保存在 [归档路线](docs/archive/gsd/ROADMAP.md)，用于参考，不要求沿用其工作流。
