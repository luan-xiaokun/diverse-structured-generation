# AIAP 当前状态

更新：2026-10-05。GSD 已弃用，项目使用普通 Markdown 记录与直接开发协作。

## 当前工作位置

单-obligation baseline 与 vocab/byte DFA 修复已完成。下一项工作是 multi-obligation grouping/search，尚未设计或实现；需要先明确算法与实验协议。

历史阶段 1 和 01.1 已完成（共 10 个计划）；这只是历史交付统计，不代表剩余研究工作的比例。最近产品代码提交为 `d5618da`。

## 已完成

- 模块化单-target BFS、canonical replay、实际覆盖 certificates、CLI 与 JSON/JSONL。
- Native 0.3.0 bytes-only vocabulary、严格 self-contained artifacts、token/direct byte trace parity。
- Missing-only obligations、reachability 分类、typed witness outcomes、reachable-only target selection。
- Reachability 性能修复：对 supplied missing meanings 建索引，一次扫描 accepting actions。
- 2026-07-24 历史验证：12/12 must-haves，Rust 15/15，hermetic pytest 89 passed/3 deselected，锁定 Qwen 离线 parity 通过。本次仅更新文档，未重新测试。

## 已锁定的语义

- Obligation 仅指选定 baseline samples 未访问的结构。
- `--sample-num` 控制 baseline prefix；`--obligation-num` 控制 reachable targets 尝试数量。
- 覆盖包括 state、byte transition、相邻有序 state-pair path。
- `byte_level_unreachable` 保留为 aggregate diagnostic，不占生成 slots；JSONL 只含 witness 尝试 certificates。
- Replay 是有效性与实际覆盖的唯一依据；搜索预测不算成功。
- Bytes-only API 和 artifact 错误严格失败；tokenizer revision 锁定。

## 待办与问题

1. 设计 multi-obligation 分组和联合可行性判定；明确 coverage labels、dominance/pruning、预算上限与小规模 exact oracle。
2. 设计共享候选池上的互补 suite selection 与固定池 baseline 比较。
3. 明确 LLM 引导方法、model lock、长度匹配的自然度评价。
4. 固定 development/held-out case 划分及等预算、可复现的实验协议。

当前 `pyproject.toml` 已注册 `real_case_parity` marker；历史验证报告中的未注册警告不再是配置层面的待办。

## 恢复入口

读 [HANDOFF.md](HANDOFF.md) 与 [ROADMAP.md](ROADMAP.md)，然后直接讨论/设计 multi-obligation 算法。已归档的 [历史资料](docs/archive/gsd/README.md) 可供查证，不再参与工作流。

本次迁移保留未跟踪的 `regex_dfa_guide/uv.lock`。研究目标和产品行为不受迁移影响。
