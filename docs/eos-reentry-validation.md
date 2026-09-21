# EOS reentry 实现验收（2026-09-21）

## 基线与移植来源

从 FAC `2c3b2a297ab089c979b836543b2efea6f8c1d773` 建立 `eos-reentry` 分支。
逐字节比对 Group2 frozen source 后确认以下三个核心文件与该 commit 相同：

| 文件 | frozen SHA-256 |
| --- | --- |
| src/diverse_guide/guide_rust.py | c5855d1f9fc253c60e4dd26d573c41df376af413e29da35393faf082a33d245e |
| regex_dfa_guide/src/diverse_guide_dfa.rs | bf95427771f66b6d78ae3de99a2d454bcd7b12f8ace5222b0a19501633233d0e |
| scripts/generate_re.py | f0f51eae4fd80d4d13eee683abf31397484f11a35e12809a907e28db9617a382 |

服务器 frozen tree 还有未包含在基线 Git 中的安全 runner/validation，已明确移植：

| frozen 文件 | 原 SHA-256 | 处理 |
| --- | --- | --- |
| scripts/run_group1_task.py | 8de2e6a41f0e2d1c55fe05d57a0c3cf43758fb6b4481e0855c2074c3c677ea05 | 复制并扩展 mode、指纹/环境、诊断、资源检查、输出分离 |
| scripts/generation_validation.py | 2b2393aacdc05d72b005fae5872580aa5a3e8c074a6329e45fcc2c2dd7016fcf | 保留原 CSS superset 预筛选，避免病理正则耗时 |
| scripts/evaluate_main_matrix.py | 46769b817bd7d59910150bff355c7614e89b4ee130fdaf164c3c7e6637d3ea39 | 移植 validation 入口，附加 EOS metadata，指标公式不改 |

配置来自 Group2 manifest，SHA-256
`b30c80202bfac57e1328b775d4f75e938590ec3bd615293b3eef2522a756c16b`。
三个诊断脚本由用户交接包移植，移除硬编码旧 wheel 路径并保存来源记录。
Git HEAD 相同不等于服务器构建相同；此次不复用历史统计 runs。

## 实现范围

- Rust-backed 与 Python reference processor 都有 `legacy/constant/entry_count`。
  每 row 的标量只按 token 两端状态增长；EOS/sentinel/重复 prefix 不多计；reset
  按初始接受态初始化。Rust 的增量 prefix catch-up 与 Python 已缓存 prefix 复用保留。
- 新参数接入公开工厂/生成 CLI、结果 metadata、配置指纹、完成结果校验。
  mask-only/Outlines 不接受非 legacy 生成 CLI 配置。
- Rust native API 未改，不使用其 state/byte penalty counters 代替 N_F。
  processor 无序列 fork/beam reorder 接口；未新增此功能。native DFA fork 仍只负责
  共享 index/重置 counters，其原回归测试保留。
- 工程基准的 Rust-only subclass 原来调用已过时的 `_allowed_token_tensor`
  三参数形式，修正到实际双参数签名并在目标 device 建立 adjustment tensor；
  新增 alternate-path EOS 测试。两项原私有测试 fixture 同步已有 native API 名称/
  prefix 状态，不改变生产计数语义。

## 测试与证据

先加入新模式计数/分数、probe 和 runner matrix 的失败测试，再实现。
本地隔离 `.venv` 通过 `uv sync --locked --group dev --group outlines` 构建原生扩展。
构建可选 C/OpenMP WD kernel 后，执行：

```bash
.venv/bin/python scripts/build_wd_kernel.py
.venv/bin/python -m pytest -q --no-cov
cargo test --locked --manifest-path regex_dfa_guide/Cargo.toml
.venv/bin/ruff check scripts/benchmark_generation_engineering.py scripts/evaluate_main_matrix.py scripts/generate_re.py scripts/generation_validation.py scripts/run_group1_task.py scripts/eos_reentry.py scripts/eos_reference src/diverse_guide/guide_python.py src/diverse_guide/guide_rust.py src/diverse_guide/eos_diagnostics.py tests/test_guide.py tests/test_eos_reentry.py tests/test_eos_runner.py
git diff --check
```

Python：319 passed；Rust：34 passed；改动文件 lint 和 diff whitespace 检查通过。
Python 的2项第三方 SWIG DeprecationWarning 不影响结果。仓库级 Ruff 仍有18项
原有问题，集中在未修改的 plotting/compare_live_dfa_coverage/resume 等脚本，未扩展
本次工单去改它们。完整 bundle 临时 clone/rebuild 结果另外记录在外层 TEST-REPORT.md。

覆盖接受态集共享计数、初始接受态、F→F、自环、token 内部穿过 F、EOS/sentinel、
重复调用、双 row、reset/失败重试、冷/热缓存、no-range、single EOS、range=0、
eta/gamma=0、非 EOS logits 精确不变。默认 legacy 与基线 frozen processor 在连续
有效/失败尝试中的 logits 和 native counters 精确一致；该测试从 bundle 祖先读取
原 processor，不依赖外部 frozen 目录。

真实 fixture 使用 seed42/attempt1403 的原 token IDs/文本/regex，保存来源 journal
hash 和必要 token-byte 映射；两种实现均在19/38/57处得到 N_F=1/2/3及对应 EOS bonus。
fixture 采用精简词表，验证 token 路径与计数/分数，不能代表完整词表的采样分布。
probe 回归验证 logits 和 Torch RNG 状态不变，最终 EOS 在 counter commit 前被记录。
runner 测试覆盖49项矩阵、mode 指纹、结果及日志损坏拒绝、失败类别分母、token 成本、
EOS 分层分母和 attempt 重复/缺号拒绝。

用本机重建 native、完整历史 tokenizer 运行可移植 `find-example.py` 和 `probes.py`：
复现 attempt1403 在19 tokens形成完整 JSON、下一 token离开 F；全图检查4,640,419条
token records，最大公式误差 `1.9868214962137642e-08`，接受态的 legacy completion
bonus 仍为0。此为历史诊断复现，不能用来断言新机制效果。

本地没有运行 Qwen LLM smoke/45项统计矩阵，也没有 GPU/FP16 效果结论。
服务器按 SERVER-HANDOFF.md 构建、验证 smoke 后才运行确认实验；结果允许无改善。
