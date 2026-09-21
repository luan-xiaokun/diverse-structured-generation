# EOS reentry：服务器交接

本轮实现 token-boundary accepting-entry EOS bonus。默认仍为 `legacy`；
`constant` 的接受态 EOS 信号为 1，`entry_count` 为本序列 N_F。
非 EOS completion、mask、原 reward/penalty、global counter 提交规则不变。
本地完成实现验证，**尚未运行 LLM smoke 或效果实验**。

## 1. 版本与交付内容

- 交付 ref：`refs/heads/eos-reentry`。精确 commit、parent、bundle SHA-256
  见随包 `RELEASE.json`；外层本文件也标出 checkout hash。
- 基线/parent：`2c3b2a297ab089c979b836543b2efea6f8c1d773`。
  完整 bundle 包含该分支全部祖先，无 prerequisite；不需服务器已有此仓库。
- `changes.patch` 是 base 到交付 commit 的完整 diff。
- Rust DFA 源码/API 未改，包含在历史与源码中，必须本机重建扩展。
  新计数位于 Python processor，每 row 动态加 EOS bonus，不缓存为 state 常量。
- 本次在独立 worktree 开发，未提交其它 worktree 的改动。
- 对 frozen source 的逐文件核验、测试范围见 `docs/eos-reentry-validation.md`。
- 不包含模型、虚拟环境、本机构建 `.so` 或全部历史结果。

## 2. 校验、隔离 checkout、构建

从解压交付目录执行（路径均应选择新的独立目录）：

```bash
sha256sum -c SHA256SUMS
EOS_COMMIT=$(python3 -c 'import json; print(json.load(open("RELEASE.json"))["commit"])')
git clone --branch eos-reentry eos-reentry.bundle /srv/eos-reentry-source
cd /srv/eos-reentry-source
git checkout --detach "$EOS_COMMIT"
git bundle verify /path/to/delivery/eos-reentry.bundle
git status --porcelain
export EOS_EVIDENCE=/srv/eos-reentry-build-evidence
mkdir -p "$EOS_EVIDENCE/wheels"
uv sync --locked --group dev --group outlines
.venv/bin/maturin build --release --locked --interpreter .venv/bin/python \
  -m regex_dfa_guide/Cargo.toml --out "$EOS_EVIDENCE/wheels"
uv pip install --python .venv/bin/python --force-reinstall \
  "$EOS_EVIDENCE"/wheels/regex_dfa_guide-*.whl
.venv/bin/python scripts/build_wd_kernel.py
.venv/bin/python -m pytest -q --no-cov
cargo test --locked --manifest-path regex_dfa_guide/Cargo.toml
git rev-parse HEAD > "$EOS_EVIDENCE/commit.txt"
git status --porcelain > "$EOS_EVIDENCE/status.txt"
git diff --binary > "$EOS_EVIDENCE/dirty.patch"
sha256sum uv.lock regex_dfa_guide/Cargo.lock \
  "$EOS_EVIDENCE"/wheels/*.whl build/native/wd_kernel/wd_kernel.so \
  > "$EOS_EVIDENCE/build.sha256"
.venv/bin/python - <<'PY' > "$EOS_EVIDENCE/runtime.json"
import json
from scripts.run_group1_task import environment_snapshot
print(json.dumps(environment_snapshot(), indent=2))
PY
```

需要 Linux、Python 3.12、uv、Rust/Cargo、C compiler/OpenMP、兼容 PyTorch
2.8.0 的 CUDA driver；依赖来自 `uv.lock`，不能安装另一份历史 wheel 代替重建。
本地验证 Python 3.12.13 / torch 2.8.0 / transformers 5.3.0 / outlines 1.2.12。
普通 Git clone 不包括 Python/Rust 依赖缓存；首次构建需要已有缓存或可下载依赖。

锁文件 SHA-256：

```text
847829d853ba80d26011d4e613d906afb31688a1ad9ae8bfdb0c30a8758b18b1  uv.lock
3520269e116fb6e18a0e181c984d8c837e7383089119b16ee0d032aeaacad382  regex_dfa_guide/Cargo.lock
```

## 3. 模型、冻结配置、资源

使用本地完整 `Qwen/Qwen2.5-1.5B-Instruct` snapshot，revision
`989aa7980e4cf806f80c7fef2b1adb7bc71aa306`，tokenizer 同源。
模型权重由 server 从该 revision 获取/核验；runner 校验原 manifest 中六个模型/
tokenizer 配置文件 hashes，不把配置校验冒充全部权重校验。保存权重校验和到 evidence。

`configs/eos-reentry/base.json` 复制 Group2 manifest 的 regex、raw prompt、完整
generation config 和模型 metadata hashes。γ=.3、harmonic、temperature=1.0、
top_k=0、top_p=1、repetition_penalty=1、batch=1、FP16；JSON=72 tokens，
Email/CSS=24。此次配置不会沿用旧 Qwen 默认 temperature=.7。

```bash
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=2
export EOS_RUN=/srv/results/eos-reentry-20260921-v1
export EOS_BACKUP=/backup/eos-reentry-20260921-v1
export EOS_MODEL=/srv/models/Qwen2.5-1.5B-Instruct/989aa7980e4cf806f80c7fef2b1adb7bc71aa306
.venv/bin/python -m scripts.eos_reentry prepare \
  --root "$EOS_RUN" --backup "$EOS_BACKUP" --model-path "$EOS_MODEL"
```

`prepare` 要求干净、已提交的 checkout 和全新 run root；冻结全部 tracked source
hashes、环境版本/native module path/hash、49 个 task 配置及其指纹。manifest
也复制到 backup。冻结后不改源码、模型配置、环境或 manifest，变更应开新实验版本。
输出、备份必须互不包含，建议不同磁盘/故障域。

每 worker 仅一个可见 CUDA GPU，启动时要求可用 RAM≥8 GiB、GPU free≥8 GiB
且≥总显存20%、结果盘及备份盘各有≥10 GiB 余量。以上是拒绝启动阈值，不是峰值
使用量保证。默认串行；先观察 smoke 峰值，再决定独立 run 的并发，不增加 batch。
任务上限 50,000 attempts / 12 小时；触限记录 incomplete，不能静默跳过或放宽。
模型、torch/CUDA 环境、编译缓存另占磁盘，不能只按结果阈值分配整台机器的空间。

## 4. Smoke → B → C → D

本轮提前冻结 **全部45个正式 run 新跑**，不自动复用/拼接旧 legacy。
这避免用相同 Git HEAD 代替历史构建等价性核验。

| 阶段 | 配置 | run 数 | 每 run 有效样本 |
| --- | --- | ---: | ---: |
| smoke | JSON，seed42，η=0/2，legacy/entry_count | 4 | 20 |
| B | JSON，η=1/2，legacy/entry_count，seeds42–46 | 20 | 500 |
| C | JSON，η=2，constant，seeds42–46 | 5 | 500 |
| D | Email/CSS，η=1，legacy/entry_count，seeds42–46 | 20 | 500 |

```bash
.venv/bin/python -m scripts.eos_reentry run-stage --manifest "$EOS_RUN/manifest.json" --stage smoke
.venv/bin/python -m scripts.eos_reentry run-stage --manifest "$EOS_RUN/manifest.json" --stage B
.venv/bin/python -m scripts.eos_reentry aggregate --manifest "$EOS_RUN/manifest.json"
# 检查 B 的日志、逐 seed 结果和失败模式，发现异常应保留结果并交回问题。
# 完成检查后明确记录结论，再执行：
.venv/bin/python -m scripts.eos_reentry run-stage --manifest "$EOS_RUN/manifest.json" --stage C --reviewed-b
.venv/bin/python -m scripts.eos_reentry run-stage --manifest "$EOS_RUN/manifest.json" --stage D --reviewed-b
.venv/bin/python -m scripts.eos_reentry aggregate --manifest "$EOS_RUN/manifest.json"
```

正式阶段要求 smoke 完整且 η=0 两模式样本逐条一致；不一致会停止，不自动豁免。
若服务器发生数值非确定性，先记录并调查，用固定 logits 回归检查定位，再决定后续。
C/D 要求 B 全部校验完成和显式 `--reviewed-b`；D 还要求 C 完成。
`worker`/原 `run_group1_task` 是底层单任务入口，不能用来绕过阶段审查。
无改善也保留结果，不因显著性或预期胜负追加 seeds。

## 5. 输出、日志与恢复

路径均在 `$EOS_RUN` 下，smoke task ID 自带 `smoke__`，不计入汇总：

- `manifest.json`：冻结配置/源码/运行环境。
- `production/data/<task-id>.json`：500（smoke20）个有效样本及完整参数、指纹。
- `production/tasks/<task-id>/status.json`：任务状态、环境、实际 native 路径/hash。
- `production/tasks/<task-id>/attempt-*/`：configuration、attempts.jsonl、evaluation、状态。
- `aggregate/report.json` 与四张 CSV：逐 run、mean/sample SD、逐 seed 差值和差值汇总。
  部分完成会列 incomplete，只有45个正式 run 通过校验才标 complete。

每次失败尝试也保留 token IDs/text/valid/终止原因、首次 F、N_F、离开 F 次数、
EOS 分层机会数/选择数；所有输出 token 成本含 EOS、不含 prompt。
JSON η2 seed42 的三模式完整500样本 run 启用 step probes（另有两项 smoke）。
记录 processor 边界、temperature/top-k/p 前的 raw/final logits，仅保存 selected/EOS，
`completion_bonus=eta*signal` 尚未乘 gamma/range；最终增量由完整公式解释。
EOS 被 mask 时仍记录 raw_logit，以 `masked=true`、`final_logit=null` 表示其不可选。
分层键 `4` 表示 N_F≥4，不能漏掉分母。
探针在提交新样本的 global counters 之前完成末 token 记录，不消耗 RNG。

每行 journal flush/fsync；结果原子写；每 task writer lock；至少约30秒/检查点及
任务完成/失败时复制备份。不是从进程中间状态恢复：中断后保留旧 attempt 目录，
同 seed 整个 run 重新生成，绝不追加 partial 样本。重新执行同一 stage 会校验并跳过
完整结果，未完成且没有最终 JSON 的任务另建 attempt 目录。若已有最终 JSON 但校验/
评估失败，runner 拒绝覆盖；先归档调查，在独立新版本重跑，不能手改 complete 状态。
备份恢复应恢复到原绝对路径并校验所有 hashes；不要直接移动后改 manifest 路径。

指标沿用 DFA state/transition/path、平均字符长度、Distinct-2/3、Vendi WD-shift
d=5,s=1；evaluation 内含 coverage trend。覆盖率差值单位为百分点；reject/失败类别
是以全部 attempts 为分母的比例。mean/SD 的重复单位是 seed，不是500个字符串。
EOS 分层有选择偏差；共享 seeds 的结果是探索性确认；共享 GPU 时间不能当加速结论。
报告 entry_count–legacy、entry_count–constant，并同时检查长度、多样性与覆盖代价。

## 6. 历史诊断脚本与回传

历史数据可选，不影响运行新实验。`--data-root` 指向解包的
`groups1-4-results-and-analysis-20260921` 根目录：

```bash
.venv/bin/python scripts/eos_reference/find-example.py --data-root /archive/groups1-4-results-and-analysis-20260921 --output "$EOS_EVIDENCE/find-example.json"
.venv/bin/python scripts/eos_reference/probes.py --data-root /archive/groups1-4-results-and-analysis-20260921 --output "$EOS_EVIDENCE/legacy-probes.json"
.venv/bin/python scripts/eos_reference/replay.py --data-root /archive/groups1-4-results-and-analysis-20260921 --output "$EOS_EVIDENCE/replay.json"
```

可显式指定 `--source-root` 和 `--native-module /path/to/extension.so`；默认当前 checkout
及当前环境重建模块。每份输出附 `.source.json`，保存 commit/dirty、module/hash。
这些重放旧 tokens；probes 测原生 legacy distance bonus，**不代表新采样轨迹**。
旧轨迹上的 `--assert-no-reversal` 可能失败，不是新 EOS 计数回归失败。

回传每阶段完成/失败清单（本轮 reused=0）、整个 run root、build evidence、备份状态、
SHA-256 清单、B 审查结论及副作用分析。保留 Groups1–4 原数据；由用户/reports 决定
是否采用机制或更新正式实验，不宣称论文结果已经更新。
