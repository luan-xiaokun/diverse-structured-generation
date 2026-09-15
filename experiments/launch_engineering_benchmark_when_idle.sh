#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

FREE_VRAM_GB="${FREE_VRAM_GB:-12}"
MAX_CPU_PERCENT="${MAX_CPU_PERCENT:-20}"
GPU_INDEX="${GPU_INDEX:-auto}"
STABLE_CHECKS="${STABLE_CHECKS:-3}"
POLL_INTERVAL_SECONDS="${POLL_INTERVAL_SECONDS:-60}"
CPU_SAMPLE_SECONDS="${CPU_SAMPLE_SECONDS:-1}"
LOG_DIR="${LOG_DIR:-logs/engineering/optimization_benchmark}"

cmd=(
  "uv" "run" "--no-sync" "python" "scripts/wait_for_idle_and_launch.py"
  "--free-vram-gb" "$FREE_VRAM_GB"
  "--max-cpu-percent" "$MAX_CPU_PERCENT"
  "--gpu-index" "$GPU_INDEX"
  "--stable-checks" "$STABLE_CHECKS"
  "--poll-interval-seconds" "$POLL_INTERVAL_SECONDS"
  "--cpu-sample-seconds" "$CPU_SAMPLE_SECONDS"
  "--log-dir" "$LOG_DIR"
)

if [[ -n "${TIMEOUT_SECONDS:-}" ]]; then
  cmd+=("--timeout-seconds" "$TIMEOUT_SECONDS")
fi
if [[ "${SKIP_GPU_CHECK:-0}" == "1" ]]; then
  cmd+=("--skip-gpu-check")
fi
if [[ "${NO_SET_CUDA_VISIBLE_DEVICES:-0}" == "1" ]]; then
  cmd+=("--no-set-cuda-visible-devices")
fi
if [[ -n "${LAUNCHER_EXTRA_ARGS:-}" ]]; then
  # shellcheck disable=SC2206
  extra_args=( $LAUNCHER_EXTRA_ARGS )
  cmd+=("${extra_args[@]}")
fi

cmd+=("--command" "experiments/run_engineering_benchmark_matrix.sh")

printf '[cmd]'
printf ' %q' "${cmd[@]}"
printf '\n'
"${cmd[@]}"
