#!/usr/bin/env bash
set -euo pipefail

# Download selected Hugging Face models for local experiments.
#
# Common usage:
#   bash download_models.sh
#   bash download_models.sh --preset cfg
#   bash download_models.sh --model Qwen/Qwen2.5-1.5B-Instruct
#   bash download_models.sh --preset large --dry-run
#
# Environment:
#   HF_HOME=/path/to/hf-cache bash download_models.sh
#   HF_TOKEN=hf_xxx bash download_models.sh
#   USE_HF_MIRROR=0 bash download_models.sh
#   HF_ENDPOINT=https://hf-mirror.com bash download_models.sh

DEFAULT_HF_HOME="$HOME/.cache/huggingface"
HF_HOME="${HF_HOME:-$DEFAULT_HF_HOME}"
LOG_DIR="${LOG_DIR:-./download_logs}"
PRESET="all"
DRY_RUN=0
FORCE=0
USE_HF_MIRROR="${USE_HF_MIRROR:-1}"

declare -a REQUESTED_MODELS=()

usage() {
  cat <<'EOF'
Usage:
  bash download_models.sh [options]

Options:
  --preset NAME       Model preset: cfg, dfa, large, all. Default: all.
  --model MODEL_ID    Download one model. Can be repeated. Overrides preset.
  --hf-home PATH      Hugging Face cache root. Default: $HF_HOME or ~/.cache/huggingface.
  --log-dir PATH      Download log directory. Default: ./download_logs.
  --official          Use the official Hugging Face endpoint.
  --mirror            Use https://hf-mirror.com unless HF_ENDPOINT is already set.
  --force             Run download even if a local snapshot already exists.
  --dry-run           Print planned downloads and exit.
  -h, --help          Show this help.

Presets:
  cfg      Qwen/Qwen2.5-1.5B-Instruct
  dfa      Qwen/Qwen2.5-0.5B-Instruct, Qwen/Qwen2.5-1.5B-Instruct
  large    Qwen/Qwen3-8B, microsoft/Phi-4-mini-instruct
  all      cfg + dfa + large

Notes:
  - The script preserves the standard Hugging Face cache layout.
  - Existing snapshots are skipped unless --force is used.
  - Set HF_TOKEN for gated/private models.
  - Set HF_HUB_ENABLE_HF_TRANSFER=1 only if hf_transfer is installed.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --preset)
      PRESET="${2:?--preset requires a value}"
      shift 2
      ;;
    --model)
      REQUESTED_MODELS+=("${2:?--model requires a value}")
      shift 2
      ;;
    --hf-home)
      HF_HOME="${2:?--hf-home requires a path}"
      shift 2
      ;;
    --log-dir)
      LOG_DIR="${2:?--log-dir requires a path}"
      shift 2
      ;;
    --official)
      USE_HF_MIRROR=0
      shift
      ;;
    --mirror)
      USE_HF_MIRROR=1
      shift
      ;;
    --force)
      FORCE=1
      shift
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

export HF_HOME
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-0}"
export HF_HUB_DISABLE_TELEMETRY=1

if [[ "$USE_HF_MIRROR" == "1" ]]; then
  export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
elif [[ -z "${HF_ENDPOINT:-}" ]]; then
  unset HF_ENDPOINT || true
fi

models_for_preset() {
  case "$1" in
    cfg)
      printf '%s\n' "Qwen/Qwen2.5-1.5B-Instruct"
      ;;
    dfa)
      printf '%s\n' \
        "Qwen/Qwen2.5-0.5B-Instruct" \
        "Qwen/Qwen2.5-1.5B-Instruct"
      ;;
    large)
      printf '%s\n' \
        "Qwen/Qwen3-8B" \
        "microsoft/Phi-4-mini-instruct"
      ;;
    all)
      printf '%s\n' \
        "Qwen/Qwen2.5-0.5B-Instruct" \
        "Qwen/Qwen2.5-1.5B-Instruct" \
        "Qwen/Qwen3-8B" \
        "microsoft/Phi-4-mini-instruct"
      ;;
    *)
      echo "Unknown preset: $1" >&2
      echo "Allowed presets: cfg, dfa, large, all" >&2
      exit 2
      ;;
  esac
}

dedupe_models() {
  awk '!seen[$0]++'
}

safe_model_name() {
  printf '%s' "$1" | tr '/:' '__'
}

snapshot_is_complete() {
  local snapshot="$1"
  python - "$snapshot" <<'PY'
from __future__ import annotations

import glob
import json
import os
import sys

snapshot = sys.argv[1]
if not os.path.exists(os.path.join(snapshot, "config.json")):
    raise SystemExit(1)

index_path = os.path.join(snapshot, "model.safetensors.index.json")
if os.path.exists(index_path):
    with open(index_path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    shards = set(data.get("weight_map", {}).values())
    if shards and all(os.path.exists(os.path.join(snapshot, shard)) for shard in shards):
        raise SystemExit(0)
    raise SystemExit(1)

patterns = (
    "*.safetensors",
    "pytorch_model*.bin",
)
for pattern in patterns:
    if glob.glob(os.path.join(snapshot, pattern)):
        raise SystemExit(0)
raise SystemExit(1)
PY
}

has_snapshot() {
  local model="$1"
  local safe_name
  safe_name="$(printf '%s' "$model" | sed 's#/#--#g')"
  local model_dir="$HF_HOME/hub/models--$safe_name"
  [[ -d "$model_dir/snapshots" ]] || return 1
  local snapshot
  while IFS= read -r snapshot; do
    if snapshot_is_complete "$snapshot"; then
      return 0
    fi
  done < <(find "$model_dir/snapshots" -mindepth 1 -maxdepth 1 -type d)
  return 1
}

find_hf_cmd() {
  if command -v hf >/dev/null 2>&1; then
    printf '%s\n' "hf"
    return
  fi
  if command -v huggingface-cli >/dev/null 2>&1; then
    printf '%s\n' "huggingface-cli"
    return
  fi
  if command -v uv >/dev/null 2>&1 && uv run --no-sync hf --help >/dev/null 2>&1; then
    printf '%s\n' "uv run --no-sync hf"
    return
  fi
  return 1
}

if [[ "${#REQUESTED_MODELS[@]}" -gt 0 ]]; then
  mapfile -t MODELS < <(printf '%s\n' "${REQUESTED_MODELS[@]}" | dedupe_models)
else
  mapfile -t MODELS < <(models_for_preset "$PRESET" | dedupe_models)
fi

mkdir -p "$LOG_DIR"

echo "HF_HOME=$HF_HOME"
echo "HF_ENDPOINT=${HF_ENDPOINT:-official}"
echo "HF_HUB_ENABLE_HF_TRANSFER=$HF_HUB_ENABLE_HF_TRANSFER"
echo "Preset=$PRESET"
echo "Force=$FORCE"
echo "Dry run=$DRY_RUN"
echo "Logs=$LOG_DIR"
echo
echo "Models:"
printf '  - %s\n' "${MODELS[@]}"
echo

if [[ "$DRY_RUN" == "1" ]]; then
  exit 0
fi

NEEDS_DOWNLOAD=0
if [[ "$FORCE" == "1" ]]; then
  NEEDS_DOWNLOAD=1
else
  for model in "${MODELS[@]}"; do
    if ! has_snapshot "$model"; then
      NEEDS_DOWNLOAD=1
      break
    fi
  done
fi

if [[ "$NEEDS_DOWNLOAD" == "0" ]]; then
  echo "All requested model snapshots already exist."
  exit 0
fi

if ! HF_CMD="$(find_hf_cmd)"; then
  echo "Hugging Face CLI not found." >&2
  echo "Install dependencies with:" >&2
  echo "  uv sync" >&2
  echo "or:" >&2
  echo "  python -m pip install -U 'huggingface_hub[cli]' hf_transfer" >&2
  exit 1
fi

python - <<'PY'
try:
    import hf_transfer  # noqa: F401
    print("hf_transfer: available")
except Exception:
    print("hf_transfer: not available; download will still work, but may be slower")
PY

download_one() {
  local model="$1"
  local safe_name
  safe_name="$(safe_model_name "$model")"
  local log_file="$LOG_DIR/${safe_name}.log"

  if [[ "$FORCE" != "1" ]] && has_snapshot "$model"; then
    echo "Skipping existing model: $model"
    echo
    return 0
  fi

  echo "============================================================"
  echo "Downloading: $model"
  echo "Log: $log_file"
  echo "Started: $(date)"
  echo "============================================================"

  # Keep standard HF cache layout; local-files-only experiment commands rely on it.
  # shellcheck disable=SC2086
  local status=0
  $HF_CMD download "$model" 2>&1 | tee "$log_file" || status=$?
  if [[ "$status" != "0" ]]; then
    echo "Download command failed for $model with status $status" >&2
    return "$status"
  fi
  if ! has_snapshot "$model"; then
    echo "Download did not produce a local snapshot for $model" >&2
    return 1
  fi

  echo "Finished: $model at $(date)"
  echo
}

declare -a FAILED=()
for model in "${MODELS[@]}"; do
  if ! download_one "$model"; then
    FAILED+=("$model")
  fi
done

if [[ "${#FAILED[@]}" -gt 0 ]]; then
  echo "Some downloads failed:" >&2
  printf '  - %s\n' "${FAILED[@]}" >&2
  exit 1
fi

echo "All downloads finished at $(date)."
echo "Cache location: $HF_HOME"
