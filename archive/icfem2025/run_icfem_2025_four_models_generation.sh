#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

export UV_CACHE_DIR="${UV_CACHE_DIR:-$REPO_ROOT/.uv-cache}"
export HF_HUB_DISABLE_TELEMETRY=1
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

SAMPLE_COUNT="${SAMPLE_COUNT:-1000}"
LOG_DIR="${LOG_DIR:-logs/icfem_2025_four_models_generation}"

MODELS=(
  "Qwen/Qwen2.5-0.5B-Instruct"
  "Qwen/Qwen2.5-1.5B-Instruct"
  "Qwen/Qwen3-8B"
  "microsoft/Phi-4-mini-instruct"
)

GRAMMARS=(
  "email"
  "css-color"
  "json"
  "no-bomb"
)

model_slug() {
  local model="$1"
  local slug="${model##*/}"
  printf '%s\n' "${slug,,}"
}

data_path() {
  local setting="$1"
  local model="$2"
  local grammar="$3"
  local slug
  slug="$(model_slug "$model")"
  printf 'data/%s/%s/%s.json\n' "$setting" "$slug" "$grammar"
}

sample_count_in_file() {
  local path="$1"
  python - "$path" <<'PY'
from __future__ import annotations

import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
if not path.exists():
    print(0)
    raise SystemExit(0)
try:
    with path.open(encoding="utf-8") as handle:
        data = json.load(handle)
    print(len(data.get("samples", [])))
except Exception:
    print(0)
PY
}

grammar_extra_args=()
set_grammar_extra_args() {
  local grammar="$1"
  grammar_extra_args=()
  if [[ "$grammar" == "json" ]]; then
    grammar_extra_args=("--max-tokens" "54")
  fi
}

run_one() {
  local setting="$1"
  local model="$2"
  local grammar="$3"
  local baseline_flag="$4"
  local output_path
  output_path="$(data_path "$setting" "$model" "$grammar")"

  local existing
  existing="$(sample_count_in_file "$output_path")"
  if [[ "$existing" -ge "$SAMPLE_COUNT" ]]; then
    echo "[skip] setting=$setting model=$model grammar=$grammar samples=$existing"
    return 0
  fi
  if [[ "$existing" -gt 0 ]]; then
    local backup_path="${output_path}.partial-$(date +%Y%m%d-%H%M%S)"
    echo "[backup] setting=$setting model=$model grammar=$grammar samples=$existing backup=$backup_path"
    mv "$output_path" "$backup_path"
  fi

  set_grammar_extra_args "$grammar"

  local slug
  slug="$(model_slug "$model")"
  local run_log_dir="$LOG_DIR/$setting/$slug"
  mkdir -p "$run_log_dir"
  local log_file="$run_log_dir/$grammar.log"

  echo "[run] setting=$setting model=$model grammar=$grammar target=$SAMPLE_COUNT existing=$existing log=$log_file"
  local cmd=(
    "uv" "run" "poe" "gen" "$grammar"
    "--model" "$model"
    "-n" "$SAMPLE_COUNT"
  )
  cmd+=("${grammar_extra_args[@]}")
  if [[ "$baseline_flag" == "true" ]]; then
    cmd+=("--baseline")
  fi
  "${cmd[@]}" >"$log_file" 2>&1

  existing="$(sample_count_in_file "$output_path")"
  echo "[done] setting=$setting model=$model grammar=$grammar samples=$existing"
}

for model in "${MODELS[@]}"; do
  for grammar in "${GRAMMARS[@]}"; do
    run_one "diverse" "$model" "$grammar" "false"
  done
  for grammar in "${GRAMMARS[@]}"; do
    run_one "baseline" "$model" "$grammar" "true"
  done
done

echo "[complete] generation matrix finished"
