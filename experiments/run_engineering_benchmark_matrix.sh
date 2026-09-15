#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

export HF_HUB_DISABLE_TELEMETRY=1
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

OUTPUT_ROOT="${OUTPUT_ROOT:-results/engineering/optimization_benchmark}"
RUNS="${RUNS:-5}"
WARMUP_RUNS="${WARMUP_RUNS:-1}"
DEFAULT_MAX_TOKENS="${DEFAULT_MAX_TOKENS:-24}"
JSON_MAX_TOKENS="${JSON_MAX_TOKENS:-72}"
SEED="${SEED:-42}"
LOCAL_FILES_ONLY="${LOCAL_FILES_ONLY:-1}"
REQUIRE_CUDA="${REQUIRE_CUDA:-1}"
FORCE="${FORCE:-0}"

MODELS=(
  "Qwen/Qwen2.5-0.5B-Instruct"
  "Qwen/Qwen2.5-1.5B-Instruct"
  "microsoft/Phi-4-mini-instruct"
)

GRAMMARS=(
  "email"
  "css-color"
  "json"
  "no-bomb"
  "semver"
  "api-route"
)

METHODS=(
  "diverse-guide-plus"
  "diverse-guide"
  "outlines-baseline"
)

if [[ -n "${MODELS_OVERRIDE:-}" ]]; then
  IFS=',' read -r -a MODELS <<<"$MODELS_OVERRIDE"
fi
if [[ -n "${GRAMMARS_OVERRIDE:-}" ]]; then
  IFS=',' read -r -a GRAMMARS <<<"$GRAMMARS_OVERRIDE"
fi
if [[ -n "${METHODS_OVERRIDE:-}" ]]; then
  IFS=',' read -r -a METHODS <<<"$METHODS_OVERRIDE"
fi

model_slug() {
  local model="$1"
  local slug="${model##*/}"
  printf '%s\n' "${slug,,}"
}

max_tokens_for_grammar() {
  local grammar="$1"
  if [[ "$grammar" == "json" ]]; then
    printf '%s\n' "$JSON_MAX_TOKENS"
  else
    printf '%s\n' "$DEFAULT_MAX_TOKENS"
  fi
}

run_one() {
  local method="$1"
  local model="$2"
  local grammar="$3"
  local slug
  slug="$(model_slug "$model")"
  local output_path="$OUTPUT_ROOT/raw/$method/$slug/$grammar.json"
  if [[ "$FORCE" != "1" && -s "$output_path" ]]; then
    echo "[skip] method=$method model=$model grammar=$grammar output=$output_path"
    return 0
  fi

  local max_tokens
  max_tokens="$(max_tokens_for_grammar "$grammar")"
  mkdir -p "$(dirname "$output_path")"

  local cmd=(
    "uv" "run" "--no-sync" "python" "scripts/benchmark_generation_engineering.py"
    "$grammar"
    "--model" "$model"
    "--method" "$method"
    "--runs" "$RUNS"
    "--warmup-runs" "$WARMUP_RUNS"
    "--max-tokens" "$max_tokens"
    "--seed" "$SEED"
    "--output" "$output_path"
  )
  if [[ "$LOCAL_FILES_ONLY" == "1" ]]; then
    cmd+=("--local-files-only")
  fi
  if [[ "$REQUIRE_CUDA" == "1" ]]; then
    cmd+=("--require-cuda")
  fi
  if [[ -n "${EXTRA_BENCHMARK_ARGS:-}" ]]; then
    # shellcheck disable=SC2206
    local extra_args=( $EXTRA_BENCHMARK_ARGS )
    cmd+=("${extra_args[@]}")
  fi

  printf '[cmd]'
  printf ' %q' "${cmd[@]}"
  printf '\n'
  "${cmd[@]}"
}

mkdir -p "$OUTPUT_ROOT/raw"

echo "[start] output_root=$OUTPUT_ROOT runs=$RUNS warmup_runs=$WARMUP_RUNS"
for model in "${MODELS[@]}"; do
  for grammar in "${GRAMMARS[@]}"; do
    for method in "${METHODS[@]}"; do
      run_one "$method" "$model" "$grammar"
    done
  done
done

uv run --no-sync python scripts/collect_engineering_benchmarks.py \
  --input-root "$OUTPUT_ROOT/raw" \
  --output-json "$OUTPUT_ROOT/summary.json" \
  --output-jsonl "$OUTPUT_ROOT/summary.jsonl"

echo "[complete] engineering benchmark matrix finished output_root=$OUTPUT_ROOT"
