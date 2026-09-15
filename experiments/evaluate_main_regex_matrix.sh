#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

DATA_ROOT="${DATA_ROOT:-data/extension/main_matrix_500}"
EVAL_DIR="${EVAL_DIR:-results/extension/main_matrix_500/evaluation}"
EXPECTED_SAMPLES="${EXPECTED_SAMPLES:-500}"
TREND_STEP="${TREND_STEP:-50}"
VENDI_D="${VENDI_D:-5}"
VENDI_S="${VENDI_S:-1}"
STRICT="${STRICT:-0}"

cmd=(
  "uv" "run" "--no-sync" "python" "scripts/evaluate_main_matrix.py"
  "--data-root" "$DATA_ROOT"
  "--output-dir" "$EVAL_DIR"
  "--expected-samples" "$EXPECTED_SAMPLES"
  "--step" "$TREND_STEP"
  "-d" "$VENDI_D"
  "-s" "$VENDI_S"
)

if [[ -n "${MODELS_OVERRIDE:-}" ]]; then
  cmd+=("--models" "$MODELS_OVERRIDE")
fi
if [[ -n "${GRAMMARS_OVERRIDE:-}" ]]; then
  cmd+=("--grammars" "$GRAMMARS_OVERRIDE")
fi
if [[ -n "${METHODS_OVERRIDE:-}" ]]; then
  cmd+=("--methods" "$METHODS_OVERRIDE")
fi
if [[ -n "${MAX_SAMPLES:-}" ]]; then
  cmd+=("--max-samples" "$MAX_SAMPLES")
fi
if [[ "$STRICT" == "1" ]]; then
  cmd+=("--strict")
fi

printf '[cmd]'
printf ' %q' "${cmd[@]}"
printf '\n'
"${cmd[@]}"
