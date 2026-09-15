#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

DATA_ROOT="${DATA_ROOT:-data/extension/refinement_phase1_harmonic_min_screen_n100}"
EVAL_DIR="${EVAL_DIR:-results/extension/refinement_phase1_harmonic_min_screen_n100/evaluation}"
EXPECTED_SAMPLES="${EXPECTED_SAMPLES:-100}"
TREND_STEP="${TREND_STEP:-50}"
VENDI_D="${VENDI_D:-5}"
VENDI_S="${VENDI_S:-1}"
STRICT="${STRICT:-0}"

MODELS=(
  "Qwen/Qwen2.5-1.5B-Instruct"
)

GRAMMARS=(
  "email"
  "css-color"
  "json"
  "no-bomb"
  "semver"
  "api-route"
)

ESTIMATORS=(
  "min"
  "harmonic"
)

GAMMAS=(
  "0.1"
  "0.2"
  "0.3"
)

if [[ -n "${MODELS_OVERRIDE:-}" ]]; then
  IFS=',' read -r -a MODELS <<<"$MODELS_OVERRIDE"
fi
if [[ -n "${GRAMMARS_OVERRIDE:-}" ]]; then
  IFS=',' read -r -a GRAMMARS <<<"$GRAMMARS_OVERRIDE"
fi
if [[ -n "${ESTIMATORS_OVERRIDE:-}" ]]; then
  IFS=',' read -r -a ESTIMATORS <<<"$ESTIMATORS_OVERRIDE"
fi
if [[ -n "${GAMMAS_OVERRIDE:-}" ]]; then
  IFS=',' read -r -a GAMMAS <<<"$GAMMAS_OVERRIDE"
fi

gamma_label() {
  local gamma="$1"
  gamma="${gamma//./p}"
  gamma="${gamma//-/m}"
  printf 'g%s\n' "$gamma"
}

methods_csv() {
  local first=1
  local method
  for estimator in "${ESTIMATORS[@]}"; do
    for gamma in "${GAMMAS[@]}"; do
      method="${estimator}_$(gamma_label "$gamma")"
      if [[ "$first" == "1" ]]; then
        first=0
        printf '%s' "$method"
      else
        printf ',%s' "$method"
      fi
    done
  done
}

join_csv() {
  local first=1
  local item
  for item in "$@"; do
    if [[ "$first" == "1" ]]; then
      first=0
      printf '%s' "$item"
    else
      printf ',%s' "$item"
    fi
  done
}

METHODS_OVERRIDE="${METHODS_OVERRIDE:-$(methods_csv)}"
MODELS_OVERRIDE="${MODELS_OVERRIDE:-$(join_csv "${MODELS[@]}")}"
GRAMMARS_OVERRIDE="${GRAMMARS_OVERRIDE:-$(join_csv "${GRAMMARS[@]}")}"

cmd=(
  "uv" "run" "--no-sync" "python" "scripts/evaluate_main_matrix.py"
  "--data-root" "$DATA_ROOT"
  "--output-dir" "$EVAL_DIR"
  "--models" "$MODELS_OVERRIDE"
  "--grammars" "$GRAMMARS_OVERRIDE"
  "--methods" "$METHODS_OVERRIDE"
  "--expected-samples" "$EXPECTED_SAMPLES"
  "--step" "$TREND_STEP"
  "-d" "$VENDI_D"
  "-s" "$VENDI_S"
)

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
