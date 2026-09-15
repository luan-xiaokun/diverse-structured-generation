#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

export HF_HUB_DISABLE_TELEMETRY=1
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

SAMPLE_COUNT="${SAMPLE_COUNT:-100}"
SEED="${SEED:-42}"
DEFAULT_MAX_TOKENS="${DEFAULT_MAX_TOKENS:-24}"
JSON_MAX_TOKENS="${JSON_MAX_TOKENS:-72}"
DATA_ROOT="${DATA_ROOT:-data/extension/refinement_phase1_harmonic_min_screen_n100}"
LOG_DIR="${LOG_DIR:-logs/extension/refinement_phase1_harmonic_min_screen_n100}"
SUMMARY_DIR="${SUMMARY_DIR:-results/extension/refinement_phase1_harmonic_min_screen_n100}"
SUMMARY_FILE="${SUMMARY_FILE:-$SUMMARY_DIR/summary.jsonl}"
REQUIRE_CUDA="${REQUIRE_CUDA:-1}"
EXPECTED_MODEL_PARAMETER_DEVICE="${EXPECTED_MODEL_PARAMETER_DEVICE:-cuda:0}"
ALLOW_UV_CACHE_DIR="${ALLOW_UV_CACHE_DIR:-0}"
LOCAL_FILES_ONLY="${LOCAL_FILES_ONLY:-1}"
FORCE="${FORCE:-0}"

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

guard_uv_cache_dir() {
  if [[ -n "${UV_CACHE_DIR:-}" && "$ALLOW_UV_CACHE_DIR" != "1" ]]; then
    cat >&2 <<'EOF'
[error] UV_CACHE_DIR is set. In this environment, setting UV_CACHE_DIR caused
torch.cuda.is_available() to become false in previous runs.
Unset UV_CACHE_DIR before launching the matrix, or set ALLOW_UV_CACHE_DIR=1 only
after a CUDA preflight confirms the model stays on cuda:0.
EOF
    exit 2
  fi
}

preflight_cuda() {
  if [[ "$REQUIRE_CUDA" != "1" ]]; then
    echo "[preflight] REQUIRE_CUDA=$REQUIRE_CUDA, skipping CUDA preflight"
    return 0
  fi

  uv run --no-sync python - <<'PY'
from __future__ import annotations

import sys

import torch

print(f"[preflight] python={sys.executable}")
print(f"[preflight] torch_cuda={torch.version.cuda}")
print(f"[preflight] cuda_available={torch.cuda.is_available()}")
print(f"[preflight] cuda_device_count={torch.cuda.device_count()}")
if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
    raise SystemExit("[error] CUDA is required but is not visible to torch.")
PY
}

model_slug() {
  local model="$1"
  local slug="${model##*/}"
  printf '%s\n' "${slug,,}"
}

gamma_label() {
  local gamma="$1"
  gamma="${gamma//./p}"
  gamma="${gamma//-/m}"
  printf 'g%s\n' "$gamma"
}

method_name() {
  local estimator="$1"
  local gamma="$2"
  printf '%s_%s\n' "$estimator" "$(gamma_label "$gamma")"
}

max_tokens_for_grammar() {
  local grammar="$1"
  if [[ "$grammar" == "json" ]]; then
    printf '%s\n' "$JSON_MAX_TOKENS"
  else
    printf '%s\n' "$DEFAULT_MAX_TOKENS"
  fi
}

data_path() {
  local method="$1"
  local model="$2"
  local grammar="$3"
  local slug
  slug="$(model_slug "$model")"
  printf '%s/%s/%s/%s.json\n' "$DATA_ROOT" "$method" "$slug" "$grammar"
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

assert_expected_device() {
  local output_path="$1"
  local expected_device="$2"
  python - "$output_path" "$expected_device" <<'PY'
from __future__ import annotations

import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
expected = sys.argv[2]
with path.open(encoding="utf-8") as handle:
    data = json.load(handle)
actual = data.get("model_parameter_device")
if actual != expected:
    raise SystemExit(
        f"[error] {path} recorded model_parameter_device={actual!r}, "
        f"expected {expected!r}."
    )
PY
}

write_summary() {
  local method="$1"
  local estimator="$2"
  local gamma="$3"
  local model="$4"
  local grammar="$5"
  local output_path="$6"
  local status="$7"
  local max_tokens="$8"

  python - "$method" "$estimator" "$gamma" "$model" "$grammar" "$output_path" "$status" "$SEED" "$SUMMARY_FILE" "$max_tokens" <<'PY'
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

(
    method,
    estimator,
    gamma,
    model,
    grammar,
    output_path,
    status,
    seed,
    summary_file,
    max_tokens,
) = sys.argv[1:]
path = Path(output_path)
summary_path = Path(summary_file)
summary_path.parent.mkdir(parents=True, exist_ok=True)

record = {
    "experiment": "refinement_phase1_harmonic_min",
    "method": method,
    "path_frequency_estimator_configured": estimator,
    "gamma_configured": float(gamma),
    "completion_bias_eta_configured": 0.0,
    "model": model,
    "grammar": grammar,
    "path": output_path,
    "status": status,
    "seed": int(seed) if seed else None,
    "configured_max_tokens": int(max_tokens),
}

try:
    record["git_commit"] = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True
    ).strip()
except Exception:
    record["git_commit"] = None

if path.exists():
    with path.open(encoding="utf-8") as handle:
        data = json.load(handle)
    accepted = data.get("accepted_samples") or len(data.get("samples", []))
    rejected = data.get("rejected_samples") or 0
    attempts = accepted + rejected
    elapsed = data.get("elapsed_seconds")
    seconds_per_accepted = elapsed / accepted if elapsed and accepted else None
    record.update(
        {
            "accepted_samples": accepted,
            "rejected_samples": rejected,
            "total_attempts": attempts,
            "attempted_batches": data.get("attempted_batches"),
            "elapsed_seconds": elapsed,
            "accepted_per_second": accepted / elapsed if elapsed else None,
            "seconds_per_accepted": seconds_per_accepted,
            "reject_ratio": rejected / attempts if attempts else 0.0,
            "max_tokens": data.get("max_tokens"),
            "top_k": data.get("top_k"),
            "top_p": data.get("top_p"),
            "temperature": data.get("temperature"),
            "gamma": data.get("gamma"),
            "completion_bias_eta": data.get("completion_bias_eta"),
            "path_frequency_estimator": data.get("path_frequency_estimator"),
            "baseline_backend": data.get("baseline_backend"),
            "device": data.get("device"),
            "model_parameter_device": data.get("model_parameter_device"),
        }
    )

with summary_path.open("a", encoding="utf-8") as handle:
    handle.write(json.dumps(record, sort_keys=True) + "\n")

print(json.dumps(record, sort_keys=True))
PY
}

run_one() {
  local model="$1"
  local grammar="$2"
  local estimator="$3"
  local gamma="$4"
  local method
  method="$(method_name "$estimator" "$gamma")"
  local output_path
  output_path="$(data_path "$method" "$model" "$grammar")"

  local max_tokens
  max_tokens="$(max_tokens_for_grammar "$grammar")"

  local existing
  existing="$(sample_count_in_file "$output_path")"
  if [[ "$FORCE" != "1" && "$existing" -ge "$SAMPLE_COUNT" ]]; then
    if [[ "$REQUIRE_CUDA" == "1" ]]; then
      assert_expected_device "$output_path" "$EXPECTED_MODEL_PARAMETER_DEVICE"
    fi
    echo "[skip] method=$method model=$model grammar=$grammar samples=$existing"
    write_summary "$method" "$estimator" "$gamma" "$model" "$grammar" "$output_path" "skipped" "$max_tokens"
    return 0
  fi
  if [[ "$existing" -gt 0 ]]; then
    local backup_path="${output_path}.partial-$(date +%Y%m%d-%H%M%S)"
    echo "[backup-partial] method=$method model=$model grammar=$grammar samples=$existing backup=$backup_path"
    mv "$output_path" "$backup_path"
  fi

  local slug
  slug="$(model_slug "$model")"
  local run_log_dir="$LOG_DIR/$method/$slug"
  mkdir -p "$run_log_dir" "$(dirname "$output_path")"
  local log_file="$run_log_dir/$grammar.log"

  echo "[run] method=$method estimator=$estimator gamma=$gamma eta=0 model=$model grammar=$grammar target=$SAMPLE_COUNT seed=$SEED max_tokens=$max_tokens output=$output_path log=$log_file"
  local cmd=(
    "uv" "run" "--no-sync" "python" "scripts/generate_re.py" "$grammar"
    "--model" "$model"
    "-n" "$SAMPLE_COUNT"
    "--seed" "$SEED"
    "--max-tokens" "$max_tokens"
    "--output" "$output_path"
    "--gamma" "$gamma"
    "--completion-bias-eta" "0.0"
    "--path-frequency-estimator" "$estimator"
  )
  if [[ "$LOCAL_FILES_ONLY" == "1" ]]; then
    cmd+=("--local-files-only")
  fi

  printf '[cmd]' >"$log_file"
  printf ' %q' "${cmd[@]}" >>"$log_file"
  printf '\n' >>"$log_file"
  "${cmd[@]}" 2>&1 | tee -a "$log_file"

  existing="$(sample_count_in_file "$output_path")"
  if [[ "$REQUIRE_CUDA" == "1" ]]; then
    assert_expected_device "$output_path" "$EXPECTED_MODEL_PARAMETER_DEVICE"
  fi
  echo "[done] method=$method model=$model grammar=$grammar samples=$existing"
  write_summary "$method" "$estimator" "$gamma" "$model" "$grammar" "$output_path" "done" "$max_tokens"
}

mkdir -p "$DATA_ROOT" "$LOG_DIR" "$SUMMARY_DIR"
: >"$SUMMARY_FILE"

guard_uv_cache_dir
preflight_cuda

echo "[start] phase1 data_root=$DATA_ROOT log_dir=$LOG_DIR summary_file=$SUMMARY_FILE sample_count=$SAMPLE_COUNT seed=$SEED default_max_tokens=$DEFAULT_MAX_TOKENS json_max_tokens=$JSON_MAX_TOKENS require_cuda=$REQUIRE_CUDA expected_device=$EXPECTED_MODEL_PARAMETER_DEVICE"
for model in "${MODELS[@]}"; do
  for grammar in "${GRAMMARS[@]}"; do
    for estimator in "${ESTIMATORS[@]}"; do
      for gamma in "${GAMMAS[@]}"; do
        run_one "$model" "$grammar" "$estimator" "$gamma"
      done
    done
  done
done

echo "[complete] Phase 1 generation matrix finished summary=$SUMMARY_FILE"
