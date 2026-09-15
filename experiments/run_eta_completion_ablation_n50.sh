#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

export HF_HUB_DISABLE_TELEMETRY=1
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

SAMPLE_COUNT="${SAMPLE_COUNT:-50}"
MODEL="${MODEL:-Qwen/Qwen2.5-1.5B-Instruct}"
SEED="${SEED:-42}"
GAMMA="${GAMMA:-0.3}"
PATH_FREQUENCY_ESTIMATOR="${PATH_FREQUENCY_ESTIMATOR:-harmonic}"
DEFAULT_MAX_TOKENS="${DEFAULT_MAX_TOKENS:-24}"
JSON_MAX_TOKENS="${JSON_MAX_TOKENS:-54}"
DATA_ROOT="${DATA_ROOT:-data/extension/eta_completion_ablation_n50}"
LOG_DIR="${LOG_DIR:-logs/extension/eta_completion_ablation_n50}"
RESULT_DIR="${RESULT_DIR:-results/extension/eta_completion_ablation_n50}"
REQUIRE_CUDA="${REQUIRE_CUDA:-1}"
EXPECTED_MODEL_PARAMETER_DEVICE="${EXPECTED_MODEL_PARAMETER_DEVICE:-cuda:0}"
ALLOW_UV_CACHE_DIR="${ALLOW_UV_CACHE_DIR:-0}"
FORCE="${FORCE:-0}"
DRY_RUN="${DRY_RUN:-0}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-eta_completion_ablation}"
QUIET_GENERATION="${QUIET_GENERATION:-0}"
NO_PROGRESS="${NO_PROGRESS:-0}"

GRAMMARS=("email" "no-bomb" "json")
ETA_VALUES=("0" "1")

if [[ -n "${GRAMMARS_OVERRIDE:-}" ]]; then
  IFS=',' read -r -a GRAMMARS <<<"$GRAMMARS_OVERRIDE"
fi

if [[ -n "${ETA_VALUES_OVERRIDE:-}" ]]; then
  IFS=',' read -r -a ETA_VALUES <<<"$ETA_VALUES_OVERRIDE"
fi

guard_uv_cache_dir() {
  if [[ -n "${UV_CACHE_DIR:-}" && "$ALLOW_UV_CACHE_DIR" != "1" ]]; then
    cat >&2 <<'EOF'
[error] UV_CACHE_DIR is set. In this environment, setting UV_CACHE_DIR caused
torch.cuda.is_available() to become false and silently moved some runs to CPU.
Unset UV_CACHE_DIR before launching the ablation, or set ALLOW_UV_CACHE_DIR=1
only after a CUDA preflight confirms the model stays on cuda:0.
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

number_label() {
  local value="$1"
  value="${value//./p}"
  value="${value//- /m}"
  printf '%s\n' "$value"
}

method_name_for_eta() {
  local eta="$1"
  local gamma_label eta_label
  gamma_label="$(number_label "$GAMMA")"
  eta_label="$(number_label "$eta")"
  printf '%s_g%s_eta%s\n' "$PATH_FREQUENCY_ESTIMATOR" "$gamma_label" "$eta_label"
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

run_one() {
  local grammar="$1"
  local eta="$2"
  local method output_path max_tokens existing slug run_log_dir log_file

  method="$(method_name_for_eta "$eta")"
  output_path="$(data_path "$method" "$MODEL" "$grammar")"
  max_tokens="$(max_tokens_for_grammar "$grammar")"
  existing="$(sample_count_in_file "$output_path")"

  if [[ "$FORCE" != "1" && "$existing" -ge "$SAMPLE_COUNT" ]]; then
    if [[ "$REQUIRE_CUDA" == "1" ]]; then
      assert_expected_device "$output_path" "$EXPECTED_MODEL_PARAMETER_DEVICE"
    fi
    echo "[skip] grammar=$grammar eta=$eta method=$method samples=$existing path=$output_path"
    return 0
  fi

  if [[ "$existing" -gt 0 ]]; then
    local backup_path="${output_path}.partial-$(date +%Y%m%d-%H%M%S)"
    echo "[backup-partial] grammar=$grammar eta=$eta samples=$existing backup=$backup_path"
    if [[ "$DRY_RUN" != "1" ]]; then
      mv "$output_path" "$backup_path"
    fi
  fi

  slug="$(model_slug "$MODEL")"
  run_log_dir="$LOG_DIR/$method/$slug"
  log_file="$run_log_dir/$grammar.log"
  mkdir -p "$run_log_dir" "$(dirname "$output_path")"

  local cmd=(
    "uv" "run" "--no-sync" "python" "scripts/generate_re.py" "$grammar"
    "--model" "$MODEL"
    "-n" "$SAMPLE_COUNT"
    "--seed" "$SEED"
    "--local-files-only"
    "--max-tokens" "$max_tokens"
    "--output" "$output_path"
    "--gamma" "$GAMMA"
    "--completion-bias-eta" "$eta"
    "--path-frequency-estimator" "$PATH_FREQUENCY_ESTIMATOR"
  )
  if [[ "$NO_PROGRESS" == "1" ]]; then
    cmd+=("--no-progress")
  fi

  echo "[run] grammar=$grammar eta=$eta method=$method samples=$SAMPLE_COUNT seed=$SEED max_tokens=$max_tokens output=$output_path log=$log_file"
  printf '[cmd]'
  printf ' %q' "${cmd[@]}"
  printf '\n'

  if [[ "$DRY_RUN" == "1" ]]; then
    return 0
  fi

  printf '[cmd]' >"$log_file"
  printf ' %q' "${cmd[@]}" >>"$log_file"
  printf '\n' >>"$log_file"
  if [[ "$QUIET_GENERATION" == "1" ]]; then
    "${cmd[@]}" >>"$log_file" 2>&1
  else
    "${cmd[@]}" 2>&1 | tee -a "$log_file"
  fi

  existing="$(sample_count_in_file "$output_path")"
  if [[ "$existing" -lt "$SAMPLE_COUNT" ]]; then
    raise_msg="[error] $output_path has only $existing samples, expected $SAMPLE_COUNT."
    echo "$raise_msg" >&2
    exit 1
  fi
  if [[ "$REQUIRE_CUDA" == "1" ]]; then
    assert_expected_device "$output_path" "$EXPECTED_MODEL_PARAMETER_DEVICE"
  fi
  echo "[done] grammar=$grammar eta=$eta method=$method samples=$existing path=$output_path"
}

evaluate_outputs() {
  if [[ "$DRY_RUN" == "1" ]]; then
    echo "[dry-run] skipping evaluation"
    return 0
  fi

  DATA_ROOT="$DATA_ROOT" \
  RESULT_DIR="$RESULT_DIR" \
  MODEL="$MODEL" \
  SAMPLE_COUNT="$SAMPLE_COUNT" \
  GRAMMARS_CSV="$(IFS=','; echo "${GRAMMARS[*]}")" \
  ETA_VALUES_CSV="$(IFS=','; echo "${ETA_VALUES[*]}")" \
  GAMMA="$GAMMA" \
  PATH_FREQUENCY_ESTIMATOR="$PATH_FREQUENCY_ESTIMATOR" \
  EXPERIMENT_NAME="$EXPERIMENT_NAME" \
  uv run --no-sync python - <<'PY'
from __future__ import annotations

import csv
import json
import os
import re
from pathlib import Path
from typing import Any

from regex_dfa_guide import DiverseGuideDFA

try:
    from metrics_eval import compute_metrics
except ModuleNotFoundError:
    from scripts.metrics_eval import compute_metrics


def model_slug(model: str) -> str:
    return model.rsplit("/", maxsplit=1)[-1].lower()


def number_label(value: str) -> str:
    return value.replace(".", "p").replace("- ", "m")


def method_name(estimator: str, gamma: str, eta: str) -> str:
    return f"{estimator}_g{number_label(gamma)}_eta{number_label(eta)}"


def compile_generation_regex(regex: str) -> re.Pattern[str]:
    return re.compile(f"^(?:{regex})$")


def validity_split(samples: list[str], regex: str) -> tuple[list[str], list[str]]:
    pattern = compile_generation_regex(regex)
    valid = []
    invalid = []
    for sample in samples:
        if pattern.match(sample):
            valid.append(sample)
        else:
            invalid.append(sample)
    return valid, invalid


def safe_metrics(dfa: DiverseGuideDFA, samples: list[str]) -> dict[str, Any]:
    if samples:
        return compute_metrics(dfa, samples, d=5, s=1)
    return {
        "sample_count": 0,
        "average_length": 0.0,
        "dfa": {
            "state_count": len(dfa.get_states()),
            "transition_count": sum(map(len, dfa.get_transitions().values())),
        },
        "metrics": {
            "state_coverage": 0.0,
            "transition_coverage": 0.0,
            "path_coverage": 0.0,
            "distinct_2gram": [0, 0],
            "distinct_3gram": [0, 0],
            "vendi_score": 0.0,
        },
    }


def avg_length(samples: list[str]) -> float:
    return sum(len(sample) for sample in samples) / len(samples) if samples else 0.0


data_root = Path(os.environ["DATA_ROOT"])
result_dir = Path(os.environ["RESULT_DIR"])
model = os.environ["MODEL"]
model_dir = model_slug(model)
sample_count = int(os.environ["SAMPLE_COUNT"])
grammars = [item for item in os.environ["GRAMMARS_CSV"].split(",") if item]
eta_values = [item for item in os.environ["ETA_VALUES_CSV"].split(",") if item]
gamma = os.environ["GAMMA"]
estimator = os.environ["PATH_FREQUENCY_ESTIMATOR"]
experiment_name = os.environ["EXPERIMENT_NAME"]

result_dir.mkdir(parents=True, exist_ok=True)
records: list[dict[str, Any]] = []
dfa_cache: dict[str, DiverseGuideDFA] = {}

for grammar in grammars:
    for eta in eta_values:
        method = method_name(estimator, gamma, eta)
        path = data_root / method / model_dir / f"{grammar}.json"
        record: dict[str, Any] = {
            "method": method,
            "grammar": grammar,
            "model": model,
            "eta": float(eta),
            "gamma": float(gamma),
            "path_frequency_estimator": estimator,
            "path": str(path),
            "expected_samples": sample_count,
        }
        if not path.exists():
            record["status"] = "missing"
            records.append(record)
            continue

        with path.open(encoding="utf-8") as handle:
            data = json.load(handle)
        samples = data.get("samples", [])
        regex = data["regex"]
        valid, invalid = validity_split(samples, regex)
        if regex not in dfa_cache:
            dfa_cache[regex] = DiverseGuideDFA(f"(?:{regex})$", 2**32 - 1, {})
        metrics_result = safe_metrics(dfa_cache[regex], valid)
        metrics = metrics_result["metrics"]
        accepted = data.get("accepted_samples")
        if accepted is None:
            accepted = len(samples)
        rejected = data.get("rejected_samples") or 0
        total_attempts = accepted + rejected
        elapsed = data.get("elapsed_seconds")

        distinct_2gram = metrics.get("distinct_2gram") or [0, 0]
        distinct_3gram = metrics.get("distinct_3gram") or [0, 0]
        record.update(
            {
                "status": "complete" if len(samples) >= sample_count else "partial",
                "saved_sample_count": len(samples),
                "accepted_samples": accepted,
                "rejected_samples": rejected,
                "total_attempts": total_attempts,
                "reject_ratio": rejected / total_attempts if total_attempts else 0.0,
                "valid_sample_count": len(valid),
                "invalid_sample_count": len(invalid),
                "validity_rate": len(valid) / len(samples) if samples else 0.0,
                "average_length": avg_length(samples),
                "valid_average_length": metrics_result["average_length"],
                "elapsed_seconds": elapsed,
                "accepted_per_second": accepted / elapsed if elapsed and accepted else None,
                "seconds_per_accepted": elapsed / accepted if elapsed and accepted else None,
                "state_coverage": metrics.get("state_coverage"),
                "transition_coverage": metrics.get("transition_coverage"),
                "path_coverage": metrics.get("path_coverage"),
                "distinct_2gram_count": distinct_2gram[0],
                "distinct_2gram_total": distinct_2gram[1],
                "distinct_2gram_ratio": (
                    distinct_2gram[0] / distinct_2gram[1]
                    if len(distinct_2gram) > 1 and distinct_2gram[1]
                    else 0.0
                ),
                "distinct_3gram_count": distinct_3gram[0],
                "distinct_3gram_total": distinct_3gram[1],
                "distinct_3gram_ratio": (
                    distinct_3gram[0] / distinct_3gram[1]
                    if len(distinct_3gram) > 1 and distinct_3gram[1]
                    else 0.0
                ),
                "vendi_score": metrics.get("vendi_score"),
                "max_tokens": data.get("max_tokens"),
                "seed": data.get("seed"),
                "model_parameter_device": data.get("model_parameter_device"),
                "invalid_examples": invalid[:5],
            }
        )
        records.append(record)

summary_json = result_dir / "summary.json"
summary_csv = result_dir / "summary.csv"
with summary_json.open("w", encoding="utf-8") as handle:
    json.dump(
        {
            "schema_version": 1,
            "experiment": experiment_name,
            "parameters": {
                "model": model,
                "grammars": grammars,
                "eta_values": eta_values,
                "gamma": float(gamma),
                "path_frequency_estimator": estimator,
                "sample_count": sample_count,
            },
            "records": records,
        },
        handle,
        indent=2,
    )

fieldnames = [
    "method",
    "grammar",
    "model",
    "eta",
    "gamma",
    "path_frequency_estimator",
    "status",
    "saved_sample_count",
    "accepted_samples",
    "rejected_samples",
    "total_attempts",
    "reject_ratio",
    "validity_rate",
    "average_length",
    "valid_average_length",
    "state_coverage",
    "transition_coverage",
    "path_coverage",
    "distinct_2gram_count",
    "distinct_2gram_ratio",
    "distinct_3gram_count",
    "distinct_3gram_ratio",
    "vendi_score",
    "elapsed_seconds",
    "seconds_per_accepted",
    "max_tokens",
    "seed",
    "model_parameter_device",
    "path",
]
with summary_csv.open("w", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(records)

print(f"[eval] summary_json={summary_json}")
print(f"[eval] summary_csv={summary_csv}")
print(
    "grammar\teta\treject\tavg_len\tstate_cov\ttrans_cov\tpath_cov\t"
    "distinct2\tdistinct3\tvendi\tsec/accepted"
)
for record in records:
    print(
        f"{record.get('grammar')}\t"
        f"{record.get('eta')}\t"
        f"{record.get('reject_ratio', 0):.4f}\t"
        f"{record.get('average_length', 0):.2f}\t"
        f"{record.get('state_coverage', 0):.4f}\t"
        f"{record.get('transition_coverage', 0):.4f}\t"
        f"{record.get('path_coverage', 0):.4f}\t"
        f"{record.get('distinct_2gram_count', 0)}\t"
        f"{record.get('distinct_3gram_count', 0)}\t"
        f"{record.get('vendi_score', 0):.4f}\t"
        f"{record.get('seconds_per_accepted')}"
    )
PY
}

mkdir -p "$DATA_ROOT" "$LOG_DIR" "$RESULT_DIR"

guard_uv_cache_dir
if [[ "$DRY_RUN" != "1" ]]; then
  preflight_cuda
else
  echo "[dry-run] skipping CUDA preflight"
fi

echo "[start] experiment=$EXPERIMENT_NAME model=$MODEL sample_count=$SAMPLE_COUNT seed=$SEED gamma=$GAMMA estimator=$PATH_FREQUENCY_ESTIMATOR etas=${ETA_VALUES[*]} grammars=${GRAMMARS[*]} data_root=$DATA_ROOT result_dir=$RESULT_DIR quiet=$QUIET_GENERATION no_progress=$NO_PROGRESS"
for grammar in "${GRAMMARS[@]}"; do
  for eta in "${ETA_VALUES[@]}"; do
    run_one "$grammar" "$eta"
  done
done

evaluate_outputs
echo "[complete] eta completion ablation finished"
