#!/usr/bin/env bash
set -euo pipefail

CURRENT_REPO="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"

export HF_HUB_DISABLE_TELEMETRY=1
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

OUTPUT_ROOT="${OUTPUT_ROOT:-$CURRENT_REPO/results/engineering/diverseguide_plus_strict}"
RUNS="${RUNS:-5}"
WARMUP_RUNS="${WARMUP_RUNS:-1}"
TARGET_GENERATED_TOKENS="${TARGET_GENERATED_TOKENS:-256}"
DEFAULT_MAX_TOKENS="${DEFAULT_MAX_TOKENS:-24}"
JSON_MAX_TOKENS="${JSON_MAX_TOKENS:-72}"
SEED="${SEED:-42}"
LOCAL_FILES_ONLY="${LOCAL_FILES_ONLY:-1}"
REQUIRE_CUDA="${REQUIRE_CUDA:-1}"
FORCE="${FORCE:-0}"
EXPECTED_CURRENT_COMMIT="${EXPECTED_CURRENT_COMMIT:-$(git -C "$CURRENT_REPO" rev-parse HEAD)}"

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

if [[ -n "${MODELS_OVERRIDE:-}" ]]; then
  IFS=',' read -r -a MODELS <<<"$MODELS_OVERRIDE"
fi
if [[ -n "${GRAMMARS_OVERRIDE:-}" ]]; then
  IFS=',' read -r -a GRAMMARS <<<"$GRAMMARS_OVERRIDE"
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
  local model="$1"
  local grammar="$2"
  local slug
  slug="$(model_slug "$model")"
  local output_path="$OUTPUT_ROOT/raw/$slug/$grammar.json"
  if [[ "$FORCE" != "1" && -s "$output_path" ]]; then
    echo "[skip] model=$model grammar=$grammar output=$output_path"
    return 0
  fi

  local max_tokens
  max_tokens="$(max_tokens_for_grammar "$grammar")"
  mkdir -p "$(dirname "$output_path")"

  local cmd=(
    "uv" "run" "--no-sync" "python"
    "scripts/benchmark_diverseguide_plus_strict.py"
    "$grammar"
    "--model" "$model"
    "--repo" "$CURRENT_REPO"
    "--grammar-source-repo" "$CURRENT_REPO"
    "--expected-current-commit" "$EXPECTED_CURRENT_COMMIT"
    "--runs" "$RUNS"
    "--warmup-runs" "$WARMUP_RUNS"
    "--target-generated-tokens" "$TARGET_GENERATED_TOKENS"
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

  echo "[run] model=$model grammar=$grammar max_tokens=$max_tokens target_tokens=$TARGET_GENERATED_TOKENS"
  printf '[cmd]'
  printf ' %q' "${cmd[@]}"
  printf '\n'
  "${cmd[@]}"
}

write_summary() {
  python3 - "$OUTPUT_ROOT" <<'PY'
import csv
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
records = []
for path in sorted((root / "raw").glob("*/*.json")):
    obj = json.loads(path.read_text())
    summary = obj["summary"]
    grammar_dfa = obj["grammar_dfa"]
    method_dfa = obj.get("method_dfa") or {}
    tokenizer = obj["tokenizer"]
    model_loading = obj["model_loading"]
    implementation = obj["implementation"]
    record = {
        "model": obj["model"],
        "grammar": obj["grammar"],
        "method": obj["method"],
        "git_commit": implementation["git_commit"],
        "grammar_source_commit": obj["grammar_source_commit"],
        "rust_extension_file": implementation["rust_extension_file"],
        "guide_rust_file": implementation["guide_rust_file"],
        "vocab_size": tokenizer["vocab_size"],
        "score_width": tokenizer["score_width"],
        "grammar_state_count": grammar_dfa["state_count"],
        "grammar_byte_transition_count": grammar_dfa["byte_transition_count"],
        "method_state_count": method_dfa.get("state_count"),
        "method_byte_transition_count": method_dfa.get("byte_transition_count"),
        "method_token_transition_count": method_dfa.get("token_transition_count"),
        "max_tokens": obj["parameters"]["max_tokens"],
        "target_generated_tokens": obj["parameters"]["target_generated_tokens"],
        "runs": obj["parameters"]["runs"],
        "warmup_runs": obj["parameters"]["warmup_runs"],
        "build_seconds_mean": summary["build_seconds"]["mean"],
        "build_seconds_std": summary["build_seconds"]["std"],
        "generated_token_count_mean": summary["generated_token_count"]["mean"],
        "call_count_mean": summary["call_count"]["mean"],
        "tokens_per_second_mean": summary["tokens_per_second"]["mean"],
        "tokens_per_second_std": summary["tokens_per_second"]["std"],
        "avg_inter_token_latency_ms_mean": summary["avg_inter_token_latency_ms"]["mean"],
        "avg_inter_token_latency_ms_std": summary["avg_inter_token_latency_ms"]["std"],
        "inter_token_latency_p50_ms_mean": summary["inter_token_latency_p50_ms"]["mean"],
        "inter_token_latency_p90_ms_mean": summary["inter_token_latency_p90_ms"]["mean"],
        "inter_token_latency_p95_ms_mean": summary["inter_token_latency_p95_ms"]["mean"],
        "ttft_ms_mean": summary["ttft_ms"]["mean"],
        "ttft_ms_std": summary["ttft_ms"]["std"],
        "total_inter_token_count": summary["total_inter_token_count"],
        "model_load_seconds": model_loading["model_load_seconds"],
        "tokenizer_load_seconds": model_loading["tokenizer_load_seconds"],
        "model_parameter_device": model_loading["model_parameter_device"],
        "path": str(path),
    }
    records.append(record)

root.mkdir(parents=True, exist_ok=True)
(root / "summary.json").write_text(json.dumps(records, indent=2, sort_keys=True) + "\n")
with (root / "summary.jsonl").open("w") as f:
    for record in records:
        f.write(json.dumps(record, sort_keys=True) + "\n")

csv_columns = [
    "model",
    "grammar",
    "build_seconds_mean",
    "build_seconds_std",
    "avg_inter_token_latency_ms_mean",
    "avg_inter_token_latency_ms_std",
    "inter_token_latency_p50_ms_mean",
    "inter_token_latency_p90_ms_mean",
    "inter_token_latency_p95_ms_mean",
    "tokens_per_second_mean",
    "generated_token_count_mean",
    "call_count_mean",
    "total_inter_token_count",
    "grammar_state_count",
    "grammar_byte_transition_count",
    "method_token_transition_count",
    "vocab_size",
]
with (root / "summary.csv").open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=csv_columns)
    writer.writeheader()
    for record in records:
        writer.writerow({column: record[column] for column in csv_columns})

md_lines = [
    "| Model | Grammar | Build ms | ITL ms | ITL p50 | ITL p90 | TPS | Intervals | DFA states | Method token transitions |",
    "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
]
for record in records:
    md_lines.append(
        "| {model} | {grammar} | {build_ms:.1f} | {itl:.1f} | {p50:.1f} | "
        "{p90:.1f} | {tps:.2f} | {intervals} | {states} | {token_transitions} |".format(
            model=record["model"].split("/")[-1],
            grammar=record["grammar"],
            build_ms=record["build_seconds_mean"] * 1000.0,
            itl=record["avg_inter_token_latency_ms_mean"],
            p50=record["inter_token_latency_p50_ms_mean"],
            p90=record["inter_token_latency_p90_ms_mean"],
            tps=record["tokens_per_second_mean"],
            intervals=record["total_inter_token_count"],
            states=record["grammar_state_count"],
            token_transitions=record["method_token_transition_count"],
        )
    )
(root / "summary.md").write_text("\n".join(md_lines) + "\n")
print(f"[summary] wrote {root / 'summary.json'}")
print(f"[summary] wrote {root / 'summary.jsonl'}")
print(f"[summary] wrote {root / 'summary.csv'}")
print(f"[summary] wrote {root / 'summary.md'}")
PY
}

mkdir -p "$OUTPUT_ROOT/raw"

echo "[start] output_root=$OUTPUT_ROOT"
echo "[config] current_repo=$CURRENT_REPO expected_current_commit=$EXPECTED_CURRENT_COMMIT runs=$RUNS warmup_runs=$WARMUP_RUNS target_generated_tokens=$TARGET_GENERATED_TOKENS"
for model in "${MODELS[@]}"; do
  for grammar in "${GRAMMARS[@]}"; do
    run_one "$model" "$grammar"
  done
done

write_summary
echo "[complete] strict diverse-guide+ benchmark finished output_root=$OUTPUT_ROOT"
