#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"

DEFAULT_MODEL="Qwen/Qwen2.5-1.5B-Instruct"
DEFAULT_PPL_MODEL="microsoft/Phi-4-mini-instruct"
DEFAULT_SEED="${SEED:-}"
GRAMMARS=("email" "css-color" "json" "no-bomb")

cd "$REPO_ROOT"

run_poe() {
    uv run poe "$@"
}

grammar_extra_args() {
    local grammar=$1
    case "$grammar" in
        json)
            printf '%s\n' "--max-tokens" "54"
            ;;
    esac
}

seed_args() {
    if [[ -n "$DEFAULT_SEED" ]]; then
        printf '%s\n' "--seed" "$DEFAULT_SEED"
    fi
}

result_setting_dir() {
    local baseline_flag=${1:-false}
    if [[ "$baseline_flag" == "true" ]]; then
        printf '%s\n' "baseline"
    else
        printf '%s\n' "diverse"
    fi
}

metric_result_path() {
    local experiment=$1
    local baseline_flag=$2
    local grammar=$3
    local suffix=${4:-}
    local setting
    setting="$(result_setting_dir "$baseline_flag")"
    if [[ -n "$suffix" ]]; then
        printf 'results/%s/%s/%s/%s.json\n' "$experiment" "$setting" "$suffix" "$grammar"
    else
        printf 'results/%s/%s/%s.json\n' "$experiment" "$setting" "$grammar"
    fi
}

runtime_result_path() {
    local baseline_flag=$1
    local grammar=$2
    local setting
    setting="$(result_setting_dir "$baseline_flag")"
    printf 'results/runtime/%s/%s.json\n' "$setting" "$grammar"
}

run_generation_suite() {
    local baseline_flag=${1:-false}
    shift || true
    local extra_args=("$@")

    for grammar in "${GRAMMARS[@]}"; do
        local cmd=("gen" "$grammar" "--model" "$DEFAULT_MODEL" "-n" "1000")
        while IFS= read -r arg; do
            [[ -n "$arg" ]] && cmd+=("$arg")
        done < <(grammar_extra_args "$grammar")
        while IFS= read -r arg; do
            [[ -n "$arg" ]] && cmd+=("$arg")
        done < <(seed_args)
        cmd+=("${extra_args[@]}")
        if [[ "$baseline_flag" == "true" ]]; then
            cmd+=("--baseline")
        fi
        run_poe "${cmd[@]}"
    done
}

run_eval_suite() {
    local baseline_flag=${1:-false}
    if (($# > 0)); then
        shift
    fi
    local experiment="diversity"
    local result_suffix=""
    local experiment_consumed=false
    if (($# > 0)); then
        case "$1" in
            ""|"diversity"|"temperature_ablation"|"component_ablation")
                experiment=${1:-diversity}
                experiment_consumed=true
                shift
                ;;
        esac
    fi
    if [[ "$experiment_consumed" == "true" ]] && (($# > 0)); then
        result_suffix=$1
        shift
    fi
    local extra_args=("$@")

    for grammar in "${GRAMMARS[@]}"; do
        local output_path
        output_path="$(metric_result_path "$experiment" "$baseline_flag" "$grammar" "$result_suffix")"
        local cmd=("eval" "$grammar" "--model" "$DEFAULT_MODEL" "--experiment" "$experiment" "--output" "$output_path")
        cmd+=("${extra_args[@]}")
        if [[ "$baseline_flag" == "true" ]]; then
            cmd+=("--baseline")
        fi
        run_poe "${cmd[@]}"
    done
}

run_runtime_suite() {
    local baseline_flag=${1:-false}

    for grammar in "${GRAMMARS[@]}"; do
        local output_path
        output_path="$(runtime_result_path "$baseline_flag" "$grammar")"
        local cmd=("eval-runtime" "$grammar" "--model" "$DEFAULT_MODEL" "--output" "$output_path")
        while IFS= read -r arg; do
            [[ -n "$arg" ]] && cmd+=("$arg")
        done < <(seed_args)
        if [[ "$baseline_flag" == "true" ]]; then
            cmd+=("--baseline")
        fi
        run_poe "${cmd[@]}"
    done
}
