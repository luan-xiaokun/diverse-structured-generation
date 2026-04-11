#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"

DEFAULT_MODEL="Qwen/Qwen2.5-1.5B-Instruct"
DEFAULT_PPL_MODEL="microsoft/Phi-4-mini-instruct"
GRAMMARS=("email" "css-color" "json" "no-bomb" "ipv4" "ipv6" "threefold")

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

run_generation_suite() {
    local baseline_flag=${1:-false}
    shift || true
    local extra_args=("$@")

    for grammar in "${GRAMMARS[@]}"; do
        local cmd=("gen" "$grammar" "--model" "$DEFAULT_MODEL" "-n" "1000")
        while IFS= read -r arg; do
            [[ -n "$arg" ]] && cmd+=("$arg")
        done < <(grammar_extra_args "$grammar")
        cmd+=("${extra_args[@]}")
        if [[ "$baseline_flag" == "true" ]]; then
            cmd+=("--baseline")
        fi
        run_poe "${cmd[@]}"
    done
}

run_eval_suite() {
    local baseline_flag=${1:-false}
    shift || true
    local extra_args=("$@")

    for grammar in "${GRAMMARS[@]}"; do
        local cmd=("eval" "$grammar" "--model" "$DEFAULT_MODEL")
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
        local cmd=("eval-runtime" "$grammar" "--model" "$DEFAULT_MODEL")
        if [[ "$baseline_flag" == "true" ]]; then
            cmd+=("--baseline")
        fi
        run_poe "${cmd[@]}"
    done
}
