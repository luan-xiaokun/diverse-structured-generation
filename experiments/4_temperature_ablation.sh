#!/usr/bin/env bash
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"

TEMPERATURE="1.5"
COMMON_ABLATION_ARGS=("--temperature" "$TEMPERATURE")
PPL_ARGS=(
    "--temperature" "$TEMPERATURE"
    "--ppl-model" "$DEFAULT_PPL_MODEL"
    "--ppl"
)

# diverse generation with temperature 1.5
run_generation_suite false "${COMMON_ABLATION_ARGS[@]}"

# baseline generation with temperature 1.5
run_generation_suite true "${COMMON_ABLATION_ARGS[@]}"

# evaluate diverse samples with diversity metrics and perplexity
run_eval_suite false "${PPL_ARGS[@]}"

# evaluate baseline samples with diversity metrics and perplexity
run_eval_suite true "${PPL_ARGS[@]}"
