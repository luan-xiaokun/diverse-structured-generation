#!/usr/bin/env bash
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"

TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-1800}"
ABLATION_COMPONENTS=("reward" "penalty" "range_scaling")

run_generation_with_timeout() {
    local component=$1
    timeout "$TIMEOUT_SECONDS" uv run poe gen css-color \
        --model "$DEFAULT_MODEL" \
        -n 1000 \
        --ablation-component "$component"
}

# generate samples with ablation of reward, penalty, and range scaling components
for component in "${ABLATION_COMPONENTS[@]}"; do
    run_generation_with_timeout "$component"
done

# evaluate the default and ablated runs
run_poe eval css-color \
    --model "$DEFAULT_MODEL" \
    --experiment component_ablation \
    --output results/component_ablation/default/css-color.json
for component in "${ABLATION_COMPONENTS[@]}"; do
    run_poe eval css-color \
        --model "$DEFAULT_MODEL" \
        --experiment component_ablation \
        --ablation-component "$component" \
        --output "results/component_ablation/$component/css-color.json"
done
