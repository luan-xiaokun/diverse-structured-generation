#!/usr/bin/env bash
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"

# runtime evaluation for diverse generation
run_runtime_suite false

# runtime evaluation for the baseline
run_runtime_suite true
