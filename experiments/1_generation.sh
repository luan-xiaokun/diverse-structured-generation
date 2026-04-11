#!/usr/bin/env bash
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"

# diverse generation with 1000 samples per grammar
run_generation_suite false

# baseline generation with 1000 samples per grammar
run_generation_suite true
