#!/usr/bin/env bash
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"

# evaluate diverse samples with diversity metrics
run_eval_suite false

# evaluate baseline samples with diversity metrics
run_eval_suite true
