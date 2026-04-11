#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/.venv"
cd "$SCRIPT_DIR"

if [[ ! -x "$VENV_DIR/bin/python" ]]; then
    echo "Expected virtual environment at $VENV_DIR" >&2
    echo "Run 'uv sync' in $SCRIPT_DIR first." >&2
    exit 1
fi

source "$VENV_DIR/bin/activate"

cleanup_old_outputs() {
    rm -f .coverage .coverage.*.json case_study_summary.json
}

require_file() {
    local path=$1
    if [[ ! -f "$path" ]]; then
        echo "Expected file not found: $path" >&2
        exit 1
    fi
}

run_test_with_coverage() {
    local setting=$1
    local grammar=$2
    local source=$3
    local coverage_file=".coverage.$setting.$grammar"
    local coverage_json_file="${coverage_file}.json"
    local test_file="tests/test_$source.py"
    local html_dir="htmlcov_${setting}_${grammar}"
    local dataset_filename="${setting}-${grammar}.json"

    require_file "$test_file"
    require_file "data/$dataset_filename"

    echo "Running tests for $source with $setting test cases..."
    TEST_DATA_FILENAME="$dataset_filename" python -m coverage run \
        --data-file="$coverage_file" \
        --source "$source" \
        -m pytest "$test_file" -qq
    python -m coverage report --data-file="$coverage_file"
    python -m coverage html -d "$html_dir" --data-file="$coverage_file"
    python -m coverage json --data-file="$coverage_file" -o "$coverage_json_file" >/dev/null
}

cleanup_old_outputs

run_test_with_coverage "baseline" "email" "email_validator"
run_test_with_coverage "diverse" "email" "email_validator"

run_test_with_coverage "baseline" "css-color" "webcolors"
run_test_with_coverage "diverse" "css-color" "webcolors"

python report_case_study.py
echo "Case study summary written to case_study_summary.json."
