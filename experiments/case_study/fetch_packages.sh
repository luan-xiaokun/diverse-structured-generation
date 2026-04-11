#!/usr/bin/env bash
set -euo pipefail

EMAIL_VALIDATOR_URL="https://github.com/JoshData/python-email-validator.git"
EMAIL_VALIDATOR_COMMIT="936aead3bf5c608f8561954e0d2955b7f97bfdad"
EMAIL_VALIDATOR_SUBDIR="email_validator"
WEBCOLORS_URL="https://github.com/ubernostrum/webcolors.git"
WEBCOLORS_COMMIT="834f77b381fad6eb31634d583894c3bc16a7ff99"
WEBCOLORS_SUBDIR="src/webcolors"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
EMAIL_VALIDATOR_TARGET_DIR="$SCRIPT_DIR/email_validator"
WEBCOLORS_TARGET_DIR="$SCRIPT_DIR/webcolors"
TMP_DIR="$(mktemp -d)"

cleanup() {
    rm -rf "$TMP_DIR"
}
trap cleanup EXIT

require_command() {
    local cmd=$1
    if ! command -v "$cmd" >/dev/null 2>&1; then
        echo "Missing required command: $cmd" >&2
        exit 1
    fi
}

export_snapshot() {
    local repo_url=$1
    local commit=$2
    local sparse_subdir=$3
    local checkout_dir=$4
    local target_dir=$5

    git clone --filter=blob:none --no-checkout "$repo_url" "$checkout_dir"
    git -C "$checkout_dir" sparse-checkout init --cone
    git -C "$checkout_dir" sparse-checkout set "$sparse_subdir"
    git -C "$checkout_dir" checkout "$commit"

    rm -rf "$target_dir"
    cp -R "$checkout_dir/$sparse_subdir" "$target_dir"
}

require_command git
require_command cp
require_command mktemp

echo "Using temporary directory: $TMP_DIR"

mkdir -p "$SCRIPT_DIR"

export_snapshot \
    "$EMAIL_VALIDATOR_URL" \
    "$EMAIL_VALIDATOR_COMMIT" \
    "$EMAIL_VALIDATOR_SUBDIR" \
    "$TMP_DIR/python-email-validator" \
    "$EMAIL_VALIDATOR_TARGET_DIR"

echo "Done."
echo "Exported $EMAIL_VALIDATOR_SUBDIR at EMAIL_VALIDATOR_COMMIT $EMAIL_VALIDATOR_COMMIT to:"
echo "  $EMAIL_VALIDATOR_TARGET_DIR"

export_snapshot \
    "$WEBCOLORS_URL" \
    "$WEBCOLORS_COMMIT" \
    "$WEBCOLORS_SUBDIR" \
    "$TMP_DIR/webcolors" \
    "$WEBCOLORS_TARGET_DIR"

echo "Done."
echo "Exported $WEBCOLORS_SUBDIR at WEBCOLORS_COMMIT $WEBCOLORS_COMMIT to:"
echo "  $WEBCOLORS_TARGET_DIR"
echo "Pinned package snapshots refreshed successfully."
