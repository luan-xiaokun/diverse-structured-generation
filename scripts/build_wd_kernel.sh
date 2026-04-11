#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if command -v python >/dev/null 2>&1; then
  PYTHON_BIN="python"
elif command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN="python3"
else
  echo "No python interpreter found in PATH (expected python or python3)." >&2
  exit 127
fi

"${PYTHON_BIN}" "${SCRIPT_DIR}/build_wd_kernel.py" "$@"
