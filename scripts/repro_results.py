"""Helpers for structured reproduction result files."""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def get_git_commit() -> str | None:
    """Return the current git commit if the repository metadata is available."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    commit = result.stdout.strip()
    return commit or None


def build_metadata(extra: dict[str, Any] | None = None) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "timestamp_utc": datetime.now(UTC)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z"),
        "git_commit": get_git_commit(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
    }
    if extra:
        metadata.update(extra)
    return metadata


def write_json(path: str | Path, data: dict[str, Any]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
        f.write("\n")
