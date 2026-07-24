#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from huggingface_hub import snapshot_download


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--lock", required=True)
    a = p.parse_args(argv)
    lock = json.loads(Path(a.lock).read_text())
    if lock.get("schema_version") != 1 or not isinstance(lock.get("tokenizers"), dict):
        raise SystemExit("invalid tokenizer lock schema")
    for key, item in lock.get("tokenizers", {}).items():
        rev = item.get("revision", "")
        files = item.get("files")
        if not item.get("repository") or not isinstance(files, list) or not files:
            raise SystemExit(f"invalid tokenizer lock entry: {key}")
        if len(rev) != 40 or any(c not in "0123456789abcdef" for c in rev):
            raise SystemExit(f"invalid revision for {key}")
        print(f"provisioning {item['repository']}@{rev}")
        root = Path(
            snapshot_download(
                repo_id=item["repository"],
                revision=rev,
                allow_patterns=files,
                local_dir=None,
            )
        )
        missing = [name for name in files if not (root / name).is_file()]
        if missing:
            raise SystemExit(
                f"incomplete tokenizer snapshot for {key}: missing {', '.join(missing)}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
