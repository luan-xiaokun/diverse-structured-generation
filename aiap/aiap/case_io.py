"""Validated explicit case-study input and tokenizer provenance."""
from __future__ import annotations
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

LOCK_PATH = Path(__file__).resolve().parents[1] / "tokenizer-lock.json"
@dataclass(frozen=True)
class CaseSpec:
    path: Path; regex: str; samples: list[str]; tokenizer_key: str; tokenizer_repository: str; tokenizer_revision: str; grammar: str; raw: dict[str, Any]
def load_lock(path: str | Path = LOCK_PATH) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8")); entries = payload.get("tokenizers", {})
    if payload.get("schema_version") != 1 or not isinstance(entries, dict): raise ValueError("invalid tokenizer lock schema")
    for key, item in entries.items():
        rev = item.get("revision", "")
        if not item.get("repository") or len(rev) != 40 or any(c not in "0123456789abcdef" for c in rev): raise ValueError(f"invalid immutable tokenizer lock entry: {key}")
    return payload
def load_case(path: str | Path, lock_path: str | Path = LOCK_PATH) -> CaseSpec:
    case_path = Path(path).expanduser().resolve()
    if not case_path.is_file(): raise ValueError(f"case JSON does not exist: {case_path}")
    try: raw = json.loads(case_path.read_text(encoding="utf-8"))
    except Exception as exc: raise ValueError(f"invalid case JSON: {exc}") from exc
    if not isinstance(raw, dict) or not isinstance(raw.get("regex"), str) or not isinstance(raw.get("samples"), list) or not all(isinstance(s, str) for s in raw["samples"]): raise ValueError("case requires regex and string samples")
    entries = load_lock(lock_path)["tokenizers"]; key = raw.get("tokenizer_key") or raw.get("tokenizer", {}).get("key"); model = raw.get("model")
    if key is None: key = next((k for k, v in entries.items() if v.get("repository") == model), None)
    if key not in entries: raise ValueError(f"unrecognized tokenizer reference: {key or model}")
    item = entries[key]; supplied = raw.get("tokenizer_revision") or raw.get("tokenizer", {}).get("revision")
    if supplied and supplied != item["revision"]: raise ValueError("mutable or mismatched tokenizer revision")
    return CaseSpec(case_path, raw["regex"], list(raw["samples"]), key, item["repository"], item["revision"], str(raw.get("grammar", case_path.stem)), raw)
