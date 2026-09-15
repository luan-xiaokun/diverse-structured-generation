"""Collect engineering benchmark JSON files into table-friendly summaries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def nested_get(data: dict[str, Any], path: tuple[str, ...]) -> Any:
    value: Any = data
    for key in path:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def flatten_result(path: Path, result: dict[str, Any]) -> dict[str, Any]:
    method_dfa = None
    for run in result.get("runs", []):
        if run.get("method_dfa") is not None:
            method_dfa = run["method_dfa"]
            break

    return {
        "path": str(path),
        "repo_path": result.get("repo_path"),
        "grammar_source_repo": result.get("grammar_source_repo"),
        "grammar": result.get("grammar"),
        "model": result.get("model"),
        "method": result.get("method"),
        "engine": nested_get(result, ("method_config", "engine")),
        "gamma": nested_get(result, ("method_config", "gamma")),
        "completion_bias_eta": nested_get(
            result, ("method_config", "completion_bias_eta")
        ),
        "path_frequency_estimator": nested_get(
            result, ("method_config", "path_frequency_estimator")
        ),
        "runs": nested_get(result, ("parameters", "runs")),
        "max_tokens": nested_get(result, ("parameters", "max_tokens")),
        "vocab_size": nested_get(result, ("tokenizer", "vocab_size")),
        "score_width": nested_get(result, ("tokenizer", "score_width")),
        "grammar_state_count": nested_get(result, ("grammar_dfa", "state_count")),
        "grammar_byte_transition_count": nested_get(
            result, ("grammar_dfa", "byte_transition_count")
        ),
        "method_state_count": (
            method_dfa.get("state_count") if method_dfa is not None else None
        ),
        "method_byte_transition_count": (
            method_dfa.get("byte_transition_count") if method_dfa is not None else None
        ),
        "method_token_transition_count": (
            method_dfa.get("token_transition_count") if method_dfa is not None else None
        ),
        "model_load_seconds": nested_get(
            result, ("model_loading", "model_load_seconds")
        ),
        "tokenizer_load_seconds": nested_get(
            result, ("model_loading", "tokenizer_load_seconds")
        ),
        "model_parameter_device": nested_get(
            result, ("model_loading", "model_parameter_device")
        ),
        "build_seconds_mean": nested_get(result, ("summary", "build_seconds", "mean")),
        "build_seconds_std": nested_get(result, ("summary", "build_seconds", "std")),
        "generation_seconds_mean": nested_get(
            result, ("summary", "generation_seconds", "mean")
        ),
        "tokens_per_second_mean": nested_get(
            result, ("summary", "tokens_per_second", "mean")
        ),
        "tokens_per_second_std": nested_get(
            result, ("summary", "tokens_per_second", "std")
        ),
        "ttft_ms_mean": nested_get(result, ("summary", "ttft_ms", "mean")),
        "ttft_ms_std": nested_get(result, ("summary", "ttft_ms", "std")),
        "avg_inter_token_latency_ms_mean": nested_get(
            result, ("summary", "avg_inter_token_latency_ms", "mean")
        ),
        "avg_inter_token_latency_ms_std": nested_get(
            result, ("summary", "avg_inter_token_latency_ms", "std")
        ),
    }


def collect(input_root: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(input_root.rglob("*.json")):
        if path.name in {"summary.json", "summary.jsonl"}:
            continue
        result = load_json(path)
        if result.get("experiment") != "generation_engineering_benchmark":
            continue
        rows.append(flatten_result(path, result))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect engineering benchmark JSON files."
    )
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> list[dict[str, Any]]:
    args = parse_args(argv)
    rows = collect(args.input_root)
    payload = {
        "schema_version": 1,
        "experiment": "generation_engineering_benchmark_summary",
        "input_root": str(args.input_root),
        "record_count": len(rows),
        "records": rows,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2, sort_keys=True))
    write_jsonl(args.output_jsonl, rows)
    print(f"[collect] records={len(rows)}")
    print(f"[collect] json={args.output_json}")
    print(f"[collect] jsonl={args.output_jsonl}")
    return rows


if __name__ == "__main__":
    main()
