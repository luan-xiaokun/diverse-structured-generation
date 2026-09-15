"""Evaluate all generated samples from the main regex generation matrix.

This script intentionally reads only saved JSON artifacts. It does not load
language models or tokenizers, so it is safe to run while generation jobs are
still in progress.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from diverse_guide.evaluation import string_kernel
from regex_dfa_guide import DiverseGuideDFA

try:
    from metrics_eval import compute_metrics
    from repro_results import build_metadata, write_json
except ModuleNotFoundError:
    from scripts.metrics_eval import compute_metrics
    from scripts.repro_results import build_metadata, write_json


DEFAULT_MODELS = [
    "Qwen/Qwen2.5-0.5B-Instruct",
    "Qwen/Qwen2.5-1.5B-Instruct",
    "microsoft/Phi-4-mini-instruct",
]
DEFAULT_GRAMMARS = [
    "email",
    "css-color",
    "json",
    "no-bomb",
    "semver",
    "api-route",
]
DEFAULT_METHODS = [
    "baseline_outlines",
    "diverse_guide",
    "diverse_guide_plus",
]
TREND_METRICS = [
    "state_coverage",
    "transition_coverage",
    "path_coverage",
    "vendi_score",
    "distinct_2gram_count",
    "distinct_3gram_count",
]


@dataclass(frozen=True)
class MatrixCell:
    method: str
    model: str
    grammar: str


def split_csv(value: str | None, default: list[str]) -> list[str]:
    if value is None or value == "":
        return list(default)
    return [item.strip() for item in value.split(",") if item.strip()]


def model_slug(model: str) -> str:
    return model.rsplit("/", maxsplit=1)[-1].lower()


def data_path(data_root: Path, cell: MatrixCell) -> Path:
    return data_root / cell.method / model_slug(cell.model) / f"{cell.grammar}.json"


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def sample_counts_for_limit(limit: int, step: int) -> list[int]:
    if limit <= 0:
        return []
    counts = list(range(step, limit + 1, step))
    if not counts or counts[-1] != limit:
        counts.append(limit)
    return counts


def compile_generation_regex(regex: str) -> re.Pattern[str]:
    return re.compile(f"^(?:{regex})$")


def validity_split(samples: list[str], regex: str) -> tuple[list[str], list[str]]:
    pattern = compile_generation_regex(regex)
    valid = []
    invalid = []
    for sample in samples:
        if pattern.match(sample):
            valid.append(sample)
        else:
            invalid.append(sample)
    return valid, invalid


def safe_compute_metrics(
    dfa: DiverseGuideDFA,
    samples: list[str],
    d: int,
    s: int,
) -> dict[str, Any]:
    if samples:
        return compute_metrics(dfa, samples, d=d, s=s)

    return {
        "sample_count": 0,
        "average_length": 0.0,
        "dfa": {
            "state_count": len(dfa.get_states()),
            "transition_count": sum(map(len, dfa.get_transitions().values())),
        },
        "metrics": {
            "state_coverage": 0.0,
            "transition_coverage": 0.0,
            "path_coverage": 0.0,
            "distinct_2gram": [0, 0],
            "distinct_3gram": [0, 0],
            "vendi_score": 0.0,
            "average_perplexity": None,
            "perplexity_count": 0,
            "perplexity_error_count": 0,
        },
    }


def avg_length(samples: list[str]) -> float:
    return float(np.mean([len(sample) for sample in samples])) if samples else 0.0


def build_efficiency(data: dict[str, Any], saved_sample_count: int) -> dict[str, Any]:
    accepted = data.get("accepted_samples")
    if accepted is None:
        accepted = saved_sample_count
    rejected = data.get("rejected_samples")
    if rejected is None:
        rejected = 0
    total_attempts = accepted + rejected
    elapsed = data.get("elapsed_seconds")
    return {
        "accepted_samples": accepted,
        "rejected_samples": rejected,
        "total_attempts": total_attempts,
        "attempted_batches": data.get("attempted_batches"),
        "elapsed_seconds": elapsed,
        "accepted_per_second": accepted / elapsed if elapsed and accepted else None,
        "seconds_per_accepted": elapsed / accepted if elapsed and accepted else None,
        "reject_ratio": rejected / total_attempts if total_attempts else 0.0,
    }


def extract_generation_parameters(data: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "max_tokens",
        "top_k",
        "top_p",
        "temperature",
        "seed",
        "gamma",
        "completion_bias_eta",
        "path_frequency_estimator",
        "baseline_backend",
        "device",
        "model_parameter_device",
    ]
    return {key: data.get(key) for key in keys}


def trend_point(
    dfa: DiverseGuideDFA,
    samples: list[str],
    regex: str,
    sample_count: int,
    d: int,
    s: int,
) -> dict[str, Any]:
    prefix = samples[:sample_count]
    valid, invalid = validity_split(prefix, regex)
    metrics = safe_compute_metrics(dfa, valid, d=d, s=s)
    metric_values = metrics["metrics"]
    return {
        "sample_count": sample_count,
        "valid_sample_count": len(valid),
        "invalid_sample_count": len(invalid),
        "validity_rate": len(valid) / len(prefix) if prefix else 0.0,
        "average_length": avg_length(prefix),
        "valid_average_length": metrics["average_length"],
        "state_coverage": metric_values["state_coverage"],
        "transition_coverage": metric_values["transition_coverage"],
        "path_coverage": metric_values["path_coverage"],
        "vendi_score": metric_values["vendi_score"],
        "distinct_2gram_count": metric_values["distinct_2gram"][0],
        "distinct_3gram_count": metric_values["distinct_3gram"][0],
    }


def build_trend_series(
    dfa: DiverseGuideDFA,
    samples: list[str],
    regex: str,
    step: int,
    max_samples: int | None,
    d: int,
    s: int,
) -> list[dict[str, Any]]:
    limit = len(samples)
    if max_samples is not None:
        limit = min(limit, max_samples)
    return [
        trend_point(dfa, samples, regex, count, d=d, s=s)
        for count in sample_counts_for_limit(limit, step)
    ]


def status_for_sample_count(
    path: Path,
    sample_count: int,
    expected_samples: int | None,
) -> str:
    if not path.exists():
        return "missing"
    if expected_samples is not None and sample_count < expected_samples:
        return "partial"
    return "complete"


def build_cell_result(
    cell: MatrixCell,
    path: Path,
    data: dict[str, Any],
    dfa: DiverseGuideDFA,
    d: int,
    s: int,
    step: int,
    max_samples: int | None,
    expected_samples: int | None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    samples = data.get("samples", [])
    if not isinstance(samples, list):
        raise ValueError(f"{path} field 'samples' must be a list")
    regex = data["regex"]
    valid, invalid = validity_split(samples, regex)
    eval_valid = valid[:max_samples] if max_samples is not None else valid
    metrics = safe_compute_metrics(dfa, eval_valid, d=d, s=s)
    efficiency = build_efficiency(data, saved_sample_count=len(samples))
    result = {
        "method": cell.method,
        "model": cell.model,
        "model_slug": model_slug(cell.model),
        "grammar": cell.grammar,
        "path": str(path),
        "status": status_for_sample_count(path, len(samples), expected_samples),
        "saved_sample_count": len(samples),
        "evaluated_valid_sample_count": len(eval_valid),
        "valid_sample_count": len(valid),
        "invalid_sample_count": len(invalid),
        "validity_rate": len(valid) / len(samples) if samples else 0.0,
        "invalid_examples": invalid[:5],
        "average_length": avg_length(samples),
        "valid_average_length": metrics["average_length"],
        "efficiency": efficiency,
        "generation_parameters": extract_generation_parameters(data),
        "dfa": metrics["dfa"],
        "metrics": metrics["metrics"],
    }
    trend = build_trend_series(
        dfa,
        samples,
        regex,
        step=step,
        max_samples=max_samples,
        d=d,
        s=s,
    )
    return result, trend


def flatten_summary(record: dict[str, Any]) -> dict[str, Any]:
    metrics = record.get("metrics", {})
    efficiency = record.get("efficiency", {})
    params = record.get("generation_parameters", {})
    return {
        "method": record["method"],
        "model": record["model"],
        "grammar": record["grammar"],
        "status": record["status"],
        "path": record["path"],
        "saved_sample_count": record.get("saved_sample_count"),
        "valid_sample_count": record.get("valid_sample_count"),
        "invalid_sample_count": record.get("invalid_sample_count"),
        "validity_rate": record.get("validity_rate"),
        "average_length": record.get("average_length"),
        "valid_average_length": record.get("valid_average_length"),
        "elapsed_seconds": efficiency.get("elapsed_seconds"),
        "accepted_per_second": efficiency.get("accepted_per_second"),
        "seconds_per_accepted": efficiency.get("seconds_per_accepted"),
        "reject_ratio": efficiency.get("reject_ratio"),
        "state_coverage": metrics.get("state_coverage"),
        "transition_coverage": metrics.get("transition_coverage"),
        "path_coverage": metrics.get("path_coverage"),
        "vendi_score": metrics.get("vendi_score"),
        "distinct_2gram_count": (
            metrics.get("distinct_2gram", [None])[0]
            if metrics.get("distinct_2gram") is not None
            else None
        ),
        "distinct_3gram_count": (
            metrics.get("distinct_3gram", [None])[0]
            if metrics.get("distinct_3gram") is not None
            else None
        ),
        "gamma": params.get("gamma"),
        "completion_bias_eta": params.get("completion_bias_eta"),
        "path_frequency_estimator": params.get("path_frequency_estimator"),
        "baseline_backend": params.get("baseline_backend"),
        "model_parameter_device": params.get("model_parameter_device"),
    }


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")


def evaluate_matrix(args: argparse.Namespace) -> dict[str, Any]:
    models = split_csv(args.models, DEFAULT_MODELS)
    grammars = split_csv(args.grammars, DEFAULT_GRAMMARS)
    methods = split_csv(args.methods, DEFAULT_METHODS)
    dfa_cache: dict[str, DiverseGuideDFA] = {}
    records = []
    trend_series: dict[str, list[dict[str, Any]]] = {}
    errors = []

    for model in models:
        for grammar in grammars:
            for method in methods:
                cell = MatrixCell(method=method, model=model, grammar=grammar)
                path = data_path(args.data_root, cell)
                key = f"{method}/{model_slug(model)}/{grammar}"
                if not path.exists():
                    record = {
                        "method": method,
                        "model": model,
                        "model_slug": model_slug(model),
                        "grammar": grammar,
                        "path": str(path),
                        "status": "missing",
                    }
                    records.append(record)
                    errors.append(f"missing: {path}")
                    continue
                try:
                    data = load_json(path)
                    regex = data["regex"]
                    if regex not in dfa_cache:
                        dfa_cache[regex] = DiverseGuideDFA(
                            f"(?:{regex})$",
                            2**32 - 1,
                            {},
                        )
                    record, trend = build_cell_result(
                        cell=cell,
                        path=path,
                        data=data,
                        dfa=dfa_cache[regex],
                        d=args.d,
                        s=args.s,
                        step=args.step,
                        max_samples=args.max_samples,
                        expected_samples=args.expected_samples,
                    )
                    records.append(record)
                    trend_series[key] = trend
                    if record["status"] != "complete":
                        errors.append(f"{record['status']}: {path}")
                except Exception as exc:
                    record = {
                        "method": method,
                        "model": model,
                        "model_slug": model_slug(model),
                        "grammar": grammar,
                        "path": str(path),
                        "status": "error",
                        "error": str(exc),
                    }
                    records.append(record)
                    errors.append(f"error: {path}: {exc}")

    metadata = build_metadata(
        {
            "string_kernel_backend": string_kernel.STRING_KERNEL_BACKEND,
            "model_loading": "none",
        }
    )
    return {
        "schema_version": 1,
        "experiment": "main_matrix_eval",
        "data_root": str(args.data_root),
        "parameters": {
            "models": models,
            "grammars": grammars,
            "methods": methods,
            "d": args.d,
            "s": args.s,
            "step": args.step,
            "max_samples": args.max_samples,
            "expected_samples": args.expected_samples,
        },
        "metadata": metadata,
        "records": records,
        "trend": {
            "metrics": TREND_METRICS,
            "series": trend_series,
        },
        "errors": errors,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate all saved artifacts from the main regex matrix."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data/extension/main_matrix_500"),
        help="Root produced by experiments/run_main_regex_matrix.sh.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/extension/main_matrix_500/evaluation"),
        help="Directory for metrics.json, metrics.jsonl, and trend_raw.json.",
    )
    parser.add_argument(
        "--models",
        type=str,
        default=None,
        help="Comma-separated list.",
    )
    parser.add_argument(
        "--grammars", type=str, default=None, help="Comma-separated list."
    )
    parser.add_argument(
        "--methods",
        type=str,
        default=None,
        help="Comma-separated list.",
    )
    parser.add_argument(
        "--expected-samples",
        type=int,
        default=500,
        help="Mark records with fewer saved samples as partial. Use 0 to disable.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional cap for diversity metrics and trend points.",
    )
    parser.add_argument("-d", type=int, default=5, help="Max k-mer degree.")
    parser.add_argument("-s", type=int, default=1, help="Max shift.")
    parser.add_argument(
        "--step",
        type=int,
        default=50,
        help="Prefix step size for trend raw data.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero if any matrix cell is missing, partial, or errored.",
    )
    args = parser.parse_args()
    if args.expected_samples is not None and args.expected_samples <= 0:
        args.expected_samples = None
    if args.max_samples is not None and args.max_samples <= 0:
        parser.error("--max-samples must be positive")
    if args.step <= 0:
        parser.error("--step must be positive")
    return args


def main() -> None:
    args = parse_args()
    result = evaluate_matrix(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = args.output_dir / "metrics.json"
    summary_path = args.output_dir / "metrics.jsonl"
    trend_path = args.output_dir / "trend_raw.json"
    write_json(metrics_path, result)
    write_jsonl(summary_path, [flatten_summary(record) for record in result["records"]])
    write_json(trend_path, result["trend"])

    complete_count = sum(
        1 for record in result["records"] if record["status"] == "complete"
    )
    print(f"[eval] records={len(result['records'])} complete={complete_count}")
    print(f"[eval] metrics={metrics_path}")
    print(f"[eval] summary={summary_path}")
    print(f"[eval] trend_raw={trend_path}")
    if result["errors"]:
        print(f"[eval] non-complete cells={len(result['errors'])}")
        for error in result["errors"][:20]:
            print(f"[eval] {error}")
    if args.strict and result["errors"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
