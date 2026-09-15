"""Prepare plot-ready n=100 frontier records for the phase-1 refinement screen.

The source runs are split across two native n=100 directories:

* gamma 0.1/0.2/0.3 for five grammars, excluding css-color
* gamma 0.4/0.5/0.6 for all six grammars

Low-gamma css-color was not produced in the native n=100 screen, but the same
cells exist in the n=500 screen. For plotting a complete 6 grammar x 6 gamma x
2 estimator frontier, this script evaluates the first 100 saved samples from
those n=500 cells and marks them with source_kind=n500_first100.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from evaluate_main_matrix import (
    avg_length,
    build_efficiency,
    model_slug,
    safe_compute_metrics,
    validity_split,
)
from regex_dfa_guide import DiverseGuideDFA


MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
GRAMMARS = ["email", "css-color", "json", "no-bomb", "semver", "api-route"]
ESTIMATORS = ["min", "harmonic"]
GAMMAS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]

DEFAULT_LOW_N100_ROOT = Path(
    "data/extension/refinement_phase1_harmonic_min_screen_n100"
)
DEFAULT_HIGH_N100_ROOT = Path(
    "data/extension/refinement_phase1_harmonic_min_screen_n100_gamma_high"
)
DEFAULT_N500_ROOT = Path("data/extension/refinement_phase1_harmonic_min_500")
DEFAULT_OUTPUT_DIR = Path("results/extension/refinement_phase1_frontier_n100")


def gamma_slug(gamma: float) -> str:
    return f"{gamma:.1f}".replace(".", "p")


def method_name(estimator: str, gamma: float) -> str:
    return f"{estimator}_g{gamma_slug(gamma)}"


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")


def write_csv(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "grammar",
        "estimator",
        "gamma",
        "method",
        "model",
        "model_slug",
        "status",
        "source_kind",
        "source_root",
        "path",
        "timing_scope",
        "metric_sample_scope",
        "sample_count_requested",
        "saved_sample_count",
        "evaluated_sample_count",
        "valid_sample_count",
        "invalid_sample_count",
        "validity_rate",
        "average_length",
        "valid_average_length",
        "state_coverage",
        "transition_coverage",
        "path_coverage",
        "state_coverage_pct",
        "transition_coverage_pct",
        "path_coverage_pct",
        "distinct_2gram_count",
        "distinct_2gram_total",
        "distinct_3gram_count",
        "distinct_3gram_total",
        "vendi_score",
        "accepted_samples",
        "rejected_samples",
        "total_attempts",
        "attempted_batches",
        "reject_ratio",
        "elapsed_seconds",
        "accepted_per_second",
        "seconds_per_accepted",
        "source_accepted_samples",
        "source_rejected_samples",
        "source_total_attempts",
        "source_elapsed_seconds",
        "source_reject_ratio",
        "max_tokens",
        "top_k",
        "top_p",
        "temperature",
        "seed",
        "completion_bias_eta",
        "path_frequency_estimator",
        "baseline_backend",
        "device",
        "model_parameter_device",
        "dfa_state_count",
        "dfa_transition_count",
        "error",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)


def candidate_paths(
    estimator: str,
    gamma: float,
    grammar: str,
    args: argparse.Namespace,
) -> list[tuple[Path, str, str, int]]:
    method = method_name(estimator, gamma)
    slug = model_slug(MODEL)
    if gamma <= 0.3:
        native_path = args.low_n100_root / method / slug / f"{grammar}.json"
        n500_path = args.n500_root / method / slug / f"{grammar}.json"
        candidates = [(native_path, "native_n100", "native_n100_run", 100)]
        if grammar == "css-color":
            candidates.append((n500_path, "n500_first100", "source_n500_run", 100))
        return candidates

    native_path = args.high_n100_root / method / slug / f"{grammar}.json"
    return [(native_path, "native_n100", "native_n100_run", 100)]


def empty_record(
    estimator: str,
    gamma: float,
    grammar: str,
    candidates: list[tuple[Path, str, str, int]],
) -> dict[str, Any]:
    return {
        "grammar": grammar,
        "estimator": estimator,
        "gamma": gamma,
        "method": method_name(estimator, gamma),
        "model": MODEL,
        "model_slug": model_slug(MODEL),
        "status": "missing",
        "source_kind": None,
        "source_root": None,
        "path": str(candidates[0][0]) if candidates else None,
        "timing_scope": None,
        "metric_sample_scope": "first_100_saved_samples",
        "sample_count_requested": 100,
        "error": "no candidate source exists",
    }


def source_root_for(path: Path, args: argparse.Namespace) -> str | None:
    for root in [args.low_n100_root, args.high_n100_root, args.n500_root]:
        try:
            path.relative_to(root)
            return str(root)
        except ValueError:
            continue
    return None


def build_record(
    path: Path,
    source_kind: str,
    timing_scope: str,
    sample_limit: int,
    estimator: str,
    gamma: float,
    grammar: str,
    dfa_cache: dict[str, DiverseGuideDFA],
    args: argparse.Namespace,
) -> dict[str, Any]:
    method = method_name(estimator, gamma)
    data = load_json(path)
    samples = data.get("samples", [])
    if not isinstance(samples, list):
        raise ValueError(f"{path} field 'samples' must be a list")

    eval_samples = samples[:sample_limit]
    regex = data["regex"]
    if regex not in dfa_cache:
        dfa_cache[regex] = DiverseGuideDFA(f"(?:{regex})$", 2**32 - 1, {})
    dfa = dfa_cache[regex]

    valid, invalid = validity_split(eval_samples, regex)
    metrics = safe_compute_metrics(dfa, valid, d=args.d, s=args.s)
    metric_values = metrics["metrics"]
    efficiency = build_efficiency(data, saved_sample_count=len(samples))

    if source_kind == "native_n100":
        accepted_samples = efficiency["accepted_samples"]
        rejected_samples = efficiency["rejected_samples"]
        total_attempts = efficiency["total_attempts"]
        elapsed_seconds = efficiency["elapsed_seconds"]
        accepted_per_second = efficiency["accepted_per_second"]
        seconds_per_accepted = efficiency["seconds_per_accepted"]
        reject_ratio = efficiency["reject_ratio"]
    else:
        accepted_samples = len(eval_samples)
        rejected_samples = None
        total_attempts = None
        elapsed_seconds = None
        accepted_per_second = None
        seconds_per_accepted = None
        reject_ratio = None

    source_total_attempts = efficiency["total_attempts"]
    record = {
        "grammar": grammar,
        "estimator": estimator,
        "gamma": gamma,
        "method": method,
        "model": data.get("model", MODEL),
        "model_slug": model_slug(data.get("model", MODEL)),
        "status": "complete" if len(eval_samples) >= sample_limit else "partial",
        "source_kind": source_kind,
        "source_root": source_root_for(path, args),
        "path": str(path),
        "timing_scope": timing_scope,
        "metric_sample_scope": "first_100_saved_samples",
        "sample_count_requested": sample_limit,
        "saved_sample_count": len(samples),
        "evaluated_sample_count": len(eval_samples),
        "valid_sample_count": len(valid),
        "invalid_sample_count": len(invalid),
        "validity_rate": len(valid) / len(eval_samples) if eval_samples else 0.0,
        "average_length": avg_length(eval_samples),
        "valid_average_length": metrics["average_length"],
        "state_coverage": metric_values["state_coverage"],
        "transition_coverage": metric_values["transition_coverage"],
        "path_coverage": metric_values["path_coverage"],
        "state_coverage_pct": 100.0 * metric_values["state_coverage"],
        "transition_coverage_pct": 100.0 * metric_values["transition_coverage"],
        "path_coverage_pct": 100.0 * metric_values["path_coverage"],
        "distinct_2gram_count": metric_values["distinct_2gram"][0],
        "distinct_2gram_total": metric_values["distinct_2gram"][1],
        "distinct_3gram_count": metric_values["distinct_3gram"][0],
        "distinct_3gram_total": metric_values["distinct_3gram"][1],
        "vendi_score": metric_values["vendi_score"],
        "accepted_samples": accepted_samples,
        "rejected_samples": rejected_samples,
        "total_attempts": total_attempts,
        "attempted_batches": efficiency["attempted_batches"],
        "reject_ratio": reject_ratio,
        "elapsed_seconds": elapsed_seconds,
        "accepted_per_second": accepted_per_second,
        "seconds_per_accepted": seconds_per_accepted,
        "source_accepted_samples": efficiency["accepted_samples"],
        "source_rejected_samples": efficiency["rejected_samples"],
        "source_total_attempts": source_total_attempts,
        "source_elapsed_seconds": efficiency["elapsed_seconds"],
        "source_reject_ratio": efficiency["reject_ratio"],
        "max_tokens": data.get("max_tokens"),
        "top_k": data.get("top_k"),
        "top_p": data.get("top_p"),
        "temperature": data.get("temperature"),
        "seed": data.get("seed"),
        "completion_bias_eta": data.get("completion_bias_eta"),
        "path_frequency_estimator": data.get("path_frequency_estimator"),
        "baseline_backend": data.get("baseline_backend"),
        "device": data.get("device"),
        "model_parameter_device": data.get("model_parameter_device"),
        "dfa_state_count": metrics["dfa"]["state_count"],
        "dfa_transition_count": metrics["dfa"]["transition_count"],
        "error": None,
    }
    return record


def prepare_records(args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[str]]:
    dfa_cache: dict[str, DiverseGuideDFA] = {}
    records: list[dict[str, Any]] = []
    errors: list[str] = []

    for grammar in GRAMMARS:
        for estimator in ESTIMATORS:
            for gamma in GAMMAS:
                candidates = candidate_paths(estimator, gamma, grammar, args)
                chosen = next(
                    (
                        (path, source_kind, timing_scope, sample_limit)
                        for path, source_kind, timing_scope, sample_limit in candidates
                        if path.exists()
                    ),
                    None,
                )
                if chosen is None:
                    record = empty_record(estimator, gamma, grammar, candidates)
                    records.append(record)
                    errors.append(f"missing: {grammar} {estimator} gamma={gamma}")
                    continue

                path, source_kind, timing_scope, sample_limit = chosen
                try:
                    record = build_record(
                        path=path,
                        source_kind=source_kind,
                        timing_scope=timing_scope,
                        sample_limit=sample_limit,
                        estimator=estimator,
                        gamma=gamma,
                        grammar=grammar,
                        dfa_cache=dfa_cache,
                        args=args,
                    )
                    records.append(record)
                    if record["status"] != "complete":
                        errors.append(
                            f"{record['status']}: {grammar} {estimator} "
                            f"gamma={gamma} path={path}"
                        )
                except Exception as exc:
                    record = empty_record(estimator, gamma, grammar, candidates)
                    record.update(
                        {
                            "status": "error",
                            "source_kind": source_kind,
                            "source_root": source_root_for(path, args),
                            "path": str(path),
                            "timing_scope": timing_scope,
                            "error": str(exc),
                        }
                    )
                    records.append(record)
                    errors.append(f"error: {path}: {exc}")

    return records, errors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Consolidate phase-1 harmonic/min n=100 screen records."
    )
    parser.add_argument("--low-n100-root", type=Path, default=DEFAULT_LOW_N100_ROOT)
    parser.add_argument("--high-n100-root", type=Path, default=DEFAULT_HIGH_N100_ROOT)
    parser.add_argument("--n500-root", type=Path, default=DEFAULT_N500_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("-d", type=int, default=5, help="Max k-mer degree.")
    parser.add_argument("-s", type=int, default=1, help="Max shift.")
    parser.add_argument("--strict", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records, errors = prepare_records(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = args.output_dir / "frontier_records.csv"
    jsonl_path = args.output_dir / "frontier_records.jsonl"
    json_path = args.output_dir / "frontier_records.json"

    write_csv(csv_path, records)
    write_jsonl(jsonl_path, records)
    complete_count = sum(1 for record in records if record["status"] == "complete")
    source_counts: dict[str, int] = {}
    for record in records:
        source_kind = record.get("source_kind") or "none"
        source_counts[source_kind] = source_counts.get(source_kind, 0) + 1

    write_json(
        json_path,
        {
            "schema_version": 1,
            "experiment": "refinement_phase1_frontier_n100",
            "parameters": {
                "model": MODEL,
                "grammars": GRAMMARS,
                "estimators": ESTIMATORS,
                "gammas": GAMMAS,
                "sample_count_requested": 100,
                "d": args.d,
                "s": args.s,
                "low_n100_root": str(args.low_n100_root),
                "high_n100_root": str(args.high_n100_root),
                "n500_root": str(args.n500_root),
            },
            "summary": {
                "record_count": len(records),
                "complete_count": complete_count,
                "source_counts": source_counts,
                "error_count": len(errors),
            },
            "errors": errors,
            "records": records,
        },
    )

    print(
        "[frontier] "
        f"records={len(records)} complete={complete_count} errors={len(errors)}"
    )
    print(f"[frontier] source_counts={source_counts}")
    print(f"[frontier] csv={csv_path}")
    print(f"[frontier] jsonl={jsonl_path}")
    print(f"[frontier] json={json_path}")
    if errors:
        for error in errors[:20]:
            print(f"[frontier] {error}")
    if args.strict and errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
