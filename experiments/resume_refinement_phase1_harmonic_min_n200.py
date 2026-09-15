#!/usr/bin/env python3
"""Resume the harmonic-vs-min gamma screen from n=100 to n=200.

The n=100 frontier is consolidated in
``results/extension/refinement_phase1_frontier_n100/frontier_records.csv``.
This runner replays the first 100 accepted samples for each cell, appends
another 100 accepted samples with the same generation settings, and writes the
new artifacts under ``data/extension/refinement_phase1_harmonic_min_screen_n200``.

Low-gamma css-color cells in the n=100 frontier come from first-100 prefixes of
an earlier n=500 run. For those cells this script builds an explicit temporary
first-100 artifact before invoking ``scripts/resume_generate_re.py`` so all cells
share the same resume-from-100 semantics.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
MODEL_SLUG = "qwen2.5-1.5b-instruct"
DEFAULT_GRAMMARS = ["email", "css-color", "no-bomb", "semver", "api-route", "json"]
DEFAULT_ESTIMATORS = ["min", "harmonic"]
DEFAULT_GAMMAS = ["0.1", "0.2", "0.3", "0.4", "0.5", "0.6"]


def split_csv(value: str | None, default: list[str]) -> list[str]:
    if not value:
        return list(default)
    return [item.strip() for item in value.split(",") if item.strip()]


def gamma_label(gamma: str) -> str:
    return gamma.replace(".", "p").replace("-", "m")


def method_name(estimator: str, gamma: str) -> str:
    return f"{estimator}_g{gamma_label(gamma)}"


def max_tokens_for(grammar: str, default_max_tokens: int, json_max_tokens: int) -> int:
    return json_max_tokens if grammar == "json" else default_max_tokens


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True)
        handle.write("\n")
    tmp.replace(path)


def sample_count(path: Path) -> int:
    if not path.exists():
        return 0
    try:
        data = load_json(path)
    except Exception:
        return 0
    samples = data.get("samples", [])
    return len(samples) if isinstance(samples, list) else 0


def read_frontier(path: Path) -> dict[tuple[str, str, str], dict[str, str]]:
    rows: dict[tuple[str, str, str], dict[str, str]] = {}
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            key = (row["grammar"], row["estimator"], row["gamma"])
            rows[key] = row
    return rows


def make_base_artifact(source_path: Path, base_path: Path, start_n: int) -> Path:
    source = load_json(source_path)
    samples = source.get("samples", [])
    if not isinstance(samples, list):
        raise ValueError(f"{source_path} field 'samples' must be a list")
    if len(samples) < start_n:
        raise ValueError(
            f"{source_path} has only {len(samples)} samples; need at least {start_n}"
        )
    base = dict(source)
    base["samples"] = samples[:start_n]
    base["accepted_samples"] = start_n
    # For n500-prefix sources, first-100 rejection/timing breakdown is not
    # recoverable from the aggregate source artifact. Keep the continuation
    # counters honest by starting rejected/elapsed from zero and preserve the
    # provenance in resume_history.
    base["rejected_samples"] = 0
    base["attempted_batches"] = start_n
    base["elapsed_seconds"] = 0.0
    history = list(base.get("resume_history", []))
    history.append(
        {
            "mode": "truncated_base_for_n200_resume",
            "source_path": str(source_path),
            "source_sample_count": len(samples),
            "base_sample_count": start_n,
            "note": (
                "Only accepted samples are recoverable for the prefix; "
                "rejection and elapsed counters are measured from the resumed "
                "continuation."
            ),
        }
    )
    base["resume_history"] = history
    write_json(base_path, base)
    return base_path


def ensure_cuda(python_bin: str) -> None:
    code = (
        "import torch\n"
        "print(f'[preflight] cuda_available={torch.cuda.is_available()}')\n"
        "print(f'[preflight] cuda_device_count={torch.cuda.device_count()}')\n"
        "raise SystemExit(0 if torch.cuda.is_available() and torch.cuda.device_count() >= 1 else 2)\n"
    )
    subprocess.run([python_bin, "-c", code], check=True)


def run_logged(cmd: list[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        log.write("[cmd] " + " ".join(cmd) + "\n")
        log.flush()
        proc = subprocess.run(
            cmd,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    if proc.returncode != 0:
        tail = ""
        try:
            lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
            tail = "\n".join(lines[-80:])
        except Exception:
            pass
        raise RuntimeError(
            f"command failed with exit code {proc.returncode}: {' '.join(cmd)}\n"
            f"log: {log_path}\n"
            f"tail:\n{tail}"
        )


def build_summary_record(
    *,
    grammar: str,
    estimator: str,
    gamma: str,
    source_path: Path,
    input_path: Path,
    output_path: Path,
    log_path: Path,
    source_kind: str,
    status: str,
    seed: int,
    configured_max_tokens: int,
    target_n: int,
    start_n: int,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "experiment": "refinement_phase1_harmonic_min_screen_n200",
        "model": MODEL,
        "method": method_name(estimator, gamma),
        "grammar": grammar,
        "path_frequency_estimator_configured": estimator,
        "gamma_configured": float(gamma),
        "completion_bias_eta_configured": 0.0,
        "seed": seed,
        "configured_max_tokens": configured_max_tokens,
        "target_n": target_n,
        "start_n": start_n,
        "source_path": str(source_path),
        "input_path": str(input_path),
        "path": str(output_path),
        "log_path": str(log_path),
        "source_kind": source_kind,
        "resume_strategy": "counter_replay_reseeded",
        "status": status,
    }
    try:
        record["git_commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip()
    except Exception:
        record["git_commit"] = None

    if output_path.exists():
        data = load_json(output_path)
        accepted = data.get("accepted_samples") or len(data.get("samples", []))
        rejected = data.get("rejected_samples") or 0
        attempts = accepted + rejected
        elapsed = data.get("elapsed_seconds")
        record.update(
            {
                "accepted_samples": accepted,
                "rejected_samples": rejected,
                "total_attempts": attempts,
                "attempted_batches": data.get("attempted_batches"),
                "elapsed_seconds": elapsed,
                "accepted_per_second": accepted / elapsed if elapsed and accepted else None,
                "seconds_per_accepted": elapsed / accepted if elapsed and accepted else None,
                "reject_ratio": rejected / attempts if attempts else 0.0,
                "max_tokens": data.get("max_tokens"),
                "top_k": data.get("top_k"),
                "top_p": data.get("top_p"),
                "temperature": data.get("temperature"),
                "gamma": data.get("gamma"),
                "completion_bias_eta": data.get("completion_bias_eta"),
                "path_frequency_estimator": data.get("path_frequency_estimator"),
                "baseline_backend": data.get("baseline_backend"),
                "device": data.get("device"),
                "model_parameter_device": data.get("model_parameter_device"),
            }
        )
    return record


def append_summary(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Resume the phase-1 harmonic/min gamma screen to n=200."
    )
    parser.add_argument(
        "--frontier-csv",
        type=Path,
        default=Path("results/extension/refinement_phase1_frontier_n100/frontier_records.csv"),
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data/extension/refinement_phase1_harmonic_min_screen_n200"),
    )
    parser.add_argument(
        "--base-root",
        type=Path,
        default=Path("data/extension/refinement_phase1_harmonic_min_screen_n200_base100"),
    )
    parser.add_argument(
        "--log-root",
        type=Path,
        default=Path("logs/extension/refinement_phase1_harmonic_min_screen_n200"),
    )
    parser.add_argument(
        "--summary-file",
        type=Path,
        default=Path("results/extension/refinement_phase1_harmonic_min_screen_n200/summary.jsonl"),
    )
    parser.add_argument("--python-bin", default=".venv/bin/python")
    parser.add_argument("--target-n", type=int, default=200)
    parser.add_argument("--start-n", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--default-max-tokens", type=int, default=24)
    parser.add_argument("--json-max-tokens", type=int, default=72)
    parser.add_argument("--grammars", default=None)
    parser.add_argument("--estimators", default=None)
    parser.add_argument("--gammas", default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--local-files-only", action="store_true", default=True)
    parser.add_argument("--require-cuda", action="store_true", default=True)
    parser.add_argument("--expected-model-parameter-device", default="cuda:0")
    args = parser.parse_args()
    if args.target_n <= args.start_n:
        parser.error("--target-n must be greater than --start-n")
    return args


def main() -> None:
    args = parse_args()
    grammars = split_csv(args.grammars, DEFAULT_GRAMMARS)
    estimators = split_csv(args.estimators, DEFAULT_ESTIMATORS)
    gammas = split_csv(args.gammas, DEFAULT_GAMMAS)

    if args.require_cuda:
        ensure_cuda(args.python_bin)

    rows = read_frontier(args.frontier_csv)
    args.summary_file.parent.mkdir(parents=True, exist_ok=True)
    args.summary_file.write_text("", encoding="utf-8")

    total = len(grammars) * len(estimators) * len(gammas)
    index = 0
    for grammar in grammars:
        for estimator in estimators:
            for gamma in gammas:
                index += 1
                method = method_name(estimator, gamma)
                key = (grammar, estimator, gamma)
                if key not in rows:
                    raise KeyError(f"missing frontier row for {key}")
                row = rows[key]
                source_path = Path(row["path"])
                if not source_path.exists():
                    raise FileNotFoundError(source_path)

                output_path = (
                    args.data_root / method / MODEL_SLUG / f"{grammar}.json"
                )
                log_path = args.log_root / method / MODEL_SLUG / f"{grammar}.log"
                configured_max_tokens = max_tokens_for(
                    grammar, args.default_max_tokens, args.json_max_tokens
                )

                existing = sample_count(output_path)
                if existing >= args.target_n and not args.force:
                    print(
                        f"[skip] {index}/{total} method={method} grammar={grammar} "
                        f"samples={existing}"
                    )
                    record = build_summary_record(
                        grammar=grammar,
                        estimator=estimator,
                        gamma=gamma,
                        source_path=source_path,
                        input_path=output_path,
                        output_path=output_path,
                        log_path=log_path,
                        source_kind=row.get("source_kind", ""),
                        status="skipped",
                        seed=args.seed,
                        configured_max_tokens=configured_max_tokens,
                        target_n=args.target_n,
                        start_n=args.start_n,
                    )
                    append_summary(args.summary_file, record)
                    continue

                if existing > 0 and not args.force:
                    input_path = output_path
                    input_kind = "partial_output"
                else:
                    source_count = sample_count(source_path)
                    if source_count == args.start_n:
                        input_path = source_path
                    elif source_count > args.start_n:
                        input_path = (
                            args.base_root / method / MODEL_SLUG / f"{grammar}.json"
                        )
                        make_base_artifact(source_path, input_path, args.start_n)
                    else:
                        raise ValueError(
                            f"{source_path} has {source_count} samples; "
                            f"need {args.start_n}."
                        )
                    input_kind = row.get("source_kind", "")

                print(
                    f"[run] {index}/{total} method={method} estimator={estimator} "
                    f"gamma={gamma} grammar={grammar} input_kind={input_kind} "
                    f"target={args.target_n} output={output_path}"
                )
                cmd = [
                    args.python_bin,
                    "scripts/resume_generate_re.py",
                    grammar,
                    "--model",
                    MODEL,
                    "-n",
                    str(args.target_n),
                    "--seed",
                    str(args.seed),
                    "--max-tokens",
                    str(configured_max_tokens),
                    "--input",
                    str(input_path),
                    "--output",
                    str(output_path),
                    "--gamma",
                    gamma,
                    "--completion-bias-eta",
                    "0.0",
                    "--path-frequency-estimator",
                    estimator,
                    "--batch-size",
                    "1",
                    "--no-progress",
                ]
                if args.local_files_only:
                    cmd.append("--local-files-only")

                run_logged(cmd, log_path)
                out = load_json(output_path)
                actual_device = out.get("model_parameter_device")
                if (
                    args.require_cuda
                    and actual_device != args.expected_model_parameter_device
                ):
                    raise RuntimeError(
                        f"{output_path} model_parameter_device={actual_device!r}, "
                        f"expected {args.expected_model_parameter_device!r}"
                    )
                accepted = out.get("accepted_samples") or len(out.get("samples", []))
                if accepted < args.target_n:
                    raise RuntimeError(
                        f"{output_path} accepted_samples={accepted}, "
                        f"target={args.target_n}"
                    )
                record = build_summary_record(
                    grammar=grammar,
                    estimator=estimator,
                    gamma=gamma,
                    source_path=source_path,
                    input_path=input_path,
                    output_path=output_path,
                    log_path=log_path,
                    source_kind=input_kind,
                    status="done",
                    seed=args.seed,
                    configured_max_tokens=configured_max_tokens,
                    target_n=args.target_n,
                    start_n=args.start_n,
                )
                append_summary(args.summary_file, record)
                print(
                    f"[done] {index}/{total} method={method} grammar={grammar} "
                    f"accepted={record.get('accepted_samples')} "
                    f"rejected={record.get('rejected_samples')} "
                    f"elapsed={record.get('elapsed_seconds')}"
                )

    print(f"[complete] summary={args.summary_file}")


if __name__ == "__main__":
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    main()
