#!/usr/bin/env python3
"""Recompute case-study coverage curves from generated main-matrix samples."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


MODELS = (
    "qwen2.5-0.5b-instruct",
    "qwen2.5-1.5b-instruct",
    "phi-4-mini-instruct",
)
METHODS = ("baseline_outlines", "diverse_guide", "diverse_guide_plus")
BUDGETS = (1, 2, 5, 10, 25, 50, 100, 200, 500)


@dataclass(frozen=True)
class CaseStudy:
    name: str
    grammar: str
    test_file: str


CASE_STUDIES = (
    CaseStudy("email_validator", "email", "tests/test_email_validator.py"),
    CaseStudy("webcolors", "css-color", "tests/test_webcolors.py"),
    CaseStudy("python_semver", "semver", "tests/test_semver.py"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data/extension/main_matrix_500"),
        help="Generated sample root with method/model/grammar.json layout.",
    )
    parser.add_argument(
        "--case-study-dir",
        type=Path,
        default=Path("experiments/case_study"),
        help="Isolated pytest/coverage case-study project.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "results/extension/case_study_coverage_curves_main_matrix_500"
        ),
        help="Directory for coverage data, JSON reports, and CSV summaries.",
    )
    parser.add_argument(
        "--models",
        default=",".join(MODELS),
        help="Comma-separated model slugs.",
    )
    parser.add_argument(
        "--methods",
        default=",".join(METHODS),
        help="Comma-separated method names.",
    )
    parser.add_argument(
        "--budgets",
        default=",".join(str(value) for value in BUDGETS),
        help="Comma-separated sample budgets.",
    )
    parser.add_argument(
        "--tmp-data-filename",
        default="_tmp_case_study_samples.json",
        help="Temporary filename under experiments/case_study/data.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recompute existing coverage JSON reports.",
    )
    return parser.parse_args()


def split_csv(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def load_samples(path: Path) -> list[str]:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    samples = payload.get("samples")
    if not isinstance(samples, list):
        raise ValueError(f"{path} does not contain a samples list")
    return [str(sample) for sample in samples]


def ensure_case_study_python(case_study_dir: Path) -> Path:
    python_path = case_study_dir.resolve() / ".venv" / "bin" / "python"
    if python_path.exists():
        return python_path
    subprocess.run(
        ["uv", "run", "python", "-c", "import coverage, pytest"],
        cwd=case_study_dir,
        check=True,
    )
    if not python_path.exists():
        raise RuntimeError(f"Expected case-study venv at {python_path}")
    return python_path


def write_tmp_samples(path: Path, source_path: Path, samples: list[str]) -> None:
    payload: dict[str, Any] = {
        "source_path": str(source_path),
        "samples": samples,
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def run_coverage(
    *,
    python_path: Path,
    case_study_dir: Path,
    case_study: CaseStudy,
    tmp_data_filename: str,
    sample_count: int,
    coverage_file: Path,
    coverage_json: Path,
) -> dict[str, Any]:
    env = os.environ.copy()
    env["COVERAGE_FILE"] = str(coverage_file.resolve())
    env["TEST_DATA_FILENAME"] = tmp_data_filename
    env["TEST_DATA_SAMPLE_SIZE"] = str(sample_count)

    coverage_file.unlink(missing_ok=True)
    subprocess.run(
        [
            str(python_path),
            "-m",
            "coverage",
            "run",
            "--rcfile",
            ".coveragerc",
            "-m",
            "pytest",
            "-q",
            case_study.test_file,
        ],
        cwd=case_study_dir,
        env=env,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    subprocess.run(
        [
            str(python_path),
            "-m",
            "coverage",
            "json",
            "--rcfile",
            ".coveragerc",
            "-o",
            str(coverage_json.resolve()),
        ],
        cwd=case_study_dir,
        env=env,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    with coverage_json.open(encoding="utf-8") as handle:
        return json.load(handle)


def summarize_report(
    *,
    report: dict[str, Any],
    case_study: CaseStudy,
    model: str,
    method: str,
    sample_count: int,
    source_path: Path,
    coverage_json: Path,
) -> dict[str, Any]:
    totals = report["totals"]
    return {
        "case_study": case_study.name,
        "grammar": case_study.grammar,
        "model": model,
        "method": method,
        "sample_count": sample_count,
        "percent_covered": totals["percent_covered"],
        "covered_lines": totals["covered_lines"],
        "num_statements": totals["num_statements"],
        "covered_branches": totals["covered_branches"],
        "num_branches": totals["num_branches"],
        "source_path": str(source_path),
        "coverage_json": str(coverage_json),
    }


def write_rows(output_dir: Path, rows: list[dict[str, Any]]) -> None:
    csv_path = output_dir / "coverage_curves.csv"
    json_path = output_dir / "coverage_curves.json"
    jsonl_path = output_dir / "coverage_curves.jsonl"
    fields = [
        "case_study",
        "grammar",
        "model",
        "method",
        "sample_count",
        "percent_covered",
        "covered_lines",
        "num_statements",
        "covered_branches",
        "num_branches",
        "source_path",
        "coverage_json",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(rows, handle, indent=2)
        handle.write("\n")
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def main() -> None:
    args = parse_args()
    models = split_csv(args.models)
    methods = split_csv(args.methods)
    budgets = [int(value) for value in split_csv(args.budgets)]
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    python_path = ensure_case_study_python(args.case_study_dir)
    tmp_data_path = args.case_study_dir / "data" / args.tmp_data_filename
    rows: list[dict[str, Any]] = []
    total = len(CASE_STUDIES) * len(models) * len(methods) * len(budgets)
    step = 0

    try:
        for case_study in CASE_STUDIES:
            for model in models:
                for method in methods:
                    source_path = (
                        args.data_root
                        / method
                        / model
                        / f"{case_study.grammar}.json"
                    )
                    samples = load_samples(source_path)
                    if max(budgets) > len(samples):
                        raise ValueError(
                            f"{source_path} has {len(samples)} samples, "
                            f"but max budget is {max(budgets)}"
                        )
                    write_tmp_samples(tmp_data_path, source_path, samples)

                    for sample_count in budgets:
                        step += 1
                        stem = (
                            f"coverage.{case_study.name}__{model}__{method}"
                            f"__n{sample_count}"
                        )
                        coverage_file = output_dir / f".{stem}"
                        coverage_json = output_dir / f"{stem}.json"
                        print(
                            "[case-study]"
                            f" {step}/{total}"
                            f" case={case_study.name}"
                            f" model={model}"
                            f" method={method}"
                            f" n={sample_count}",
                            flush=True,
                        )
                        if coverage_json.exists() and not args.force:
                            with coverage_json.open(encoding="utf-8") as handle:
                                report = json.load(handle)
                        else:
                            report = run_coverage(
                                python_path=python_path,
                                case_study_dir=args.case_study_dir,
                                case_study=case_study,
                                tmp_data_filename=args.tmp_data_filename,
                                sample_count=sample_count,
                                coverage_file=coverage_file,
                                coverage_json=coverage_json,
                            )
                        rows.append(
                            summarize_report(
                                report=report,
                                case_study=case_study,
                                model=model,
                                method=method,
                                sample_count=sample_count,
                                source_path=source_path,
                                coverage_json=coverage_json,
                            )
                        )
                        write_rows(output_dir, rows)
    finally:
        tmp_data_path.unlink(missing_ok=True)

    write_rows(output_dir, rows)
    print(f"[case-study] wrote {len(rows)} rows to {output_dir}")


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as exc:
        print(f"command failed: {exc}", file=sys.stderr)
        if exc.stdout:
            print(exc.stdout, file=sys.stderr)
        raise
