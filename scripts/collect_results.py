"""Collect structured reproduction JSON results into CSV tables."""

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

METRIC_FIELDS = [
    "experiment",
    "setting",
    "grammar",
    "model",
    "sample_count",
    "average_length",
    "state_count",
    "transition_count",
    "state_coverage",
    "transition_coverage",
    "path_coverage",
    "distinct_2gram_count",
    "distinct_2gram_samples",
    "distinct_3gram_count",
    "distinct_3gram_samples",
    "vendi_score",
    "average_perplexity",
    "perplexity_count",
    "perplexity_error_count",
    "temperature",
    "ablation_component",
]

RUNTIME_FIELDS = [
    "experiment",
    "setting",
    "grammar",
    "model",
    "target_tokens",
    "generated_tokens",
    "seconds",
    "tokens_per_second",
    "max_tokens",
    "temperature",
]

METRIC_EXPERIMENTS = [
    "diversity",
    "temperature_ablation",
    "component_ablation",
]

REQUIRED_INPUTS = [
    *METRIC_EXPERIMENTS,
    "runtime",
]
COLLECTOR_EXPERIMENTS = [
    "all",
    *REQUIRED_INPUTS,
]


class MissingResultInputsError(Exception):
    """Raised when required result inputs are missing or empty."""

    def __init__(self, missing_inputs: list[tuple[Path, str]]) -> None:
        lines = ["Missing result inputs:"]
        lines.extend(f"- {path} ({reason})" for path, reason in missing_inputs)
        super().__init__("\n".join(lines))
        self.missing_inputs = missing_inputs


def read_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as f:
        return json.load(f)


def metric_row(data: dict[str, Any]) -> dict[str, Any]:
    dfa = data["dfa"]
    metrics = data["metrics"]
    parameters = data.get("parameters", {})
    distinct_2gram = metrics["distinct_2gram"]
    distinct_3gram = metrics["distinct_3gram"]

    return {
        "experiment": data["experiment"],
        "setting": data["setting"],
        "grammar": data["grammar"],
        "model": data["model"],
        "sample_count": data["sample_count"],
        "average_length": data["average_length"],
        "state_count": dfa["state_count"],
        "transition_count": dfa["transition_count"],
        "state_coverage": metrics["state_coverage"],
        "transition_coverage": metrics["transition_coverage"],
        "path_coverage": metrics["path_coverage"],
        "distinct_2gram_count": distinct_2gram[0],
        "distinct_2gram_samples": distinct_2gram[1],
        "distinct_3gram_count": distinct_3gram[0],
        "distinct_3gram_samples": distinct_3gram[1],
        "vendi_score": metrics["vendi_score"],
        "average_perplexity": metrics["average_perplexity"],
        "perplexity_count": metrics["perplexity_count"],
        "perplexity_error_count": metrics["perplexity_error_count"],
        "temperature": parameters.get("temperature"),
        "ablation_component": parameters.get("ablation_component"),
    }


def runtime_row(data: dict[str, Any]) -> dict[str, Any]:
    parameters = data.get("parameters", {})
    tokens = data["tokens"]
    timing = data["timing"]

    return {
        "experiment": data["experiment"],
        "setting": data["setting"],
        "grammar": data["grammar"],
        "model": data["model"],
        "target_tokens": tokens["target"],
        "generated_tokens": tokens["generated"],
        "seconds": timing["seconds"],
        "tokens_per_second": timing["tokens_per_second"],
        "max_tokens": parameters["max_tokens"],
        "temperature": parameters.get("temperature"),
    }


def write_csv(
    path: str | Path, rows: list[dict[str, Any]], fields: list[str]
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _json_paths(experiment_dir: Path) -> list[Path]:
    if not experiment_dir.exists():
        raise FileNotFoundError(f"Missing results directory: {experiment_dir}")
    paths = sorted(experiment_dir.rglob("*.json"))
    if not paths:
        raise FileNotFoundError(f"No JSON result files found under: {experiment_dir}")
    return paths


def _validate_required_inputs(results_dir: Path, input_names: list[str]) -> None:
    missing_inputs = []
    for input_name in input_names:
        input_path = results_dir / input_name
        if not input_path.exists():
            missing_inputs.append((input_path, "missing directory"))
            continue
        if not list(input_path.rglob("*.json")):
            missing_inputs.append((input_path, "no JSON result files"))

    if missing_inputs:
        raise MissingResultInputsError(missing_inputs)


def collect_metric_table(results_dir: str | Path, experiment: str) -> Path:
    results_dir = Path(results_dir)
    experiment_dir = results_dir / experiment
    rows = [metric_row(read_json(path)) for path in _json_paths(experiment_dir)]
    output_path = results_dir / "tables" / f"{experiment}.csv"
    write_csv(output_path, rows, METRIC_FIELDS)
    return output_path


def collect_runtime_table(results_dir: str | Path) -> Path:
    results_dir = Path(results_dir)
    experiment_dir = results_dir / "runtime"
    rows = [runtime_row(read_json(path)) for path in _json_paths(experiment_dir)]
    output_path = results_dir / "tables" / "runtime.csv"
    write_csv(output_path, rows, RUNTIME_FIELDS)
    return output_path


def collect_experiment(results_dir: str | Path, experiment: str) -> Path:
    results_dir = Path(results_dir)
    _validate_required_inputs(results_dir, [experiment])
    if experiment == "runtime":
        return collect_runtime_table(results_dir)
    if experiment in METRIC_EXPERIMENTS:
        return collect_metric_table(results_dir, experiment)
    raise ValueError(f"Unknown experiment: {experiment}")


def collect_all(results_dir: str | Path = "results") -> list[Path]:
    results_dir = Path(results_dir)
    _validate_required_inputs(results_dir, REQUIRED_INPUTS)

    output_paths = [
        collect_metric_table(results_dir, experiment)
        for experiment in METRIC_EXPERIMENTS
    ]
    output_paths.append(collect_runtime_table(results_dir))
    return output_paths


def parse_args():
    parser = argparse.ArgumentParser(
        description="Collect reproduction JSON results into CSV tables."
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results"),
        help="Directory containing structured JSON result files.",
    )
    parser.add_argument(
        "--experiment",
        choices=COLLECTOR_EXPERIMENTS,
        default="all",
        help="Experiment group to collect. Use 'all' to require every group.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        if args.experiment == "all":
            collect_all(args.results_dir)
        else:
            collect_experiment(args.results_dir, args.experiment)
    except MissingResultInputsError as exc:
        print(exc, file=sys.stderr)
        raise SystemExit(1) from None
    print(f"CSV tables written to: {args.results_dir / 'tables'}")


if __name__ == "__main__":
    main()
