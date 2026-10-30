"""Evaluate diversity metrics as a function of sample count and plot trends."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from diverse_guide.evaluation import string_kernel
from diverse_guide.evaluation.paths import get_data_dir_path
from regex_dfa_guide import DiverseGuideDFA

try:
    from metrics_eval import compute_metrics
    from repro_results import build_metadata, write_json
except ModuleNotFoundError:
    from scripts.metrics_eval import compute_metrics
    from scripts.repro_results import build_metadata, write_json


TREND_METRICS = [
    ("state_coverage", "State Coverage"),
    ("transition_coverage", "Transition Coverage"),
    ("path_coverage", "Path Coverage"),
    ("vendi_score", "Vendi Score"),
    ("distinct_2gram_count", "Distinct 2-gram Count"),
    ("distinct_3gram_count", "Distinct 3-gram Count"),
]

SETTING_STYLE = {
    "baseline": {"color": "#6c757d", "marker": "o"},
    "diverse": {"color": "#1f77b4", "marker": "s"},
}


def build_data_args(args: argparse.Namespace, baseline: bool) -> argparse.Namespace:
    return argparse.Namespace(
        model=args.model,
        baseline=baseline,
        top_k=args.top_k,
        top_p=args.top_p,
        temperature=args.temperature,
        ablation_component=args.ablation_component,
    )


def load_generation_data(
    args: argparse.Namespace, baseline: bool
) -> tuple[Path, dict[str, Any]]:
    data_args = build_data_args(args, baseline=baseline)
    json_path = get_data_dir_path(data_args) / f"{args.grammar}.json"
    with json_path.open(encoding="utf-8") as handle:
        return json_path, json.load(handle)


def build_sample_counts(limit: int, step: int) -> list[int]:
    counts = list(range(step, limit + 1, step))
    if not counts or counts[-1] != limit:
        counts.append(limit)
    return counts


def metric_value(result: dict[str, Any], metric_name: str) -> float:
    metrics = result["metrics"]
    if metric_name == "distinct_2gram_count":
        return float(metrics["distinct_2gram"][0])
    if metric_name == "distinct_3gram_count":
        return float(metrics["distinct_3gram"][0])
    return float(metrics[metric_name])


def build_series(
    dfa: DiverseGuideDFA,
    samples: list[str],
    sample_counts: list[int],
    d: int,
    s: int,
) -> list[dict[str, Any]]:
    series = []
    for count in sample_counts:
        result = compute_metrics(dfa, samples, d=d, s=s, n=count)
        point = {"sample_count": count}
        for metric_name, _ in TREND_METRICS:
            point[metric_name] = metric_value(result, metric_name)
        series.append(point)
    return series


def build_result(
    args: argparse.Namespace,
    baseline_path: Path,
    diverse_path: Path,
    baseline_series: list[dict[str, Any]],
    diverse_series: list[dict[str, Any]],
    limit: int,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "experiment": "diversity_trend",
        "grammar": args.grammar,
        "model": args.model,
        "input_paths": {
            "baseline": str(baseline_path),
            "diverse": str(diverse_path),
        },
        "parameters": {
            "d": args.d,
            "s": args.s,
            "step": args.step,
            "max_samples": args.max_samples,
            "evaluated_sample_count": limit,
            "top_k": args.top_k,
            "top_p": args.top_p,
            "temperature": args.temperature,
            "ablation_component": args.ablation_component,
        },
        "metadata": build_metadata(
            {"string_kernel_backend": string_kernel.STRING_KERNEL_BACKEND}
        ),
        "metrics": [metric_name for metric_name, _ in TREND_METRICS],
        "series": {
            "baseline": baseline_series,
            "diverse": diverse_series,
        },
    }


def default_output_paths(args: argparse.Namespace) -> tuple[Path, Path]:
    model_slug = args.model.split("/")[-1].lower()
    stem = args.grammar
    if args.temperature is not None:
        stem += f"-temperature_{args.temperature}"
    if args.top_k is not None:
        stem += f"-top_k_{args.top_k}"
    if args.top_p is not None:
        stem += f"-top_p_{args.top_p}"
    if args.ablation_component is not None:
        stem += f"-no_{args.ablation_component}"
    base_dir = Path("results") / "metric_trends" / model_slug
    return base_dir / f"{stem}.json", base_dir / f"{stem}.png"


def plot_trends(
    result: dict[str, Any],
    output_path: str | Path,
    title: str | None = None,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    baseline_series = result["series"]["baseline"]
    diverse_series = result["series"]["diverse"]
    x_values = [point["sample_count"] for point in baseline_series]

    fig, axes = plt.subplots(2, 3, figsize=(16, 9), constrained_layout=True)
    axes_flat = axes.flatten()

    for axis, (metric_name, metric_label) in zip(axes_flat, TREND_METRICS, strict=True):
        for setting, series in (
            ("baseline", baseline_series),
            ("diverse", diverse_series),
        ):
            style = SETTING_STYLE[setting]
            axis.plot(
                x_values,
                [point[metric_name] for point in series],
                label=setting,
                color=style["color"],
                marker=style["marker"],
                linewidth=2,
                markersize=5,
            )
        axis.set_title(metric_label)
        axis.set_xlabel("Sample Count")
        axis.grid(True, alpha=0.3)

    axes_flat[0].set_ylabel("Metric Value")
    axes_flat[3].set_ylabel("Metric Value")
    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
    fig.suptitle(
        title or f"Diversity Metric Trends: {result['model']} / {result['grammar']}",
        fontsize=14,
    )
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate diversity metrics as sample count increases."
    )
    parser.add_argument("grammar", type=str, help="Grammar name")
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2.5-1.5B-Instruct",
        help="Model name used to locate generation data.",
    )
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument(
        "--ablation-component",
        type=str,
        choices=["reward", "penalty", "range_scaling"],
        help="Component to ablate for diverse runs.",
    )
    parser.add_argument(
        "-d", type=int, default=5, help="Max k-mer degree for Vendi score"
    )
    parser.add_argument("-s", type=int, default=1, help="Max shift for Vendi score")
    parser.add_argument(
        "--step",
        type=int,
        default=50,
        help="Evaluate metrics every N samples.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional cap on evaluated samples.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Path to save structured trend results as JSON.",
    )
    parser.add_argument(
        "--output-figure",
        type=Path,
        default=None,
        help="Path to save the matplotlib figure.",
    )
    parser.add_argument(
        "--title",
        type=str,
        default=None,
        help="Optional figure title override.",
    )
    args = parser.parse_args()
    if args.step <= 0:
        parser.error("--step must be positive")
    if args.max_samples is not None and args.max_samples <= 0:
        parser.error("--max-samples must be positive")
    return args


def main() -> None:
    args = parse_args()
    baseline_path, baseline_data = load_generation_data(args, baseline=True)
    diverse_path, diverse_data = load_generation_data(args, baseline=False)

    baseline_regex = baseline_data["regex"]
    diverse_regex = diverse_data["regex"]
    if baseline_regex != diverse_regex:
        raise ValueError("Baseline and diverse regex definitions do not match")

    baseline_samples = baseline_data["samples"]
    diverse_samples = diverse_data["samples"]
    limit = min(len(baseline_samples), len(diverse_samples))
    if args.max_samples is not None:
        limit = min(limit, args.max_samples)
    sample_counts = build_sample_counts(limit=limit, step=args.step)

    regex = f"(?:{baseline_regex})$"
    dfa = DiverseGuideDFA(regex, 2**32 - 1, {})

    baseline_series = build_series(
        dfa, baseline_samples, sample_counts, d=args.d, s=args.s
    )
    diverse_series = build_series(
        dfa, diverse_samples, sample_counts, d=args.d, s=args.s
    )

    result = build_result(
        args=args,
        baseline_path=baseline_path,
        diverse_path=diverse_path,
        baseline_series=baseline_series,
        diverse_series=diverse_series,
        limit=limit,
    )

    output_json, output_figure = default_output_paths(args)
    if args.output_json is not None:
        output_json = args.output_json
    if args.output_figure is not None:
        output_figure = args.output_figure

    write_json(output_json, result)
    plot_trends(result, output_figure, title=args.title)

    print(f"Baseline input: {baseline_path}")
    print(f"Diverse input: {diverse_path}")
    print(f"Evaluated sample counts: {sample_counts}")
    print(f"Trend JSON saved to: {output_json}")
    print(f"Trend figure saved to: {output_figure}")


if __name__ == "__main__":
    main()
