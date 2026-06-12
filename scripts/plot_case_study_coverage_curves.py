#!/usr/bin/env python3
"""Plot downstream case-study coverage growth curves.

This script expects the long-form CSV produced by the case-study coverage
budget sweep. It intentionally keeps plotting separate from experiment
execution, so figure style can be adjusted without rerunning coverage jobs.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path


CASE_ORDER = ("email_validator", "webcolors", "python_semver")
CASE_LABELS = {
    "email_validator": "(a) email-validator",
    "webcolors": "(b) webcolors",
    "python_semver": "(c) python-semver",
}

MODEL_ORDER = (
    "qwen2.5-0.5b-instruct",
    "qwen2.5-1.5b-instruct",
    "phi-4-mini-instruct",
)
MODEL_LABELS = {
    "qwen2.5-0.5b-instruct": "Qwen2.5-0.5B",
    "qwen2.5-1.5b-instruct": "Qwen2.5-1.5B",
    "phi-4-mini-instruct": "Phi-4-mini",
}

METHOD_ORDER = ("baseline_outlines", "diverse_guide", "diverse_guide_plus")
METHOD_LABELS = {
    "baseline_outlines": "Outlines",
    "diverse_guide": "DiverseGuide",
    "diverse_guide_plus": "DiverseGuide+",
}
METHOD_COLORS = {
    "baseline_outlines": "#4c566a",
    "diverse_guide": "#2b6cb0",
    "diverse_guide_plus": "#c2410c",
}
METHOD_MARKERS = {
    "baseline_outlines": "o",
    "diverse_guide": "s",
    "diverse_guide_plus": "^",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(
            "results/extension/case_study_coverage_curves/coverage_curves.csv"
        ),
        help="Long-form CSV with case_study, model, method, sample_count, and percent_covered columns.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/extension/case_study_coverage_curves"),
        help="Directory for PDF outputs.",
    )
    parser.add_argument(
        "--prefix",
        default="case_study_coverage",
        help="Output filename prefix.",
    )
    parser.add_argument(
        "--width",
        type=float,
        default=7.0,
        help="Combined figure width in inches, suitable for acmart figure*.",
    )
    parser.add_argument(
        "--height",
        type=float,
        default=2.35,
        help="Combined figure height in inches.",
    )
    parser.add_argument(
        "--no-band",
        action="store_true",
        help="Disable min-max bands across models.",
    )
    parser.add_argument(
        "--no-grid",
        action="store_true",
        help="Do not emit the 3x3 case-study-by-model grid figure.",
    )
    return parser.parse_args()


def load_rows(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            row["sample_count"] = int(row["sample_count"])
            row["percent_covered"] = float(row["percent_covered"])
            rows.append(row)
    return rows


def summarize(rows: list[dict]) -> dict[tuple[str, str, int], dict[str, float]]:
    grouped: dict[tuple[str, str, int], list[float]] = defaultdict(list)
    for row in rows:
        key = (row["case_study"], row["method"], row["sample_count"])
        grouped[key].append(row["percent_covered"])

    summary = {}
    for key, values in grouped.items():
        summary[key] = {
            "mean": sum(values) / len(values),
            "min": min(values),
            "max": max(values),
        }
    return summary


def configure_matplotlib():
    import matplotlib

    matplotlib.use("pdf")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 7,
            "axes.labelsize": 7,
            "axes.titlesize": 8,
            "legend.fontsize": 7,
            "xtick.labelsize": 6,
            "ytick.labelsize": 6,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    return plt


def plot_combined(rows: list[dict], args: argparse.Namespace) -> Path:
    plt = configure_matplotlib()
    summary = summarize(rows)
    budgets = sorted({row["sample_count"] for row in rows})

    fig, axes = plt.subplots(
        1,
        len(CASE_ORDER),
        figsize=(args.width, args.height),
        sharex=True,
        constrained_layout=False,
    )

    for axis, case_study in zip(axes, CASE_ORDER, strict=True):
        for method in METHOD_ORDER:
            means = [summary[(case_study, method, budget)]["mean"] for budget in budgets]
            mins = [summary[(case_study, method, budget)]["min"] for budget in budgets]
            maxs = [summary[(case_study, method, budget)]["max"] for budget in budgets]
            axis.plot(
                budgets,
                means,
                label=METHOD_LABELS[method],
                color=METHOD_COLORS[method],
                marker=METHOD_MARKERS[method],
                linewidth=1.4,
                markersize=3.2,
            )
            if not args.no_band:
                axis.fill_between(
                    budgets,
                    mins,
                    maxs,
                    color=METHOD_COLORS[method],
                    alpha=0.10,
                    linewidth=0,
                )

        axis.set_xscale("log")
        axis.set_xticks(budgets)
        axis.set_xticklabels([str(budget) for budget in budgets], rotation=35)
        axis.set_title(CASE_LABELS[case_study], pad=3)
        axis.grid(True, which="major", alpha=0.25, linewidth=0.5)

    axes[0].set_ylabel("Coverage (%)")
    for axis in axes:
        axis.set_xlabel("Samples")

    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 1.03),
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93), w_pad=1.0)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / f"{args.prefix}_combined_acmart.pdf"
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_individual(rows: list[dict], args: argparse.Namespace) -> list[Path]:
    plt = configure_matplotlib()
    summary = summarize(rows)
    budgets = sorted({row["sample_count"] for row in rows})
    outputs = []

    for case_study in CASE_ORDER:
        fig, axis = plt.subplots(figsize=(3.3, 2.25))
        for method in METHOD_ORDER:
            means = [summary[(case_study, method, budget)]["mean"] for budget in budgets]
            mins = [summary[(case_study, method, budget)]["min"] for budget in budgets]
            maxs = [summary[(case_study, method, budget)]["max"] for budget in budgets]
            axis.plot(
                budgets,
                means,
                label=METHOD_LABELS[method],
                color=METHOD_COLORS[method],
                marker=METHOD_MARKERS[method],
                linewidth=1.4,
                markersize=3.2,
            )
            if not args.no_band:
                axis.fill_between(
                    budgets,
                    mins,
                    maxs,
                    color=METHOD_COLORS[method],
                    alpha=0.10,
                    linewidth=0,
                )
        axis.set_xscale("log")
        axis.set_xticks(budgets)
        axis.set_xticklabels([str(budget) for budget in budgets], rotation=35)
        axis.set_xlabel("Samples")
        axis.set_ylabel("Coverage (%)")
        axis.set_title(CASE_LABELS[case_study], pad=3)
        axis.grid(True, which="major", alpha=0.25, linewidth=0.5)
        axis.legend(frameon=False, loc="lower right")
        fig.tight_layout()

        output_path = args.output_dir / f"{args.prefix}_{case_study}_acmart.pdf"
        fig.savefig(output_path, bbox_inches="tight")
        plt.close(fig)
        outputs.append(output_path)

    return outputs


def plot_model_grid(rows: list[dict], args: argparse.Namespace) -> Path:
    plt = configure_matplotlib()
    budgets = sorted({row["sample_count"] for row in rows})
    index = {
        (row["case_study"], row["model"], row["method"], row["sample_count"]): row[
            "percent_covered"
        ]
        for row in rows
    }

    fig, axes = plt.subplots(
        len(CASE_ORDER),
        len(MODEL_ORDER),
        figsize=(args.width, 5.25),
        sharex=True,
        constrained_layout=False,
    )

    for row_index, case_study in enumerate(CASE_ORDER):
        for col_index, model in enumerate(MODEL_ORDER):
            axis = axes[row_index][col_index]
            for method in METHOD_ORDER:
                values = [
                    index[(case_study, model, method, budget)] for budget in budgets
                ]
                axis.plot(
                    budgets,
                    values,
                    label=METHOD_LABELS[method],
                    color=METHOD_COLORS[method],
                    marker=METHOD_MARKERS[method],
                    linewidth=1.15,
                    markersize=2.6,
                )

            axis.set_xscale("log")
            axis.set_xticks(budgets)
            axis.set_xticklabels([str(budget) for budget in budgets], rotation=35)
            axis.grid(True, which="major", alpha=0.22, linewidth=0.45)

            if row_index == 0:
                axis.set_title(MODEL_LABELS[model], pad=3)
            if col_index == 0:
                axis.set_ylabel(f"{CASE_LABELS[case_study]}\nCoverage (%)")
            if row_index == len(CASE_ORDER) - 1:
                axis.set_xlabel("Samples")

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 1.02),
    )
    fig.tight_layout(rect=(0, 0, 1, 0.965), h_pad=0.8, w_pad=0.7)

    output_path = args.output_dir / f"{args.prefix}_model_grid_acmart.pdf"
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main() -> None:
    args = parse_args()
    rows = load_rows(args.input)
    outputs = [plot_combined(rows, args), *plot_individual(rows, args)]
    if not args.no_grid:
        outputs.append(plot_model_grid(rows, args))
    for output in outputs:
        print(output)


if __name__ == "__main__":
    main()
