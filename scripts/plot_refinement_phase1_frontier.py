#!/usr/bin/env python3
"""Plot per-grammar gamma frontier figures for the refinement phase-1 screen."""

from __future__ import annotations

import argparse
import csv
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any


ESTIMATOR_ORDER = ("harmonic", "min")
ESTIMATOR_LABELS = {
    "harmonic": "Harmonic",
    "min": "Min",
}
ESTIMATOR_COLORS = {
    "harmonic": "#2563eb",
    "min": "#c2410c",
}
ESTIMATOR_MARKERS = {
    "harmonic": "o",
    "min": "s",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(
            "results/extension/refinement_phase1_frontier_n100/frontier_records.csv"
        ),
        help="Long-form frontier CSV produced by prepare_refinement_phase1_frontier.py.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/extension/refinement_phase1_frontier_n100/figures"),
        help="Directory for per-grammar PDF/PNG outputs.",
    )
    parser.add_argument(
        "--prefix",
        default="frontier_n100",
        help="Output filename prefix.",
    )
    parser.add_argument("--width", type=float, default=3.45)
    parser.add_argument("--height", type=float, default=3.45)
    return parser.parse_args()


def configure_matplotlib():
    import matplotlib

    matplotlib.use("Agg")
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


def slugify(value: str) -> str:
    value = value.lower()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    return value.strip("_")


def parse_float(row: dict[str, str], key: str) -> float:
    value = row.get(key, "")
    if value == "" or value is None:
        return math.nan
    return float(value)


def load_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append(
                {
                    "grammar": row["grammar"],
                    "estimator": row["estimator"],
                    "gamma": float(row["gamma"]),
                    "path_coverage_pct": parse_float(row, "path_coverage_pct"),
                    "vendi_score": parse_float(row, "vendi_score"),
                    "reject_pct": (
                        100.0 * parse_float(row, "reject_ratio")
                        if row.get("reject_ratio")
                        else math.nan
                    ),
                    "source_kind": row.get("source_kind"),
                }
            )
    return rows


def group_rows(rows: list[dict[str, Any]]) -> dict[str, dict[str, list[dict[str, Any]]]]:
    grouped: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in rows:
        grouped[row["grammar"]][row["estimator"]].append(row)
    for by_estimator in grouped.values():
        for estimator_rows in by_estimator.values():
            estimator_rows.sort(key=lambda item: item["gamma"])
    return grouped


def finite_values(values: list[float]) -> list[float]:
    return [value for value in values if not math.isnan(value)]


def set_padded_ylim(axis, values: list[float], floor_at_zero: bool = True) -> None:
    finite = finite_values(values)
    if not finite:
        return
    ymin = min(finite)
    ymax = max(finite)
    if ymin == ymax:
        pad = 1.0 if ymax == 0 else abs(ymax) * 0.08
    else:
        pad = (ymax - ymin) * 0.12
    lower = ymin - pad
    upper = ymax + pad
    if floor_at_zero:
        lower = max(0.0, lower)
    axis.set_ylim(lower, upper)


def plot_grammar(
    plt,
    grammar: str,
    by_estimator: dict[str, list[dict[str, Any]]],
    args: argparse.Namespace,
) -> list[Path]:
    fig, axes = plt.subplots(
        3,
        1,
        figsize=(args.width, args.height),
        sharex=True,
        gridspec_kw={"height_ratios": [1.35, 1.05, 1.0], "hspace": 0.12},
    )
    coverage_axis, vendi_axis, reject_axis = axes

    all_coverage: list[float] = []
    all_vendi: list[float] = []
    all_reject: list[float] = []

    for estimator in ESTIMATOR_ORDER:
        rows = by_estimator.get(estimator, [])
        if not rows:
            continue
        gammas = [row["gamma"] for row in rows]
        coverage = [row["path_coverage_pct"] for row in rows]
        vendi = [row["vendi_score"] for row in rows]
        reject = [row["reject_pct"] for row in rows]
        all_coverage.extend(coverage)
        all_vendi.extend(vendi)
        all_reject.extend(reject)

        line_kwargs = {
            "label": ESTIMATOR_LABELS[estimator],
            "color": ESTIMATOR_COLORS[estimator],
            "marker": ESTIMATOR_MARKERS[estimator],
            "linewidth": 1.4,
            "markersize": 3.2,
        }
        coverage_axis.plot(gammas, coverage, **line_kwargs)
        vendi_axis.plot(gammas, vendi, **line_kwargs)
        reject_axis.plot(gammas, reject, **line_kwargs)

    coverage_axis.set_title(grammar, pad=3)
    coverage_axis.set_ylabel("Path cov. (%)")
    vendi_axis.set_ylabel("Vendi")
    reject_axis.set_ylabel("Reject (%)")
    reject_axis.set_xlabel(r"$\gamma$")

    for axis in axes:
        axis.grid(True, which="major", alpha=0.25, linewidth=0.5)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)

    gammas = sorted({row["gamma"] for rows in by_estimator.values() for row in rows})
    reject_axis.set_xticks(gammas)
    reject_axis.set_xticklabels([f"{gamma:.1f}" for gamma in gammas])

    set_padded_ylim(coverage_axis, all_coverage)
    set_padded_ylim(vendi_axis, all_vendi)
    set_padded_ylim(reject_axis, all_reject)

    handles, labels = coverage_axis.get_legend_handles_labels()
    coverage_axis.legend(handles, labels, frameon=False, loc="best")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{args.prefix}_{slugify(grammar)}"
    pdf_path = args.output_dir / f"{stem}.pdf"
    png_path = args.output_dir / f"{stem}.png"
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=240, bbox_inches="tight")
    plt.close(fig)
    return [pdf_path, png_path]


def main() -> None:
    args = parse_args()
    rows = load_rows(args.input)
    grouped = group_rows(rows)
    plt = configure_matplotlib()

    outputs: list[Path] = []
    for grammar in sorted(grouped):
        outputs.extend(plot_grammar(plt, grammar, grouped[grammar], args))

    print(f"[plot] grammars={len(grouped)} files={len(outputs)}")
    for output in outputs:
        print(f"[plot] {output}")


if __name__ == "__main__":
    main()
