"""Branch-aware balance metrics for min vs harmonic refinement pilots.

This is an exploratory analysis script. It intentionally avoids model/tokenizer
loading and reads only generated JSON artifacts. Coverage denominators are
observed feature-unit unions within each collection/grammar/region, so the
numbers are diagnostic rather than final paper metrics.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Any


DEFAULT_SOURCE_ROOT = Path(
    "/home/lxk/projects/diverse-dfa-gen/.worktrees/master-counter-fix"
)

SOURCE_SPECS = [
    (
        "phase1_n100",
        "data/extension/refinement_phase1_harmonic_min_screen_n100",
    ),
    (
        "phase1_n100",
        "data/extension/refinement_phase1_harmonic_min_screen_n100_gamma_high",
    ),
    (
        "phase1_n100",
        "data/extension/refinement_phase1_harmonic_mid_gamma_pilot_n100",
    ),
    (
        "phase1_email_css_n500",
        "data/extension/refinement_phase1_harmonic_min_500",
    ),
]


@dataclass(frozen=True)
class Cell:
    collection: str
    source_root: str
    method: str
    estimator: str
    gamma: float
    model_slug: str
    grammar: str
    path: Path
    samples: tuple[str, ...]
    accepted_samples: int
    rejected_samples: int
    elapsed_seconds: float | None


def parse_method(method: str) -> tuple[str, float] | None:
    match = re.fullmatch(r"(min|harmonic)_g([0-9]+(?:p[0-9]+)?)", method)
    if not match:
        return None
    return match.group(1), float(match.group(2).replace("p", "."))


def load_cells(source_root: Path) -> list[Cell]:
    cells: list[Cell] = []
    seen: set[tuple[str, str, str, float, str, str]] = set()
    for collection, rel_root in SOURCE_SPECS:
        root = source_root / rel_root
        if not root.exists():
            continue
        for path in sorted(root.glob("*/*/*.json")):
            method = path.parent.parent.name
            parsed = parse_method(method)
            if parsed is None:
                continue
            estimator, gamma = parsed
            model_slug = path.parent.name
            grammar = path.stem
            key = (collection, estimator, method, gamma, model_slug, grammar)
            if key in seen:
                continue
            seen.add(key)
            with path.open(encoding="utf-8") as handle:
                data = json.load(handle)
            samples = tuple(str(sample) for sample in data.get("samples", []))
            cells.append(
                Cell(
                    collection=collection,
                    source_root=str(root),
                    method=method,
                    estimator=estimator,
                    gamma=gamma,
                    model_slug=model_slug,
                    grammar=grammar,
                    path=path,
                    samples=samples,
                    accepted_samples=int(data.get("accepted_samples") or len(samples)),
                    rejected_samples=int(data.get("rejected_samples") or 0),
                    elapsed_seconds=data.get("elapsed_seconds"),
                )
            )
    return cells


def classify_region(grammar: str, sample: str) -> str:
    text = sample.strip()
    lower = text.lower()

    if grammar == "api-route":
        path = lower[1:] if lower.startswith("/") else lower
        parts = path.split("?", 1)[0].strip("/").split("/")
        if len(parts) >= 2 and parts[0] == "api":
            parts = parts[2:] if len(parts) >= 2 else []
        first = parts[0] if parts else ""
        if first in {"users", "accounts", "customers"}:
            return "entity"
        if first in {"orgs", "teams"}:
            return "org_project"
        if first in {"files", "assets"}:
            return "file_asset"
        if first in {"search,healthz,metrics"}:
            return "utility"
        if first in {"search", "healthz", "metrics"}:
            return "utility"
        return "unknown"

    if grammar == "css-color":
        if lower.startswith("#"):
            return "hex"
        func = lower.split("(", 1)[0]
        if func in {"rgb", "rgba"}:
            return "rgb"
        if func in {"hsl", "hsla"}:
            return "hsl"
        if func == "hwb":
            return "hwb"
        if func in {"lab", "oklab"}:
            return "lab"
        if func in {"lch", "oklch"}:
            return "lch"
        if func == "color":
            return "color_function"
        if re.fullmatch(r"[a-z]+", lower):
            return "named"
        return "unknown"

    if grammar == "email":
        local, sep, domain = text.partition("@")
        domain_lower = domain.lower()
        if domain_lower.startswith("["):
            return "domain_literal"
        if local.startswith('"'):
            return "quoted_local"
        if "." in local:
            return "dotted_local"
        if "+" in local:
            return "tagged_local"
        return "simple_local"

    if grammar == "semver":
        has_pre = "-" in text
        has_build = "+" in text
        if has_pre and has_build:
            return "prerelease_build"
        if has_pre:
            return "prerelease"
        if has_build:
            return "build"
        return "core"

    if grammar == "json":
        gender = "unknown_gender"
        try:
            obj = json.loads(text)
            raw_gender = str(obj.get("gender", "")).lower()
            if raw_gender in {"male", "female"}:
                gender = raw_gender
        except Exception:
            match = re.search(r'"gender"\s*:\s*"(fe)?male"', text)
            if match:
                gender = "female" if match.group(1) else "male"
        return f"gender_{gender}"

    if grammar == "no-bomb":
        if "bom" in lower:
            return "contains_bom"
        if "bo" in lower:
            return "contains_bo"
        if "b" in lower:
            return "contains_b"
        return "no_b_prefix"

    return "all"


def char_ngrams(text: str, n: int = 3) -> set[str]:
    normalized = re.sub(r"\s+", " ", text.lower().strip())
    if len(normalized) <= n:
        return {f"char:{normalized}"}
    return {f"char:{normalized[idx:idx+n]}" for idx in range(len(normalized) - n + 1)}


def bucket_number(value: str) -> str:
    try:
        number = int(value)
    except ValueError:
        return "num:nonint"
    if number < 10:
        return "num:0-9"
    if number < 100:
        return "num:10-99"
    if number < 1000:
        return "num:100-999"
    return "num:1000+"


def feature_units(grammar: str, sample: str, region: str) -> set[str]:
    text = sample.strip()
    lower = text.lower()
    units = {f"region:{region}"}
    units |= char_ngrams(text, 3)

    if grammar == "api-route":
        path, _, query = lower.partition("?")
        parts = [part for part in path.strip("/").split("/") if part]
        for idx, part in enumerate(parts):
            if part.startswith(":"):
                units.add(f"seg{idx}:colon_param")
            elif part.startswith("{") and part.endswith("}"):
                units.add(f"seg{idx}:brace_param")
            elif part.startswith("<") and part.endswith(">"):
                units.add(f"seg{idx}:angle_param")
            elif part in {"*", "**"}:
                units.add(f"seg{idx}:glob")
            else:
                units.add(f"seg{idx}:{part}")
        if query:
            for key in re.findall(r"([a-z_]+)=", query):
                units.add(f"query:{key}")

    elif grammar == "css-color":
        units.add(f"css:{region}")
        for number in re.findall(r"-?\d+(?:\.\d+)?", lower):
            units.add(bucket_number(number.split(".", 1)[0].lstrip("-")))
        for pct in re.findall(r"\d+(?:\.\d+)?%", lower):
            units.add(f"pct:{bucket_number(pct.rstrip('%').split('.', 1)[0])}")

    elif grammar == "email":
        local, _, domain = lower.partition("@")
        units.add(f"local_len:{min(len(local) // 4, 8)}")
        labels = [label for label in domain.strip("[]").split(".") if label]
        if labels:
            units.add(f"tld:{labels[-1][-4:]}")
            units.add(f"domain_labels:{min(len(labels), 5)}")
        for ch in set(local):
            if ch.isalnum():
                units.add(f"local_char:{ch}")
            else:
                units.add(f"local_punct:{ch}")

    elif grammar == "semver":
        main, *suffix = re.split(r"[-+]", text, maxsplit=1)
        parts = main.split(".")
        for idx, part in enumerate(parts[:3]):
            units.add(f"core{idx}_digits:{len(part)}")
            units.add(f"core{idx}_{bucket_number(part)}")
        for ident in re.split(r"[.+-]", text):
            if ident:
                units.add(f"ident_len:{min(len(ident), 12)}")
                if any(ch.isalpha() for ch in ident):
                    units.add("ident:alpha")

    elif grammar == "json":
        for key in re.findall(r'"([^"]+)"\s*:', text):
            units.add(f"key:{key}")
        age_match = re.search(r'"age"\s*:\s*(\d+)', text)
        if age_match:
            units.add(bucket_number(age_match.group(1)))

    elif grammar == "no-bomb":
        for prefix in ["b", "bo", "bom"]:
            if prefix in lower:
                units.add(f"partial:{prefix}")
        units.add(f"len_bucket:{min(len(text) // 8, 12)}")

    return units


def harmonic_mean(values: list[float]) -> float:
    if not values or any(value <= 0.0 for value in values):
        return 0.0
    return len(values) / sum(1.0 / value for value in values)


def normalized_entropy(counts: Counter[str], region_count: int) -> float:
    total = sum(counts.values())
    if total == 0 or region_count <= 1:
        return 0.0
    entropy = 0.0
    for count in counts.values():
        if count:
            p = count / total
            entropy -= p * math.log(p)
    return entropy / math.log(region_count)


def effective_branches(counts: Counter[str]) -> float:
    total = sum(counts.values())
    if total == 0:
        return 0.0
    entropy = 0.0
    for count in counts.values():
        if count:
            p = count / total
            entropy -= p * math.log(p)
    return math.exp(entropy)


def auc(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def select_rare_regions(
    cells: list[Cell],
    universes: dict[str, set[str]],
) -> tuple[set[str], str]:
    min_cells = [cell for cell in cells if cell.estimator == "min"]
    if min_cells:
        reference = min(min_cells, key=lambda cell: cell.gamma)
        counts = Counter(classify_region(reference.grammar, sample) for sample in reference.samples)
        source = f"min_g{reference.gamma:g}"
    else:
        counts = Counter()
        for cell in cells:
            counts.update(classify_region(cell.grammar, sample) for sample in cell.samples)
        source = "aggregate"

    candidate_regions = [region for region in universes if universes[region]]
    if not candidate_regions:
        return set(), source
    values = [counts.get(region, 0) for region in candidate_regions]
    threshold = median(values)
    rare = {region for region in candidate_regions if counts.get(region, 0) <= threshold}
    if len(rare) == len(candidate_regions) and len(candidate_regions) > 1:
        rare = set(sorted(candidate_regions, key=lambda region: counts.get(region, 0))[: max(1, len(candidate_regions) // 2)])
    return rare, source


def compute_cell_metrics(
    cell: Cell,
    universes: dict[str, set[str]],
    rare_regions: set[str],
) -> dict[str, Any]:
    all_regions = sorted(region for region, units in universes.items() if units)
    covered: dict[str, set[str]] = {region: set() for region in all_regions}
    region_counts: Counter[str] = Counter()
    prior_unit_counts: dict[str, Counter[str]] = defaultdict(Counter)
    brc_curve: list[float] = []
    rare_curve: list[float] = []
    single_unseen_dominated = 0

    for sample in cell.samples:
        region = classify_region(cell.grammar, sample)
        units = feature_units(cell.grammar, sample, region)
        region_counts[region] += 1
        if region in covered:
            prior_counts = [prior_unit_counts[region].get(unit, 0) for unit in units]
            if len(prior_counts) >= 3 and min(prior_counts) == 0 and median(prior_counts) >= 2:
                single_unseen_dominated += 1
            covered[region].update(units)
            prior_unit_counts[region].update(units)

        covs = [
            len(covered[reg]) / len(universes[reg])
            for reg in all_regions
            if universes[reg]
        ]
        brc_curve.append(harmonic_mean(covs))
        rare_covs = [
            len(covered[reg]) / len(universes[reg])
            for reg in sorted(rare_regions)
            if universes.get(reg)
        ]
        rare_curve.append(sum(rare_covs) / len(rare_covs) if rare_covs else 0.0)

    final_covs = {
        region: (len(covered[region]) / len(universes[region]) if universes[region] else 0.0)
        for region in all_regions
    }
    total_samples = len(cell.samples)
    reject_attempts = cell.accepted_samples + cell.rejected_samples
    dominant_share = (
        max(region_counts.values()) / total_samples if total_samples and region_counts else 0.0
    )
    return {
        "collection": cell.collection,
        "source_root": cell.source_root,
        "method": cell.method,
        "estimator": cell.estimator,
        "gamma": cell.gamma,
        "model_slug": cell.model_slug,
        "grammar": cell.grammar,
        "path": str(cell.path),
        "sample_count": total_samples,
        "accepted_samples": cell.accepted_samples,
        "rejected_samples": cell.rejected_samples,
        "reject_ratio": cell.rejected_samples / reject_attempts if reject_attempts else 0.0,
        "elapsed_seconds": cell.elapsed_seconds,
        "regions": all_regions,
        "rare_regions": sorted(rare_regions),
        "region_counts": dict(sorted(region_counts.items())),
        "region_coverage": final_covs,
        "brc_at_n": brc_curve[-1] if brc_curve else 0.0,
        "brc_auc": auc(brc_curve),
        "rare_cov_at_n": rare_curve[-1] if rare_curve else 0.0,
        "rare_cov_auc": auc(rare_curve),
        "branch_entropy": normalized_entropy(region_counts, len(all_regions)),
        "effective_branches": effective_branches(region_counts),
        "dominant_region_share": dominant_share,
        "single_unseen_unit_dominated_rate": (
            single_unseen_dominated / total_samples if total_samples else 0.0
        ),
    }


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "collection",
        "grammar",
        "model_slug",
        "estimator",
        "gamma",
        "sample_count",
        "reject_ratio",
        "brc_at_n",
        "brc_auc",
        "rare_cov_at_n",
        "rare_cov_auc",
        "branch_entropy",
        "effective_branches",
        "dominant_region_share",
        "single_unseen_unit_dominated_rate",
        "rare_regions",
        "region_counts",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    field: json.dumps(row[field], sort_keys=True)
                    if isinstance(row.get(field), (dict, list))
                    else row.get(field)
                    for field in fields
                }
            )


def compute_pairwise(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_key: dict[tuple[str, str, str, float, int], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        key = (
            row["collection"],
            row["grammar"],
            row["model_slug"],
            row["gamma"],
            row["sample_count"],
        )
        by_key[key][row["estimator"]] = row

    deltas: list[dict[str, Any]] = []
    for key, pair in sorted(by_key.items()):
        if "min" not in pair or "harmonic" not in pair:
            continue
        min_row = pair["min"]
        harm_row = pair["harmonic"]
        collection, grammar, model_slug, gamma, sample_count = key
        delta = {
            "collection": collection,
            "grammar": grammar,
            "model_slug": model_slug,
            "gamma": gamma,
            "sample_count": sample_count,
            "harmonic_brc_auc": harm_row["brc_auc"],
            "min_brc_auc": min_row["brc_auc"],
            "delta_brc_auc": harm_row["brc_auc"] - min_row["brc_auc"],
            "harmonic_rare_cov_auc": harm_row["rare_cov_auc"],
            "min_rare_cov_auc": min_row["rare_cov_auc"],
            "delta_rare_cov_auc": harm_row["rare_cov_auc"] - min_row["rare_cov_auc"],
            "harmonic_branch_entropy": harm_row["branch_entropy"],
            "min_branch_entropy": min_row["branch_entropy"],
            "delta_branch_entropy": harm_row["branch_entropy"] - min_row["branch_entropy"],
            "harmonic_dominant_region_share": harm_row["dominant_region_share"],
            "min_dominant_region_share": min_row["dominant_region_share"],
            "delta_dominant_region_share": harm_row["dominant_region_share"] - min_row["dominant_region_share"],
            "harmonic_single_unseen_unit_dominated_rate": harm_row["single_unseen_unit_dominated_rate"],
            "min_single_unseen_unit_dominated_rate": min_row["single_unseen_unit_dominated_rate"],
            "delta_single_unseen_unit_dominated_rate": (
                harm_row["single_unseen_unit_dominated_rate"]
                - min_row["single_unseen_unit_dominated_rate"]
            ),
            "harmonic_reject_ratio": harm_row["reject_ratio"],
            "min_reject_ratio": min_row["reject_ratio"],
            "delta_reject_ratio": harm_row["reject_ratio"] - min_row["reject_ratio"],
        }
        deltas.append(delta)
    return deltas


def format_float(value: float) -> str:
    return f"{value:.3f}"


def write_summary(path: Path, rows: list[dict[str, Any]], deltas: list[dict[str, Any]]) -> None:
    metric_wins = {
        "brc_auc": sum(1 for row in deltas if row["delta_brc_auc"] > 0),
        "rare_cov_auc": sum(1 for row in deltas if row["delta_rare_cov_auc"] > 0),
        "branch_entropy": sum(1 for row in deltas if row["delta_branch_entropy"] > 0),
        "lower_dominant_share": sum(1 for row in deltas if row["delta_dominant_region_share"] < 0),
        "lower_single_unseen": sum(
            1 for row in deltas if row["delta_single_unseen_unit_dominated_rate"] < 0
        ),
    }

    lines: list[str] = []
    lines.append("# Branch Balance Metric Probe")
    lines.append("")
    lines.append(
        "Coverage denominators are observed feature-unit unions per collection/grammar/region. "
        "This is a diagnostic probe, not a final DFA-edge metric."
    )
    lines.append("")
    lines.append(f"Rows analyzed: {len(rows)}")
    lines.append(f"Paired min/harmonic comparisons: {len(deltas)}")
    lines.append("")
    lines.append("## Harmonic Wins")
    lines.append("")
    for metric, wins in metric_wins.items():
        total = len(deltas)
        lines.append(f"- {metric}: {wins}/{total}")

    lines.append("")
    lines.append("## Pairwise Deltas")
    lines.append("")
    lines.append(
        "| collection | grammar | gamma | n | ΔBRC-AUC | ΔRare-AUC | ΔEntropy | ΔDominant | ΔSingle-unseen | ΔReject |"
    )
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in deltas:
        lines.append(
            "| {collection} | {grammar} | {gamma:.2f} | {sample_count} | {brc} | {rare} | {entropy} | {dom} | {single} | {reject} |".format(
                collection=row["collection"],
                grammar=row["grammar"],
                gamma=row["gamma"],
                sample_count=row["sample_count"],
                brc=format_float(row["delta_brc_auc"]),
                rare=format_float(row["delta_rare_cov_auc"]),
                entropy=format_float(row["delta_branch_entropy"]),
                dom=format_float(row["delta_dominant_region_share"]),
                single=format_float(row["delta_single_unseen_unit_dominated_rate"]),
                reject=format_float(row["delta_reject_ratio"]),
            )
        )

    lines.append("")
    lines.append("## Largest Harmonic Improvements")
    lines.append("")
    for metric in ["delta_brc_auc", "delta_rare_cov_auc", "delta_branch_entropy"]:
        lines.append(f"### {metric}")
        for row in sorted(deltas, key=lambda item: item[metric], reverse=True)[:8]:
            lines.append(
                f"- {row['collection']} {row['grammar']} g={row['gamma']:.2f} n={row['sample_count']}: {row[metric]:.3f}"
            )
        lines.append("")

    lines.append("## Largest Harmonic Regressions")
    lines.append("")
    for metric in ["delta_brc_auc", "delta_rare_cov_auc", "delta_branch_entropy"]:
        lines.append(f"### {metric}")
        for row in sorted(deltas, key=lambda item: item[metric])[:8]:
            lines.append(
                f"- {row['collection']} {row['grammar']} g={row['gamma']:.2f} n={row['sample_count']}: {row[metric]:.3f}"
            )
        lines.append("")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/extension/branch_balance_metrics"),
    )
    args = parser.parse_args()

    cells = load_cells(args.source_root)
    if not cells:
        raise SystemExit(f"No generated cells found under {args.source_root}")

    grouped: dict[tuple[str, str, str], list[Cell]] = defaultdict(list)
    for cell in cells:
        grouped[(cell.collection, cell.model_slug, cell.grammar)].append(cell)

    rows: list[dict[str, Any]] = []
    for (_collection, _model_slug, _grammar), group_cells in grouped.items():
        universes: dict[str, set[str]] = defaultdict(set)
        for cell in group_cells:
            for sample in cell.samples:
                region = classify_region(cell.grammar, sample)
                universes[region].update(feature_units(cell.grammar, sample, region))
        rare_regions, rare_source = select_rare_regions(group_cells, universes)
        for cell in group_cells:
            row = compute_cell_metrics(cell, universes, rare_regions)
            row["rare_region_source"] = rare_source
            rows.append(row)

    rows.sort(
        key=lambda row: (
            row["collection"],
            row["grammar"],
            row["model_slug"],
            row["gamma"],
            row["estimator"],
        )
    )
    deltas = compute_pairwise(rows)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "branch_balance_metrics.jsonl", rows)
    write_jsonl(args.output_dir / "branch_balance_pairwise_deltas.jsonl", deltas)
    write_csv(args.output_dir / "branch_balance_metrics.csv", rows)
    write_summary(args.output_dir / "branch_balance_summary.md", rows, deltas)
    print(f"[metrics] rows={len(rows)} pairs={len(deltas)} output={args.output_dir}")


if __name__ == "__main__":
    main()
