"""Freeze and execute the EOS reentry diagnostic (4 smoke + 45 fresh runs)."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

from scripts.run_group1_task import (
    atomic_copy,
    atomic_json,
    completed_is_valid,
    digest,
    environment_snapshot,
    file_hash,
    run_task,
    verify_sources,
)

REPO = Path(__file__).resolve().parents[1]


def build_tasks(base, model_path, source_fingerprint):
    templates = {t["grammar"]: t for t in base["templates"]}
    cells = [
        ("smoke", "json", eta, mode, 42)
        for eta in [0, 2]
        for mode in ["legacy", "entry_count"]
    ]
    cells += [
        ("B", "json", eta, mode, seed)
        for eta in [1, 2]
        for mode in ["legacy", "entry_count"]
        for seed in range(42, 47)
    ]
    cells += [("C", "json", 2, "constant", seed) for seed in range(42, 47)]
    cells += [
        ("D", grammar, 1, mode, seed)
        for grammar in ["email", "css-color"]
        for mode in ["legacy", "entry_count"]
        for seed in range(42, 47)
    ]
    tasks = []
    for stage, grammar, eta, mode, seed in cells:
        task = dict(
            templates[grammar],
            stage=stage,
            gamma=0.3,
            completion_bias_eta=eta,
            eos_completion_mode=mode,
            seed=seed,
            model_path=str(model_path),
            path_frequency_estimator="harmonic",
            target_valid_samples=20 if stage == "smoke" else 500,
            source_fingerprint=source_fingerprint,
            step_probes=grammar == "json" and seed == 42 and eta == 2,
        )
        task["id"] = f"{stage}__{grammar}__eta-{eta}__{mode}__seed-{seed}"
        task["fingerprint"] = digest(task)
        tasks.append(task)
    return tasks


def prepare(args):
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=REPO, text=True
    )
    if dirty.strip():
        raise ValueError("freeze a clean committed source checkout")
    base = json.loads((REPO / "configs/eos-reentry/base.json").read_text())
    model = args.model_path.resolve()
    for name, expected in base["templates"][0]["model_config_sha256"].items():
        if file_hash(model / name) != expected:
            raise ValueError(f"model/tokenizer metadata mismatch: {name}")
    root = args.root.resolve()
    backup = args.backup.resolve()
    if root == backup or root in backup.parents or backup in root.parents:
        raise ValueError("backup must be independent of run directory")
    if root.exists():
        raise FileExistsError(root)
    paths = (
        subprocess.check_output(["git", "ls-files", "-z"], cwd=REPO)
        .decode()
        .split("\0")
    )
    hashes = {p: file_hash(REPO / p) for p in paths if p and (REPO / p).is_file()}
    environment = environment_snapshot()
    manifest = dict(
        source_root=str(REPO),
        source_sha256=hashes,
        environment=environment,
        source_commit=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO, text=True
        ).strip(),
        run_root=str(root),
        backup_root=str(backup),
        pilot_samples=20,
        max_attempts=50000,
        task_timeout_seconds=43200,
        min_ram_gib=8,
        min_disk_gib=10,
        min_vram_gib=8,
        source_manifest_sha256=base["source_manifest_sha256"],
        reuse_policy="all 45 statistical runs are fresh; no automatic reuse",
        tasks=build_tasks(
            base, model, digest(dict(source=hashes, environment=environment))
        ),
    )
    atomic_json(root / "manifest.json", manifest)
    atomic_copy(root / "manifest.json", backup / "manifest.json")
    print(root / "manifest.json")


def verified_payload(manifest, task):
    root = Path(manifest["run_root"]) / "production"
    state = root / "tasks" / task["id"] / "status.json"
    if not state.exists():
        raise ValueError(f"missing {task['id']}")
    status = json.loads(state.read_text())
    result = root / "data" / (task["id"] + ".json")
    if not completed_is_valid(status, task, result):
        raise ValueError(f"incomplete/corrupt {task['id']}")
    return json.loads(result.read_text()), status


def check_smoke(manifest):
    smoke = [t for t in manifest["tasks"] if t["stage"] == "smoke"]
    values = {}
    for t in smoke:
        payload, state = verified_payload(manifest, t)
        values[(t["completion_bias_eta"], t["eos_completion_mode"])] = payload[
            "samples"
        ]
    if values[(0, "legacy")] != values[(0, "entry_count")]:
        raise ValueError(
            "eta=0 smoke outputs differ: investigate numerical determinism "
            "before proceeding"
        )
    return True


def run_stage(args, manifest):
    verify_sources(manifest)
    if args.stage != "smoke":
        check_smoke(manifest)
    if args.stage in ["C", "D"]:
        if not args.reviewed_b:
            raise ValueError(
                "review stage B report first; pass --reviewed-b to continue"
            )
        for t in manifest["tasks"]:
            if t["stage"] == "B":
                verified_payload(manifest, t)
    if args.stage == "D":
        for t in manifest["tasks"]:
            if t["stage"] == "C":
                verified_payload(manifest, t)
    for task in manifest["tasks"]:
        if task["stage"] == args.stage:
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "scripts.eos_reentry",
                    "worker",
                    "--manifest",
                    str(args.manifest),
                    "--task",
                    task["id"],
                ],
                cwd=REPO,
                check=True,
            )
    if args.stage == "smoke":
        check_smoke(manifest)


def attempt_statistics(path):
    rows = []
    with Path(path).open() as f:
        for line in f:
            row = json.loads(line)
            if row["event"] == "finish":
                rows.append(row)
    if [r["attempt"] for r in rows] != list(range(1, len(rows) + 1)):
        raise ValueError("attempt journal contains duplicates or gaps")
    accepted = sum(r["valid"] for r in rows)
    if not rows or not accepted:
        raise ValueError("empty attempts or no accepted samples")
    total = len(rows)
    never = sum(not r["valid"] and not r["diagnostics"]["ever_accepting"] for r in rows)
    lost = sum(not r["valid"] and r["diagnostics"]["ever_accepting"] for r in rows)
    tokens = sum(len(r["token_ids"][0]) for r in rows)
    result = dict(
        attempts=total,
        accepted=accepted,
        rejected=total - accepted,
        reject_ratio=(total - accepted) / total,
        never_accepting_ratio=never / total,
        lost_after_accepting_ratio=lost / total,
        tokens_per_valid=tokens / accepted,
        total_tokens=tokens,
        accepted_tokens=sum(len(r["token_ids"][0]) for r in rows if r["valid"]),
    )
    result["rejected_tokens"] = tokens - result["accepted_tokens"]
    for bucket in ["1", "2", "3", "4"]:
        n = sum(r["diagnostics"]["eos_opportunities"].get(bucket, 0) for r in rows)
        k = sum(r["diagnostics"]["eos_selected"].get(bucket, 0) for r in rows)
        result["eos_opportunities_" + bucket] = n
        result["eos_selected_" + bucket] = k
        result["eos_rate_" + bucket] = k / n if n else None
    return result


def aggregate(args, manifest):
    rows = []
    incomplete = []
    for task in manifest["tasks"]:
        if task["stage"] == "smoke":
            continue
        try:
            payload, state = verified_payload(manifest, task)
        except ValueError as exc:
            incomplete.append(str(exc))
            continue
        ev = json.loads(Path(state["evaluation_path"]).read_text())["record"]
        stats = attempt_statistics(Path(state["attempt_dir"]) / "attempts.jsonl")
        if stats["accepted"] != 500 or stats["rejected"] != payload["rejected_samples"]:
            raise ValueError("journal/result count mismatch")
        row = {
            k: task[k]
            for k in [
                "id",
                "stage",
                "grammar",
                "seed",
                "completion_bias_eta",
                "eos_completion_mode",
            ]
        }
        row.update(stats, average_length=ev["average_length"])
        row.update(
            {
                k: 100 * ev["metrics"][k]
                for k in ["state_coverage", "transition_coverage", "path_coverage"]
            }
        )
        row["vendi_score"] = ev["metrics"]["vendi_score"]
        for n in [2, 3]:
            row[f"distinct_{n}gram"] = ev["metrics"][f"distinct_{n}gram"][0]
        rows.append(row)
    grouped = defaultdict(list)
    for row in rows:
        grouped[
            (row["grammar"], row["completion_bias_eta"], row["eos_completion_mode"])
        ].append(row)
    summaries = []
    differences = []
    metrics = [
        "reject_ratio",
        "never_accepting_ratio",
        "lost_after_accepting_ratio",
        "tokens_per_valid",
        "state_coverage",
        "transition_coverage",
        "path_coverage",
        "average_length",
        "vendi_score",
        "distinct_2gram",
        "distinct_3gram",
    ]

    def summarize(group, values):
        return dict(
            **group,
            n_runs=len(values),
            mean=statistics.mean(values),
            sample_sd=statistics.stdev(values) if len(values) > 1 else None,
        )

    for (grammar, eta, mode), items in grouped.items():
        for metric in metrics:
            summaries.append(
                summarize(
                    dict(grammar=grammar, eta=eta, mode=mode, metric=metric),
                    [r[metric] for r in items],
                )
            )
    index = {
        (r["grammar"], r["completion_bias_eta"], r["eos_completion_mode"], r["seed"]): r
        for r in rows
    }
    for r in rows:
        if r["eos_completion_mode"] != "entry_count":
            continue
        for comparator in ["legacy", "constant"]:
            other = index.get(
                (r["grammar"], r["completion_bias_eta"], comparator, r["seed"])
            )
            if other:
                for metric in metrics:
                    differences.append(
                        dict(
                            grammar=r["grammar"],
                            eta=r["completion_bias_eta"],
                            seed=r["seed"],
                            comparator=comparator,
                            metric=metric,
                            delta=r[metric] - other[metric],
                        )
                    )
    pairs = defaultdict(list)
    for r in differences:
        pairs[(r["grammar"], r["eta"], r["comparator"], r["metric"])].append(r["delta"])
    paired = [
        summarize(dict(grammar=g, eta=e, comparator=c, metric=m), v)
        for (g, e, c, m), v in pairs.items()
    ]
    output = Path(manifest["run_root"]) / "aggregate"
    atomic_json(
        output / "report.json",
        dict(
            runs=rows,
            summaries=summaries,
            paired_differences=differences,
            paired_summaries=paired,
            incomplete=incomplete,
            complete=len(rows) == 45,
            limitations=(
                "Exploratory shared seeds. Coverage deltas are percentage "
                "points; rejection ratios are fractions. EOS strata have "
                "selection bias. Timing is not a controlled benchmark."
            ),
        ),
    )
    for name, data in [
        ("runs", rows),
        ("summary", summaries),
        ("paired_differences", differences),
        ("paired_summary", paired),
    ]:
        if data:
            with (output / (name + ".csv")).open("w") as f:
                writer = csv.DictWriter(f, fieldnames=list(data[0]))
                writer.writeheader()
                writer.writerows(data)
    print(f"{len(rows)}/45 complete; report: {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--backup", type=Path, required=True)
    p.add_argument("--model-path", type=Path, required=True)
    for command in ["run-stage", "worker", "aggregate"]:
        p = sub.add_parser(command)
        p.add_argument("--manifest", type=Path, required=True)
        if command == "run-stage":
            p.add_argument("--stage", choices=["smoke", "B", "C", "D"], required=True)
            p.add_argument("--reviewed-b", action="store_true")
        if command == "worker":
            p.add_argument("--task", required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        return prepare(args)
    manifest = json.loads(args.manifest.read_text())
    if args.command == "run-stage":
        return run_stage(args, manifest)
    if args.command == "worker":
        return run_task(manifest, args.task, "production")
    aggregate(args, manifest)


if __name__ == "__main__":
    main()
