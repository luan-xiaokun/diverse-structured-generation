"""Export a source bundle and checksummed historical Group 1 evidence.

Uses only the Python standard library. This prepares deployment, not a claim
that historical configuration or the five-repeat experiment is validated.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import re
import shutil
import subprocess
import tarfile
from pathlib import Path

MODELS = (
    "Qwen/Qwen2.5-0.5B-Instruct",
    "Qwen/Qwen2.5-1.5B-Instruct",
    "microsoft/Phi-4-mini-instruct",
)
GRAMMARS = ("email", "css-color", "json", "no-bomb", "semver", "api-route")
METHODS = ("baseline_outlines", "diverse_guide", "diverse_guide_plus")


def git(repo: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True)


def sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def copy_tree(source: Path, target: Path) -> None:
    if not source.exists():
        raise FileNotFoundError(source)
    if source.is_dir():
        shutil.copytree(
            source, target, ignore=shutil.ignore_patterns("__pycache__", "*.pyc")
        )
    else:
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)


def export(args: argparse.Namespace) -> None:
    repo = Path(__file__).resolve().parents[1]
    source = args.legacy_root.resolve()
    output = args.output.resolve()
    if git(repo, "status", "--porcelain").strip():
        raise RuntimeError("Commit source changes before exporting a bundle.")
    if git(repo, "branch", "--show-current").strip() != "fac-revision":
        raise RuntimeError("Export from the fac-revision branch.")
    # A fresh directory prevents accidental replacement of previous releases.
    output.mkdir(parents=True, exist_ok=False)
    imported = output / "imported-seed42"
    imported.mkdir()
    evidence = output / "evidence"
    evidence.mkdir()
    commit = git(repo, "rev-parse", "HEAD").strip()
    git(
        repo,
        "bundle",
        "create",
        str(output / "source.bundle"),
        "refs/heads/fac-revision",
    )
    git(repo, "bundle", "verify", str(output / "source.bundle"))
    copy_tree(args.task, output / "experiment-group-1-task.md")
    copy_tree(repo / "docs/group1-server-handoff.md", output / "README.md")

    required = [
        "data/extension/main_matrix_500",
        "results/extension/main_matrix_500",
        "results/extension/case_study_coverage_curves",
        "logs/extension/main_matrix_500",
    ]
    optional = [
        "results/extension/case_study_coverage_curves_main_matrix_500",
        "results/extension/case_study_coverage_curves_main_matrix_500_latest_release",
        "results/extension/case_study_main_matrix_500",
        "results/extension/case_study_main_matrix_500_latest_libs",
        "results/extension/case_study_semver_main_matrix_500",
    ]
    copied = []
    for relative in required + optional:
        if relative in required or (source / relative).exists():
            copy_tree(source / relative, imported / relative)
            copied.append(relative)
    # Preserve current fetched libraries as evidence, not as approved targets.
    for name in ("email_validator", "webcolors", "semver"):
        library = source / "experiments/case_study" / name
        if library.exists():
            copy_tree(library, evidence / "library-snapshot-candidates" / name)
    (evidence / "legacy-working-tree.diff").write_text(git(source, "diff", "HEAD"))
    (evidence / "legacy-status.txt").write_text(git(source, "status", "--short"))
    (evidence / "snapshot-vs-historical-source.diff").write_text(
        git(
            repo,
            "diff",
            "cf7189e",
            "671ec48",
            "--",
            "src",
            "regex_dfa_guide",
            "scripts",
            "experiments",
        )
    )
    (evidence / "historical-fetch-packages.sh").write_text(
        git(repo, "show", "cf7189e:experiments/case_study/fetch_packages.sh")
    )

    evaluation = json.loads(
        (
            imported / "results/extension/main_matrix_500/evaluation/metrics.json"
        ).read_text()
    )
    evaluated = {
        (r["model"], r["grammar"], r["method"]): r for r in evaluation["records"]
    }
    if len(evaluated) != 54 or len(evaluation["records"]) != 54 or evaluation["errors"]:
        raise ValueError("Historical evaluation matrix is incomplete or duplicated")
    tasks = []
    for model, grammar, method in itertools.product(MODELS, GRAMMARS, METHODS):
        relative = (
            Path("data/extension/main_matrix_500")
            / method
            / model.split("/")[-1].lower()
            / f"{grammar}.json"
        )
        path = imported / relative
        data = json.loads(path.read_text())
        samples = data["samples"]
        record = evaluated[(model, grammar, method)]
        if len(samples) != 500 or data["seed"] != 42:
            raise ValueError(f"Wrong sample count or seed: {relative}")
        if data["model"] != model or data["grammar"] != grammar:
            raise ValueError(f"Wrong model/grammar: {relative}")
        pattern = re.compile("^(?:" + data["regex"] + ")$")
        if not all(pattern.match(sample) for sample in samples):
            raise ValueError(f"Invalid saved samples: {relative}")
        if (
            record["status"] != "complete"
            or record["saved_sample_count"] != 500
            or record["valid_sample_count"] != 500
        ):
            raise ValueError(f"Historical evaluation mismatch: {relative}")
        for key, value in record["generation_parameters"].items():
            if data.get(key) != value:
                raise ValueError(f"Historical parameter mismatch: {relative}: {key}")
        saved = {key: value for key, value in data.items() if key != "samples"}
        for seed in range(42, 47):
            tasks.append(
                {
                    "id": f"{model}/{grammar}/{method}/seed-{seed}",
                    "model": model,
                    "grammar": grammar,
                    "method": method,
                    "seed": seed,
                    "target_valid_samples": 500,
                    "status": (
                        "imported_pending_compatibility"
                        if seed == 42 and model.startswith("microsoft/")
                        else "pending_preflight"
                    ),
                    "temperature": 1.0,
                    "rerun_historical": seed == 42 and model.startswith("Qwen/"),
                    "historical_source": str(Path("imported-seed42") / relative),
                    "historical_sha256": sha(path),
                    "historical_metadata": saved,
                    "regex_sha256": hashlib.sha256(data["regex"].encode()).hexdigest(),
                    "prompt_sha256": hashlib.sha256(
                        data["prompt"].encode()
                    ).hexdigest(),
                    "effective_configuration_status": "unresolved",
                }
            )

    models = []
    for model in MODELS:
        cached = args.hf_cache / ("models--" + model.replace("/", "--"))
        revision = (cached / "refs/main").read_text().strip()
        snapshot = cached / "snapshots" / revision
        files = []
        for path in sorted(snapshot.rglob("*")):
            if path.is_file() and path.suffix in {".json", ".txt", ".jinja"}:
                target = (
                    evidence
                    / "models"
                    / model.replace("/", "--")
                    / revision
                    / path.relative_to(snapshot)
                )
                copy_tree(path, target)
                files.append(
                    {"path": str(target.relative_to(output)), "sha256": sha(target)}
                )
        models.append(
            {
                "model": model,
                "candidate_revision": revision,
                "provenance": (
                    "local HF refs/main at export; "
                    "historical runs did not record revisions"
                ),
                "historical_revision_confirmed": False,
                "files": files,
                "generation_config": json.loads(
                    (snapshot / "generation_config.json").read_text()
                ),
            }
        )

    case_studies = []
    for relative in copied:
        curve = imported / relative / "coverage_curves.json"
        if not curve.exists():
            continue
        payload = json.loads(curve.read_text())
        rows = payload["runs"] if isinstance(payload, dict) else payload
        denominators = sorted(
            {(r["case_study"], r["num_statements"], r["num_branches"]) for r in rows}
        )
        case_studies.append(
            {
                "path": relative,
                "rows": len(rows),
                "denominators": denominators,
                "budgets": sorted({r["sample_count"] for r in rows}),
            }
        )
    write_json(
        output / "manifest.json",
        {
            "schema_version": 1,
            "purpose": "group1 deployment and historical evidence transfer",
            "source_commit": commit,
            "branch": "fac-revision",
            "source_clean": True,
            "legacy_head": git(source, "rev-parse", "HEAD").strip(),
            "production_ready": False,
            "seeds": [42, 43, 44, 45, 46],
            "imported_runs": 54,
            "pending_runs": 252,
            "reuse_candidates": 18,
            "user_decision": {
                "date": "2026-09-16",
                "temperature": 1.0,
                "qwen_seed42": "rerun all 36 configurations",
                "phi_seed42": "reuse only after compatibility confirmation",
            },
            "imported_samples": 27000,
            "historical_attempt_traces": (
                "not recorded by original generate_re.py; aggregate counts only"
            ),
            "model_weights_included": False,
            "models": models,
            "case_study_candidates": case_studies,
            "blockers": [
                "Freeze remaining effective sampling parameters explicitly; "
                "temperature=1.0 and Qwen seed42 reruns are user-approved.",
                "Historical model/tokenizer revisions absent from run JSON; "
                "cached revision candidates require confirmation.",
                "Library coverage denominators differ between preserved result sets; "
                "select manuscript source and matching library snapshots.",
                "671ec48 modifies steering code relative to cf7189e; establish "
                "historical implementation equivalence before combining seeds.",
                "Original runner lacks atomic writes, attempt traces, safe restart "
                "and task claiming; develop dedicated runner before production.",
                "Server pilot, environment capture and frozen-evaluator recheck "
                "still required.",
            ],
            "tasks": tasks,
        },
    )
    files = sorted(path for path in output.rglob("*") if path.is_file())
    # Compare each copied historical file with its source before sealing it.
    for path in imported.rglob("*"):
        if path.is_file() and sha(path) != sha(source / path.relative_to(imported)):
            raise RuntimeError(f"Source changed during export: {path}")
    (output / "SHA256SUMS").write_text(
        "".join(f"{sha(path)}  {path.relative_to(output)}\n" for path in files)
    )
    for path in imported.rglob("*"):
        path.chmod(0o555 if path.is_dir() else 0o444)
    imported.chmod(0o555)
    archive = output.with_name(output.name + ".tar.gz")
    with tarfile.open(archive, "x:gz") as tar:
        tar.add(output, arcname=output.name)
    archive.with_name(archive.name + ".sha256").write_text(
        f"{sha(archive)}  {archive.name}\n"
    )
    print(
        json.dumps(
            {
                "archive": str(archive),
                "bytes": archive.stat().st_size,
                "commit": commit,
                "files": len(files),
                "tasks": len(tasks),
            },
            indent=2,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--legacy-root", type=Path, required=True)
    parser.add_argument("--task", type=Path, required=True)
    parser.add_argument("--hf-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    export(parser.parse_args())


if __name__ == "__main__":
    main()
