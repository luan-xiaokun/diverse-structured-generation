"""Durable single-run Group 1 worker. Incomplete attempts restart from seed."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import shutil
import signal
import time
import traceback
import uuid
from pathlib import Path


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    with temp.open("x") as f:
        json.dump(value, f, ensure_ascii=False, indent=2)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(temp, path)
    fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def atomic_copy(source, target):
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = target.with_name(target.name + "." + uuid.uuid4().hex + ".tmp")
    shutil.copyfile(source, temp)
    with temp.open("rb") as f:
        os.fsync(f.fileno())
    os.replace(temp, target)
    if file_hash(source) != file_hash(target):
        raise RuntimeError(f"backup checksum mismatch: {target}")


def completed_is_valid(status, task, result):
    if (
        status.get("status") != "complete"
        or status.get("fingerprint") != task["fingerprint"]
    ):
        return False
    if not result.is_file() or file_hash(result) != status.get("result_sha256"):
        return False
    data = json.loads(result.read_text())
    return (
        data.get("configuration_fingerprint") == task["fingerprint"]
        and len(data.get("samples", [])) == task["target_valid_samples"]
        and data.get("eos_completion_mode", "legacy")
        == task.get("eos_completion_mode", "legacy")
        and all(
            Path(p).is_file() and file_hash(p) == h
            for p, h in status.get("artifacts_sha256", {}).items()
        )
        and bool(status.get("artifacts_sha256"))
    )


def verify_sources(manifest):
    if Path(manifest["source_root"]).resolve() != Path(__file__).resolve().parents[1]:
        raise RuntimeError("manifest belongs to another source checkout")
    for path, expected in manifest["source_sha256"].items():
        if file_hash(Path(manifest["source_root"]) / path) != expected:
            raise RuntimeError(f"frozen source changed: {path}")
    if "environment" in manifest and environment_snapshot() != manifest["environment"]:
        raise RuntimeError("frozen runtime environment changed")


def environment_snapshot():
    import importlib.metadata
    import platform

    import regex_dfa_guide.regex_dfa_guide_rs as native

    return dict(
        python=platform.python_version(),
        platform=platform.platform(),
        native_module=str(Path(native.__file__).resolve()),
        native_sha256=file_hash(native.__file__),
        packages={
            name: importlib.metadata.version(name)
            for name in [
                "torch",
                "transformers",
                "outlines",
                "regex-dfa-guide",
                "numpy",
                "scipy",
            ]
        },
    )


def run_task(manifest, task_id, phase):
    import importlib.metadata

    import psutil
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

    from diverse_guide import DiverseGuide
    from scripts.generate_re import make_outlines_regex_generator, set_generation_seed
    from scripts.generation_validation import (
        VALIDATION_REVISION,
        compile_generation_regex,
    )

    task = next(t for t in manifest["tasks"] if t["id"] == task_id)
    task = dict(task)
    if task["fingerprint"] != digest(
        {k: v for k, v in task.items() if k != "fingerprint"}
    ):
        raise ValueError("task configuration fingerprint mismatch")
    if (
        task.get("eos_completion_mode", "legacy") != "legacy"
        and task["method"] != "diverse_guide_plus"
    ):
        raise ValueError("EOS experiment modes are restricted to DiverseGuide+")
    if phase == "pilot":
        task["target_valid_samples"] = manifest["pilot_samples"]
        task["phase"] = "pilot"
        task["fingerprint"] = digest(
            {k: v for k, v in task.items() if k != "fingerprint"}
        )
    verify_sources(manifest)
    root = Path(manifest["run_root"]) / phase
    backup = Path(manifest["backup_root"]) / phase
    task_root = root / "tasks" / task_id
    task_root.mkdir(parents=True, exist_ok=True)
    lock = (task_root / "writer.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    result = root / "data" / (task_id + ".json")
    state_path = task_root / "status.json"
    if state_path.exists() and completed_is_valid(
        json.loads(state_path.read_text()), task, result
    ):
        print("Verified complete; skipping", task_id, flush=True)
        return
    if result.exists():
        raise RuntimeError(
            "existing result did not validate; preserve and investigate before retry"
        )
    attempt = task_root / (
        "attempt-" + time.strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:8]
    )
    attempt.mkdir()
    start = time.monotonic()
    samples, rejected, attempts = [], 0, 0
    state = {
        "task_id": task_id,
        "model": task["model"],
        "grammar": task["grammar"],
        "method": task["method"],
        "seed": task["seed"],
        "phase": phase,
        "target": task["target_valid_samples"],
        "fingerprint": task["fingerprint"],
        "status": "loading",
        "attempt_dir": str(attempt),
        "pid": os.getpid(),
        "gpu": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "started_at": time.time(),
        "accepted": 0,
        "rejected": 0,
        "attempts": 0,
        "updated_at": time.time(),
    }
    atomic_json(attempt / "configuration.json", task)
    atomic_json(state_path, state)
    journal = None
    generation_start = None
    generation_duration = None
    last_save = last_backup = 0.0
    latest_tokens = []
    probe = None

    def checkpoint(status=None, force_backup=False):
        nonlocal last_save, last_backup
        if status:
            state["status"] = status
        state.update(
            accepted=len(samples),
            rejected=rejected,
            attempts=attempts,
            updated_at=time.time(),
            elapsed_seconds=time.monotonic() - start,
            generation_seconds=(
                generation_duration
                if generation_duration is not None
                else (time.monotonic() - generation_start if generation_start else 0)
            ),
            rss_mib=psutil.Process().memory_info().rss / 2**20,
        )
        if torch.cuda.is_initialized():
            state["peak_cuda_allocated_mib"] = torch.cuda.max_memory_allocated() / 2**20
            state["peak_cuda_reserved_mib"] = torch.cuda.max_memory_reserved() / 2**20
        atomic_json(state_path, state)
        atomic_json(attempt / "status.json", state)
        last_save = time.monotonic()
        if force_backup or last_save - last_backup > 30:
            for path in attempt.iterdir():
                if path.is_file():
                    atomic_copy(path, backup / path.relative_to(root))
            atomic_copy(state_path, backup / state_path.relative_to(root))
            last_backup = last_save

    def interrupted(signum, frame):
        raise InterruptedError(f"interrupted by signal {signum}")

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGALRM, interrupted)
    signal.alarm(manifest["task_timeout_seconds"])
    try:
        if psutil.virtual_memory().available < manifest.get("min_ram_gib", 8) * 2**30:
            raise RuntimeError("insufficient available host RAM")
        for directory in (root, backup):
            directory.mkdir(parents=True, exist_ok=True)
            if (
                shutil.disk_usage(directory).free
                < manifest.get("min_disk_gib", 10) * 2**30
            ):
                raise RuntimeError("insufficient disk headroom")
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError("worker requires exactly one visible CUDA GPU")
        free, total = torch.cuda.mem_get_info()
        if free < max(manifest.get("min_vram_gib", 8) * 2**30, total * 0.2):
            raise RuntimeError("insufficient GPU headroom; wait or reduce concurrency")
        torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "2")))
        torch.set_num_interop_threads(1)
        set_generation_seed(task["seed"])
        model_path = Path(task["model_path"])
        for filename, expected in task["model_config_sha256"].items():
            if file_hash(model_path / filename) != expected:
                raise RuntimeError(f"model metadata changed: {filename}")
        tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        model = (
            AutoModelForCausalLM.from_pretrained(
                model_path, local_files_only=True, dtype=torch.float16
            )
            .to("cuda")
            .eval()
        )
        config = GenerationConfig.from_dict(task["generation_config"])
        model.generation_config = config
        state.update(
            device_name=torch.cuda.get_device_name(),
            versions={
                n: importlib.metadata.version(n)
                for n in ["torch", "transformers", "outlines", "regex-dfa-guide"]
            },
        )
        import regex_dfa_guide.regex_dfa_guide_rs as native

        state["native_module"] = str(Path(native.__file__).resolve())
        state["native_sha256"] = file_hash(native.__file__)
        original_generate = model.generate

        def capture_generate(*args, _generate=original_generate, **kwargs):
            nonlocal latest_tokens
            output = _generate(*args, **kwargs)
            input_ids = kwargs.get("input_ids", args[0] if args else None)
            prompt_length = input_ids.shape[-1]
            latest_tokens = output[:, prompt_length:].detach().cpu().tolist()
            if probe is not None and latest_tokens[0]:
                probe.finish(latest_tokens[0][-1])
            return output

        model.generate = capture_generate
        if task["method"] == "baseline_outlines":
            generator = make_outlines_regex_generator(
                model, tokenizer, task["regex"], generation_config=config
            )
        else:
            generator = DiverseGuide(
                model,
                tokenizer,
                task["regex"],
                gamma=task["gamma"],
                completion_bias_eta=task["completion_bias_eta"],
                path_frequency_estimator=task["path_frequency_estimator"],
                eos_completion_mode=task.get("eos_completion_mode", "legacy"),
                generation_config=config,
            )
        pattern = compile_generation_regex(task["regex"])
        generation_start = time.monotonic()
        checkpoint("generating", True)
        journal = (attempt / "attempts.jsonl").open("x", buffering=1)

        def emit(event):
            journal.write(json.dumps(event, ensure_ascii=False) + "\n")
            journal.flush()
            os.fsync(journal.fileno())

        if task.get("step_probes", False):
            from diverse_guide.eos_diagnostics import StepProbeProcessor

            probe = StepProbeProcessor(
                generator.logits_processor,
                lambda row: emit(dict(event="step_probe", attempt=attempts, **row)),
            )
            generator.logits_processor = probe

        while len(samples) < task["target_valid_samples"]:
            if attempts >= manifest["max_attempts"]:
                raise RuntimeError("attempt limit reached")
            attempts += 1
            emit({"event": "start", "attempt": attempts, "time": time.time()})
            latest_tokens = []
            with torch.inference_mode():
                texts = generator.generate_batch(
                    task["prompt"], n=1, max_tokens=task["max_tokens"]
                )
            if len(texts) != 1:
                raise RuntimeError("batch size invariant violated")
            text = texts[0]
            emit(
                {
                    "event": "generated",
                    "attempt": attempts,
                    "time": time.time(),
                    "text": text,
                    "token_ids": latest_tokens,
                }
            )
            valid = bool(pattern.match(text))
            from diverse_guide.eos_diagnostics import trajectory

            diagnostics = (
                trajectory(
                    generator.logits_processor.dfa,
                    latest_tokens[0],
                    tokenizer.eos_token_id,
                )
                if task["method"] != "baseline_outlines"
                else {}
            )
            if valid:
                samples.append(text)
            else:
                rejected += 1
            emit(
                {
                    "event": "finish",
                    "attempt": attempts,
                    "time": time.time(),
                    "valid": valid,
                    "accepted_index": len(samples) if valid else None,
                    "text": text,
                    "token_ids": latest_tokens,
                    "accepted": len(samples),
                    "rejected": rejected,
                    "diagnostics": diagnostics,
                }
            )
            if (
                time.monotonic() - last_save >= 5
                or len(samples) == task["target_valid_samples"]
            ):
                checkpoint()
        generation_seconds = time.monotonic() - generation_start
        generation_duration = generation_seconds
        journal.close()
        journal = None
        payload = {
            k: task[k]
            for k in [
                "model",
                "grammar",
                "regex",
                "prompt",
                "max_tokens",
                "seed",
                "gamma",
                "completion_bias_eta",
                "path_frequency_estimator",
            ]
        }
        payload.update(
            samples=samples,
            accepted_samples=len(samples),
            rejected_samples=rejected,
            attempted_batches=attempts,
            elapsed_seconds=generation_seconds,
            temperature=config.temperature,
            top_k=config.top_k,
            top_p=config.top_p,
            repetition_penalty=config.repetition_penalty,
            device="cuda",
            model_parameter_device="cuda:0",
            baseline_backend=(
                "outlines" if task["method"] == "baseline_outlines" else "internal"
            ),
            configuration_fingerprint=task["fingerprint"],
            validation_revision=VALIDATION_REVISION,
            execution_source_fingerprint=digest(manifest["source_sha256"]),
            source_commit=manifest["source_commit"],
            model_revision=task["model_revision"],
            eos_completion_mode=task.get("eos_completion_mode", "legacy"),
            generation_config=config.to_dict(),
            phase=phase,
            attempt_dir=str(attempt),
        )
        atomic_json(result, payload)
        state.update(
            result_path=str(result),
            result_sha256=file_hash(result),
            generation_complete=True,
        )
        atomic_copy(result, backup / result.relative_to(root))
        checkpoint("evaluating", True)
        # Release CUDA before CPU evaluation; queue slot stays occupied until verified.
        model.generate = original_generate
        del capture_generate, original_generate, generator, model, tokenizer
        import gc

        gc.collect()
        torch.cuda.empty_cache()
        from regex_dfa_guide import DiverseGuideDFA
        from scripts.evaluate_main_matrix import MatrixCell, build_cell_result

        record, trend = build_cell_result(
            cell=MatrixCell(task["method"], task["model"], task["grammar"]),
            path=result,
            data=payload,
            dfa=DiverseGuideDFA("(?:" + task["regex"] + ")$", 2**32 - 1, {}),
            d=5,
            s=1,
            step=50,
            max_samples=None,
            expected_samples=task["target_valid_samples"],
        )
        if record["status"] != "complete":
            raise RuntimeError("evaluation did not validate all samples")
        atomic_json(attempt / "evaluation.json", {"record": record, "trend": trend})
        state["evaluation_path"] = str(attempt / "evaluation.json")
        state["artifacts_sha256"] = {
            str(p): file_hash(p)
            for p in [
                attempt / "evaluation.json",
                attempt / "attempts.jsonl",
                attempt / "configuration.json",
            ]
        }
        checkpoint("complete", True)
        print(
            f"COMPLETE {task_id}: {len(samples)} accepted / {attempts} attempts",
            flush=True,
        )
    except BaseException as exc:
        if journal:
            journal.close()
        atomic_json(
            attempt / "partial.json",
            {
                "samples": samples,
                "accepted": len(samples),
                "rejected": rejected,
                "attempts": attempts,
                "fingerprint": task["fingerprint"],
            },
        )
        state.update(
            error=f"{type(exc).__name__}: {exc}",
            traceback=traceback.format_exc(),
            restart_policy=(
                "preserve attempt; restart full task with original seed, never append"
            ),
        )
        try:
            checkpoint("incomplete", True)
        except Exception:
            atomic_json(state_path, state)
        raise
    finally:
        signal.alarm(0)
        fcntl.flock(lock, fcntl.LOCK_UN)
        lock.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--phase", choices=["pilot", "production"], required=True)
    args = parser.parse_args()
    run_task(json.loads(args.manifest.read_text()), args.task, args.phase)


if __name__ == "__main__":
    main()
