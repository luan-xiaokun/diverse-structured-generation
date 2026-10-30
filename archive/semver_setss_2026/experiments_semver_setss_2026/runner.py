"""Generation and measurement runner for the SemVer SETSS extension."""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import importlib.metadata
import json
import random
import shlex
import sys
import time
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .protocol import (
    ACCEPTED_EXAMPLES,
    REJECTED_EXAMPLES,
    SEMVER_COMPILER_REGEX,
    SEMVER_DIVERSE_REGEX,
    SEMVER_NORMALIZATIONS,
    SEMVER_OUTLINES_REGEX,
    SEMVER_PROMPT,
    SEMVER_REGEX_SOURCE,
    SEMVER_REGEX_SOURCE_COMMIT,
    SEMVER_REGEX_SOURCE_FILE,
    SEMVER_STRICT_ASCII_ALTERNATIVE,
    build_semver_dfa,
    dfa_fullmatch,
    dfa_statistics,
)

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
MODEL_REVISION = "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
TOKENIZER_REVISION = MODEL_REVISION
OUTLINES_VERSION = "0.2.1"
OUTLINES_CORE_VERSION = "0.1.26"
TRANSFORMERS_VERSION = "4.49.0"
TOKENIZERS_VERSION = "0.21.0"
GAMMA = 0.5
BETA = 3.0
MAX_TOKENS = 18
REQUESTED_SAMPLES = 1000
BATCH_SIZE = 1


def classify_attempt(
    token_ids: list[int], *, eos_token_id: int, max_tokens: int, valid: bool
) -> dict[str, bool | str]:
    """Classify an attempt without changing the historical acceptance rule.

    The ICFEM runner retried until it had the requested number of regex
    full-matches.  Consequently a valid string at the token limit was retained
    even if EOS had not fit in the budget.  We preserve that rule but expose the
    termination distinction in every raw record.
    """

    eos_seen = bool(token_ids) and token_ids[-1] == eos_token_id
    invalid_completed = eos_seen and not valid
    if eos_seen:
        reason = "eos" if valid else "eos_invalid"
    elif len(token_ids) >= max_tokens:
        reason = "max_tokens_valid_fullmatch" if valid else "max_tokens_incomplete"
    else:
        reason = "other_valid_fullmatch" if valid else "other_incomplete"

    return {
        "completed": valid,
        "token_budget_failure": not valid
        and not eos_seen
        and len(token_ids) >= max_tokens,
        "invalid_completed": invalid_completed,
        "termination_reason": reason,
    }


def summarize_attempt_records(
    records: Sequence[Mapping[str, object]],
) -> dict[str, int | float]:
    """Aggregate raw attempts without deduplicating or dropping failures."""

    attempts = len(records)
    completed = sum(bool(record["completed"]) for record in records)
    failures = sum(bool(record["token_budget_failure"]) for record in records)
    invalid_completed = sum(bool(record["invalid_completed"]) for record in records)
    eos_completed = sum(
        bool(record["completed"]) and record["termination_reason"] == "eos"
        for record in records
    )
    at_budget = sum(
        bool(record["completed"])
        and record["termination_reason"] == "max_tokens_valid_fullmatch"
        for record in records
    )
    return {
        "attempts": attempts,
        "completed": completed,
        "token_budget_failures": failures,
        "invalid_completed": invalid_completed,
        "completion_rate": completed / attempts if attempts else 0.0,
        "eos_completed": eos_completed,
        "valid_fullmatch_at_budget": at_budget,
    }


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    temporary.replace(path)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _require_package_version(package: str, expected: str) -> None:
    actual = importlib.metadata.version(package)
    if actual != expected:
        raise RuntimeError(f"{package}=={expected} is required, found {actual}")


def verify_runtime_versions() -> dict[str, str]:
    expected = {
        "outlines": OUTLINES_VERSION,
        "outlines-core": OUTLINES_CORE_VERSION,
        "transformers": TRANSFORMERS_VERSION,
        "tokenizers": TOKENIZERS_VERSION,
    }
    for package, version in expected.items():
        _require_package_version(package, version)
    return {
        package: importlib.metadata.version(package)
        for package in [*expected, "torch", "numpy", "scipy"]
    }


def configure_seed(seed: int | None) -> None:
    if seed is None:
        return
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _state_digest(state: Any) -> str:
    return hashlib.sha256(bytes(state.detach().cpu().tolist())).hexdigest()


def rng_provenance(seed: int | None) -> dict[str, Any]:
    import torch

    result = {
        "configured_seed": seed,
        "torch_initial_seed": torch.initial_seed(),
        "torch_cpu_state_sha256": _state_digest(torch.random.get_rng_state()),
    }
    if torch.cuda.is_available():
        result["torch_cuda_state_sha256"] = _state_digest(torch.cuda.get_rng_state())
    return result


def load_model_and_tokenizer():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available():
        raise RuntimeError("The formal protocol requires CUDA, but CUDA is unavailable")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        revision=TOKENIZER_REVISION,
        local_files_only=True,
        padding_side="left",
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        local_files_only=True,
        device_map="cuda",
        torch_dtype=torch.float16,
    )
    model.eval()
    return model, tokenizer


def _attach_token_capture(outlines_model) -> None:
    original = outlines_model._generate_output_seq

    def capturing_generate_output_seq(*args, **kwargs):
        generated_ids = original(*args, **kwargs)
        outlines_model.last_generated_ids = generated_ids.detach().cpu()
        return generated_ids

    outlines_model.last_generated_ids = None
    outlines_model._generate_output_seq = capturing_generate_output_seq


class OriginalDiverseOutlinesAdapter:
    """Run the original DiverseGuide processor through the ICFEM Outlines adapter."""

    def __init__(self, outlines_model, logits_processor, sampler):
        self.model = outlines_model
        self.logits_processor = logits_processor
        self.sampling_params = sampler.sampling_params

    def __call__(self, prompt: str, max_tokens: int) -> str:
        from outlines.generate.api import GenerationParameters

        self.logits_processor.dfa.reset_local_state_counter()
        self.logits_processor.reset()
        generation_parameters = GenerationParameters(max_tokens, None, None)
        return self.model.generate(
            prompt,
            generation_parameters,
            self.logits_processor,
            self.sampling_params,
        )

    def update_generated_content(self, text: str) -> None:
        self.logits_processor.dfa.update_path_counter(text)


def build_generator(method: str, temperature: float, model, tokenizer):
    import outlines

    from diverse_guide.guide_rust import DiverseRegexLogitsProcessor

    sampler = outlines.samplers.multinomial(
        top_k=None,
        top_p=None,
        temperature=temperature,
    )
    outlines_model = outlines.models.Transformers(model, tokenizer)
    _attach_token_capture(outlines_model)

    if method == "outlines":
        generator = outlines.generate.regex(
            outlines_model,
            SEMVER_OUTLINES_REGEX,
            sampler,
        )
        implementation = {
            "name": "Outlines regex constrained generation",
            "package": "outlines",
            "version": OUTLINES_VERSION,
            "regex_passed": SEMVER_OUTLINES_REGEX,
            "fullmatch": (
                "interegular regex FSM accepts complete strings; `$` is unsupported"
            ),
        }
    elif method == "diverse":
        processor = DiverseRegexLogitsProcessor(
            SEMVER_DIVERSE_REGEX,
            tokenizer,
            gamma=GAMMA,
            beta=BETA,
            no_reward=False,
            no_penalty=False,
            no_range_scaling=False,
        )
        generator = OriginalDiverseOutlinesAdapter(outlines_model, processor, sampler)
        implementation = {
            "name": "original DiverseGuide",
            "class": "diverse_guide.guide_rust.DiverseRegexLogitsProcessor",
            "gamma": GAMMA,
            "beta": BETA,
            "soft_aggregation": False,
            "completion_proximity": False,
            "ablation": None,
            "regex_passed": SEMVER_DIVERSE_REGEX,
            "fullmatch": "DiverseRegexGuide wraps the regex as `(?:...)$`",
        }
    else:
        raise ValueError(f"Unknown method: {method}")

    return generator, outlines_model, sampler, implementation


def captured_token_ids(outlines_model) -> list[int]:
    generated_ids = outlines_model.last_generated_ids
    if generated_ids is None:
        raise RuntimeError("Outlines model did not expose generated token ids")
    if generated_ids.ndim == 2:
        generated_ids = generated_ids[0]
    return [int(token_id) for token_id in generated_ids.tolist()]


def _history_snapshot(generator) -> dict[str, Any]:
    dfa = generator.logits_processor.dfa
    initial_state = dfa.get_initial_state()
    token_ids, reward_counts, _ = dfa.compute_counts(initial_state)
    encoded = json.dumps([token_ids, reward_counts], separators=(",", ":")).encode()
    return {
        "initial_state": initial_state,
        "allowed_token_count": len(token_ids),
        "reward_count_sum": sum(reward_counts),
        "reward_count_max": max(reward_counts, default=0),
        "reward_counts_sha256": hashlib.sha256(encoded).hexdigest(),
    }


def _prepare_new_group(
    result_root: Path, category: str, method: str, temperature: float
) -> Path:
    slug = f"{method}-t{temperature:.1f}".replace(".", "p")
    group_dir = result_root / category / slug
    if group_dir.exists() and any(group_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite existing group: {group_dir}")
    group_dir.mkdir(parents=True, exist_ok=True)
    return group_dir


def run_generation_group(args) -> None:
    import torch

    versions = verify_runtime_versions()
    configure_seed(args.seed)
    validator = build_semver_dfa()
    group_dir = _prepare_new_group(
        args.result_root, "generation", args.method, args.temperature
    )
    started_at = utc_now()
    setup_start = time.perf_counter()
    model, tokenizer = load_model_and_tokenizer()
    generator, outlines_model, sampler, implementation = build_generator(
        args.method, args.temperature, model, tokenizer
    )
    setup_seconds = time.perf_counter() - setup_start
    starting_rng = rng_provenance(args.seed)
    history_snapshots = []
    if args.method == "diverse":
        history_snapshots.append(
            {"completed_samples": 0, **_history_snapshot(generator)}
        )

    records: list[dict[str, Any]] = []
    samples: list[str] = []
    generation_seconds = 0.0
    raw_path = group_dir / "raw_outputs.jsonl"
    timing_path = group_dir / "timing.jsonl"
    with (
        raw_path.open("x", encoding="utf-8", buffering=1) as raw_stream,
        timing_path.open("x", encoding="utf-8", buffering=1) as timing_stream,
    ):
        while len(samples) < args.requested_samples:
            if args.max_attempts is not None and len(records) >= args.max_attempts:
                raise RuntimeError(
                    f"Reached --max-attempts={args.max_attempts} with "
                    f"{len(samples)} completed samples"
                )
            before_rng = rng_provenance(args.seed)
            attempt_started = utc_now()
            start = time.perf_counter()
            text = generator(SEMVER_PROMPT, max_tokens=args.max_tokens)
            elapsed = time.perf_counter() - start
            generation_seconds += elapsed
            token_ids = captured_token_ids(outlines_model)
            valid = dfa_fullmatch(validator, text)
            classification = classify_attempt(
                token_ids,
                eos_token_id=tokenizer.eos_token_id,
                max_tokens=args.max_tokens,
                valid=valid,
            )
            if classification["completed"]:
                samples.append(text)
                sample_index: int | None = len(samples)
                if args.method == "diverse":
                    generator.update_generated_content(text)
                    if len(samples) <= 2:
                        history_snapshots.append(
                            {
                                "completed_samples": len(samples),
                                **_history_snapshot(generator),
                            }
                        )
            else:
                sample_index = None

            record = {
                "run_index": len(records) + 1,
                "sample_index": sample_index,
                "method": args.method,
                "temperature": args.temperature,
                "configured_seed": args.seed,
                "rng_before": before_rng,
                "started_at": attempt_started,
                "decoded_string": text,
                "token_ids": token_ids,
                "completion_status": (
                    "completed" if classification["completed"] else "incomplete"
                ),
                **classification,
                "valid": valid,
                "token_count": len(token_ids),
                "content_token_count": len(token_ids)
                - (1 if token_ids and token_ids[-1] == tokenizer.eos_token_id else 0),
                "character_count": len(text),
                "elapsed_seconds": elapsed,
            }
            records.append(record)
            raw_stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            timing_stream.write(
                json.dumps(
                    {
                        "run_index": record["run_index"],
                        "elapsed_seconds": elapsed,
                        "token_count": record["token_count"],
                        "termination_reason": record["termination_reason"],
                    }
                )
                + "\n"
            )

            if len(samples) % 25 == 0 or not classification["completed"]:
                print(
                    f"{args.method} T={args.temperature}: "
                    f"attempts={len(records)} completed={len(samples)}/"
                    f"{args.requested_samples} "
                    f"last={classification['termination_reason']}",
                    flush=True,
                )

    summary = summarize_attempt_records(records)
    summary.update(
        {
            "schema_version": 1,
            "grammar": "semver",
            "method": args.method,
            "temperature": args.temperature,
            "requested_samples": args.requested_samples,
            "max_tokens": args.max_tokens,
            "batch_size": BATCH_SIZE,
            "prompt": SEMVER_PROMPT,
            "model": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "tokenizer_revision": TOKENIZER_REVISION,
            "tokenizer_eos_token_id": tokenizer.eos_token_id,
            "tokenizer_pad_token_id": tokenizer.pad_token_id,
            "sampling": dataclasses.asdict(sampler.sampling_params),
            "implementation": implementation,
            "packages": versions,
            "configured_seed": args.seed,
            "rng_at_sampling_start": starting_rng,
            "rng_at_sampling_end": rng_provenance(args.seed),
            "global_history_snapshots": history_snapshots,
            "global_history_persisted_across_samples": (
                len(history_snapshots) >= 3
                and len(
                    {
                        snapshot["reward_counts_sha256"]
                        for snapshot in history_snapshots[:3]
                    }
                )
                == 3
                if args.method == "diverse"
                else None
            ),
            "setup_seconds": setup_seconds,
            "generation_seconds": generation_seconds,
            "started_at": started_at,
            "finished_at": utc_now(),
            "command": shlex.join(sys.argv),
        }
    )
    write_json(group_dir / "generation_summary.json", summary)
    write_json(
        group_dir / "samples.json",
        {
            "grammar": "semver",
            "regex": SEMVER_DIVERSE_REGEX,
            "regex_source": SEMVER_REGEX_SOURCE,
            "prompt": SEMVER_PROMPT,
            "model": MODEL_ID,
            "revision": MODEL_REVISION,
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
            "seed": args.seed,
            "samples": samples,
        },
    )
    if summary["invalid_completed"]:
        raise RuntimeError(
            f"Found {summary['invalid_completed']} completed-but-invalid outputs"
        )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    del generator, outlines_model, model
    torch.cuda.empty_cache()


def run_throughput_group(args) -> None:
    versions = verify_runtime_versions()
    configure_seed(args.seed)
    validator = build_semver_dfa()
    group_dir = _prepare_new_group(
        args.result_root, "throughput", args.method, args.temperature
    )
    setup_start = time.perf_counter()
    model, tokenizer = load_model_and_tokenizer()
    generator, outlines_model, sampler, implementation = build_generator(
        args.method, args.temperature, model, tokenizer
    )
    setup_seconds = time.perf_counter() - setup_start

    generated_token_count = 0
    total_seconds = 0.0
    records = []
    raw_path = group_dir / "timing.jsonl"
    with raw_path.open("x", encoding="utf-8", buffering=1) as stream:
        while generated_token_count < args.token_target:
            started = time.time()
            text = generator(SEMVER_PROMPT, max_tokens=args.max_tokens)
            elapsed = time.time() - started
            # Preserve the ICFEM runtime evaluator: retokenize decoded output and
            # count those ids, rather than counting captured generation ids.
            counted_tokens = len(tokenizer.encode(text))
            generated_token_count += counted_tokens
            total_seconds += elapsed
            token_ids = captured_token_ids(outlines_model)
            record = {
                "run_index": len(records) + 1,
                "method": args.method,
                "temperature": args.temperature,
                "decoded_string": text,
                "token_ids": token_ids,
                "actual_generated_token_count": len(token_ids),
                "historical_retokenized_count": counted_tokens,
                "valid": dfa_fullmatch(validator, text),
                "elapsed_seconds": elapsed,
                "cumulative_historical_tokens": generated_token_count,
            }
            records.append(record)
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            if len(records) % 25 == 0:
                print(
                    f"{args.method} throughput: {generated_token_count}/"
                    f"{args.token_target} historical tokens",
                    flush=True,
                )

    result = {
        "schema_version": 1,
        "grammar": "semver",
        "method": args.method,
        "temperature": args.temperature,
        "token_target": args.token_target,
        "generated_tokens": generated_token_count,
        "seconds": total_seconds,
        "tokens_per_second": generated_token_count / total_seconds,
        "calls": len(records),
        "warmup_calls": 0,
        "timing_interval": "each generator(prompt, max_tokens) call only",
        "token_counting": "len(tokenizer.encode(decoded_output))",
        "global_path_counter_updates": 0,
        "max_tokens": args.max_tokens,
        "batch_size": BATCH_SIZE,
        "sampling": dataclasses.asdict(sampler.sampling_params),
        "implementation": implementation,
        "packages": versions,
        "configured_seed": args.seed,
        "setup_seconds_excluded": setup_seconds,
        "command": shlex.join(sys.argv),
        "finished_at": utc_now(),
    }
    write_json(group_dir / "throughput.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


def run_conformance(args) -> None:
    dfa = build_semver_dfa()
    cases = []
    for expected, values in [(True, ACCEPTED_EXAMPLES), (False, REJECTED_EXAMPLES)]:
        for value in values:
            actual = dfa_fullmatch(dfa, value)
            cases.append(
                {
                    "value": value,
                    "expected": expected,
                    "actual": actual,
                    "passed": actual == expected,
                }
            )
    result = {
        "passed": all(case["passed"] for case in cases),
        "case_count": len(cases),
        "cases": cases,
        "regex_source": SEMVER_REGEX_SOURCE,
        "regex_outlines": SEMVER_OUTLINES_REGEX,
        "regex_diverse": SEMVER_DIVERSE_REGEX,
        "regex_compiler": SEMVER_COMPILER_REGEX,
        "normalizations": SEMVER_NORMALIZATIONS,
        "dfa": dfa_statistics(dfa),
        "command": shlex.join(sys.argv),
        "finished_at": utc_now(),
    }
    output = args.result_root / "regex_conformance.json"
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    write_json(output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result["passed"]:
        raise SystemExit(1)


def run_init(args) -> None:
    root = args.result_root
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f"Refusing to initialize non-empty result root: {root}")
    root.mkdir(parents=True, exist_ok=True)
    config = {
        "schema_version": 1,
        "grammar": "semver",
        "prompt": SEMVER_PROMPT,
        "regex_source": SEMVER_REGEX_SOURCE,
        "regex_source_commit": SEMVER_REGEX_SOURCE_COMMIT,
        "regex_source_file": SEMVER_REGEX_SOURCE_FILE,
        "regex_outlines": SEMVER_OUTLINES_REGEX,
        "regex_diverse": SEMVER_DIVERSE_REGEX,
        "regex_compiler": SEMVER_COMPILER_REGEX,
        "strict_ascii_alternative": SEMVER_STRICT_ASCII_ALTERNATIVE,
        "normalizations": SEMVER_NORMALIZATIONS,
        "dfa": dfa_statistics(build_semver_dfa()),
        "settings": {
            "model": MODEL_ID,
            "revision": MODEL_REVISION,
            "tokenizer_revision": TOKENIZER_REVISION,
            "beta": BETA,
            "gamma": GAMMA,
            "max_tokens": args.max_tokens,
            "requested_runs_parameter": args.requested_samples,
            "requested_runs_semantics": "retry until this many DFA-fullmatch samples",
            "batch_size": BATCH_SIZE,
            "temperatures": [1.0, 1.5],
            "methods": ["outlines", "diverse"],
            "seed": args.seed,
        },
        "created_at": utc_now(),
        "command": shlex.join(sys.argv),
    }
    write_json(root / "config.json", config)
    with (root / "semver.regex").open("x", encoding="utf-8") as stream:
        stream.write(SEMVER_COMPILER_REGEX + "\n")
    write_json(
        root / "regex_provenance.json",
        {
            key: config[key]
            for key in [
                "regex_source",
                "regex_source_commit",
                "regex_source_file",
                "regex_outlines",
                "regex_diverse",
                "regex_compiler",
                "strict_ascii_alternative",
                "normalizations",
                "dfa",
            ]
        },
    )
    print(f"Initialized {root}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_result_root(command_parser):
        command_parser.add_argument("--result-root", type=Path, required=True)

    init_parser = subparsers.add_parser("init")
    add_result_root(init_parser)
    init_parser.add_argument("--requested-samples", type=int, default=REQUESTED_SAMPLES)
    init_parser.add_argument("--max-tokens", type=int, default=MAX_TOKENS)
    init_parser.add_argument("--seed", type=int, default=None)
    init_parser.set_defaults(func=run_init)

    conformance_parser = subparsers.add_parser("conformance")
    add_result_root(conformance_parser)
    conformance_parser.set_defaults(func=run_conformance)

    generate_parser = subparsers.add_parser("generate")
    add_result_root(generate_parser)
    generate_parser.add_argument(
        "--method", choices=["outlines", "diverse"], required=True
    )
    generate_parser.add_argument(
        "--temperature", type=float, choices=[1.0, 1.5], required=True
    )
    generate_parser.add_argument(
        "--requested-samples", type=int, default=REQUESTED_SAMPLES
    )
    generate_parser.add_argument("--max-tokens", type=int, default=MAX_TOKENS)
    generate_parser.add_argument("--max-attempts", type=int, default=None)
    generate_parser.add_argument("--seed", type=int, default=None)
    generate_parser.set_defaults(func=run_generation_group)

    throughput_parser = subparsers.add_parser("throughput")
    add_result_root(throughput_parser)
    throughput_parser.add_argument(
        "--method", choices=["outlines", "diverse"], required=True
    )
    throughput_parser.add_argument("--temperature", type=float, default=1.0)
    throughput_parser.add_argument("--token-target", type=int, default=1000)
    throughput_parser.add_argument("--max-tokens", type=int, default=MAX_TOKENS)
    throughput_parser.add_argument("--seed", type=int, default=None)
    throughput_parser.set_defaults(func=run_throughput_group)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
