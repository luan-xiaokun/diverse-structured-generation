#!/usr/bin/env python3
"""Resume regex-constrained generation from an existing JSON artifact.

This script is intentionally separate from ``generate_re.py`` so the original
entry point remains unchanged. It restores the diversity counter state by
replaying accepted samples from the existing artifact, then appends new accepted
samples until the requested total is reached.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from time import perf_counter
from typing import Any

import torch
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from diverse_guide import DiverseGuide, baseline_regex
from diverse_guide.evaluation.paths import get_data_dir_path

from generate_re import (
    GRAMMAR_PROMPT,
    GRAMMAR_REGEX,
    GeneratedData,
    get_model_parameter_device,
    make_outlines_regex_generator,
    set_generation_seed,
)


CHECKED_FIELDS = [
    "grammar",
    "regex",
    "model",
    "max_tokens",
    "top_k",
    "top_p",
    "temperature",
    "gamma",
    "completion_bias_eta",
    "path_frequency_estimator",
    "baseline_backend",
    "seed",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Resume text generation from a saved regex artifact."
    )
    parser.add_argument("grammar", type=str, help="The grammar to generate text for.")
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2.5-1.5B-Instruct",
        help="The model to use for generation.",
    )
    parser.add_argument(
        "-n",
        type=int,
        required=True,
        help="Target total number of accepted samples after resuming.",
    )
    parser.add_argument("--max-tokens", type=int, default=18)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--gamma", type=float, default=0.15)
    parser.add_argument("--completion-bias-eta", type=float, default=0.15)
    parser.add_argument(
        "--path-frequency-estimator",
        choices=["min", "harmonic"],
        default="harmonic",
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--baseline", "-b", action="store_true")
    parser.add_argument(
        "--baseline-backend",
        choices=["internal", "outlines"],
        default="internal",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Existing JSON artifact to resume from. Defaults to --output if it exists.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Output JSON path or directory. If omitted, uses generate_re.py's "
            "default data path for the current parameters."
        ),
    )
    parser.add_argument(
        "--allow-parameter-mismatch",
        action="store_true",
        help="Do not fail if existing artifact parameters differ from CLI arguments.",
    )
    parser.add_argument(
        "--stdout-only",
        action="store_true",
        help="Print generated samples only; cannot be used when resuming from disk.",
    )
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument(
        "--ablation-component",
        choices=["reward", "penalty", "range_scaling"],
        default=None,
    )
    args = parser.parse_args()
    if args.n <= 0:
        parser.error("-n must be positive")
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    if args.gamma < 0.0:
        parser.error("--gamma must be non-negative")
    if args.completion_bias_eta < 0.0:
        parser.error("--completion-bias-eta must be non-negative")
    if args.baseline_backend == "outlines" and not args.baseline:
        parser.error("--baseline-backend outlines requires --baseline")
    if args.stdout_only and (args.input is not None or args.output is not None):
        parser.error("--stdout-only cannot be used with --input/--output")
    return args


def output_path_for_args(args: argparse.Namespace) -> Path | None:
    if args.stdout_only:
        return None
    if args.output is not None:
        if args.output.suffix.lower() == ".json":
            return args.output
        return args.output / f"{args.grammar}.json"
    return get_data_dir_path(args) / f"{args.grammar}.json"


def load_existing(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.exists():
        return None
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def write_json_atomic(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)
        handle.write("\n")
    tmp_path.replace(path)


def values_equal(expected: Any, actual: Any) -> bool:
    if isinstance(expected, float) or isinstance(actual, float):
        try:
            return math.isclose(float(expected), float(actual), rel_tol=1e-9, abs_tol=1e-12)
        except (TypeError, ValueError):
            return False
    return expected == actual


def current_parameters(args: argparse.Namespace, regex: str) -> dict[str, Any]:
    return {
        "grammar": args.grammar,
        "regex": regex,
        "model": args.model,
        "max_tokens": args.max_tokens,
        "top_k": args.top_k,
        "top_p": args.top_p,
        "temperature": args.temperature,
        "gamma": args.gamma,
        "completion_bias_eta": args.completion_bias_eta,
        "path_frequency_estimator": args.path_frequency_estimator,
        "baseline_backend": args.baseline_backend if args.baseline else "internal",
        "seed": args.seed,
    }


def validate_existing_artifact(
    existing: dict[str, Any],
    args: argparse.Namespace,
    regex: str,
) -> None:
    pattern = re.compile(f"^(?:{regex})$")
    samples = existing.get("samples")
    if not isinstance(samples, list):
        raise ValueError("existing artifact field 'samples' must be a list")
    invalid = [sample for sample in samples if not isinstance(sample, str) or not pattern.match(sample)]
    if invalid:
        raise ValueError(
            f"existing artifact contains {len(invalid)} invalid/non-string samples; "
            f"first invalid={invalid[0]!r}"
        )

    if args.allow_parameter_mismatch:
        return

    expected = current_parameters(args, regex)
    mismatches = []
    for field in CHECKED_FIELDS:
        if field in existing and not values_equal(expected[field], existing[field]):
            mismatches.append((field, existing[field], expected[field]))
    if mismatches:
        details = "; ".join(
            f"{field}: existing={old!r}, cli={new!r}"
            for field, old, new in mismatches
        )
        raise ValueError(
            "existing artifact parameters differ from CLI arguments. "
            f"Use --allow-parameter-mismatch to override. Mismatches: {details}"
        )


def build_generation_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    generation_kwargs: dict[str, Any] = {}
    if args.top_k is not None:
        generation_kwargs["top_k"] = args.top_k
    if args.top_p is not None:
        generation_kwargs["top_p"] = args.top_p
    if args.temperature is not None:
        generation_kwargs["temperature"] = args.temperature
    return generation_kwargs


def build_generator(args, model, tokenizer, regex: str, generation_kwargs: dict[str, Any]):
    if args.baseline:
        if args.baseline_backend == "outlines":
            outlines_generation_kwargs = {**generation_kwargs, "do_sample": True}
            if getattr(tokenizer, "pad_token_id", None) is None:
                outlines_generation_kwargs["pad_token_id"] = tokenizer.eos_token_id
            return make_outlines_regex_generator(
                model,
                tokenizer,
                regex,
                **outlines_generation_kwargs,
            )
        return baseline_regex(model, tokenizer, regex, **generation_kwargs)

    return DiverseGuide(
        model,
        tokenizer,
        regex,
        gamma=args.gamma,
        completion_bias_eta=args.completion_bias_eta,
        path_frequency_estimator=args.path_frequency_estimator,
        ablation_component=args.ablation_component,
        **generation_kwargs,
    )


def replay_sample(generator, tokenizer, sample: str) -> bool:
    """Replay one accepted sample into the generator's global diversity counters."""
    logits_processor = getattr(generator, "logits_processor", None)
    if logits_processor is None:
        return False

    python_fallback = getattr(logits_processor, "_python_fallback", None)
    if python_fallback is not None:
        generator.update_generated_content(sample)
        return True

    token_ids = tokenizer.encode(sample, add_special_tokens=False)
    if hasattr(logits_processor, "update_generated_token_ids"):
        tensor = torch.tensor(token_ids, dtype=torch.long)
        logits_processor.update_generated_token_ids(tensor)
        return True

    if hasattr(generator, "update_generated_content"):
        generator.update_generated_content(sample)
        return True
    return False


def replay_existing_samples(generator, tokenizer, samples: list[str]) -> int:
    replayed = 0
    for sample in samples:
        if replay_sample(generator, tokenizer, sample):
            replayed += 1
    return replayed


def make_initial_data(
    args: argparse.Namespace,
    regex: str,
    prompt: str,
    device: str | None,
    model_parameter_device: str | None,
) -> GeneratedData:
    return GeneratedData(
        args.grammar,
        regex,
        prompt,
        args.model,
        args.max_tokens,
        args.top_k,
        args.top_p,
        args.temperature,
        args.seed,
        device,
        model_parameter_device,
        args.gamma,
        args.completion_bias_eta,
        args.path_frequency_estimator,
        args.baseline_backend if args.baseline else "internal",
        [],
        0.0,
        0,
        0,
        0,
    )


def to_output_dict(gen_data: GeneratedData, resume_record: dict[str, Any]) -> dict[str, Any]:
    output = gen_data.to_dict()
    history = []
    if isinstance(getattr(gen_data, "_resume_history", None), list):
        history.extend(gen_data._resume_history)  # type: ignore[attr-defined]
    history.append(resume_record)
    output["resume_history"] = history
    return output


def restore_generated_data(
    existing: dict[str, Any] | None,
    args: argparse.Namespace,
    regex: str,
    prompt: str,
    device: str | None,
    model_parameter_device: str | None,
) -> GeneratedData:
    if existing is None:
        return make_initial_data(args, regex, prompt, device, model_parameter_device)

    gen_data = GeneratedData.from_dict(existing)
    gen_data.device = device
    gen_data.model_parameter_device = model_parameter_device
    gen_data.accepted_samples = len(gen_data.samples)
    gen_data.rejected_samples = gen_data.rejected_samples or 0
    gen_data.attempted_batches = gen_data.attempted_batches or 0
    gen_data.elapsed_seconds = gen_data.elapsed_seconds or 0.0
    gen_data._resume_history = existing.get("resume_history", [])  # type: ignore[attr-defined]
    return gen_data


def main() -> None:
    args = parse_args()
    if args.grammar not in GRAMMAR_REGEX:
        raise ValueError(
            f"Unknown grammar {args.grammar!r}. Choose from: {list(GRAMMAR_REGEX)}"
        )

    regex = GRAMMAR_REGEX[args.grammar]
    prompt = GRAMMAR_PROMPT[args.grammar]
    output_path = output_path_for_args(args)
    input_path = args.input
    if input_path is None and output_path is not None and output_path.exists():
        input_path = output_path
    existing = load_existing(input_path)
    if existing is not None:
        validate_existing_artifact(existing, args, regex)

    set_generation_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    print(
        "[resume] "
        f"grammar={args.grammar} target_n={args.n} "
        f"input={input_path} output={output_path}"
    )
    print(f"[resume] device={device}")

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=dtype,
        local_files_only=args.local_files_only,
    ).to(device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        local_files_only=args.local_files_only,
    )
    model_parameter_device = get_model_parameter_device(model)

    generation_kwargs = build_generation_kwargs(args)
    generator = build_generator(args, model, tokenizer, regex, generation_kwargs)
    gen_data = restore_generated_data(
        existing,
        args,
        regex,
        prompt,
        device,
        model_parameter_device,
    )
    if len(gen_data.samples) >= args.n:
        print(
            f"[resume] existing accepted samples ({len(gen_data.samples)}) "
            f"already meet target ({args.n}); no generation needed."
        )
        if output_path is not None:
            write_json_atomic(
                output_path,
                to_output_dict(
                    gen_data,
                    {
                        "mode": "noop",
                        "input_path": str(input_path) if input_path else None,
                        "output_path": str(output_path),
                        "started_with_samples": len(gen_data.samples),
                        "target_n": args.n,
                        "replayed_samples": 0,
                        "new_accepted_samples": 0,
                        "new_rejected_samples": 0,
                        "elapsed_seconds": 0.0,
                    },
                ),
            )
        return

    existing_samples = list(gen_data.samples)
    replay_started = perf_counter()
    replayed_samples = 0
    if existing_samples and not args.baseline:
        replayed_samples = replay_existing_samples(generator, tokenizer, existing_samples)
    replay_elapsed = perf_counter() - replay_started
    print(
        "[resume] "
        f"loaded_samples={len(existing_samples)} replayed_samples={replayed_samples} "
        f"replay_elapsed={replay_elapsed:.3f}s"
    )
    if args.baseline and existing_samples:
        print("[resume] baseline mode has no diversity counter to replay")

    if output_path is not None:
        write_json_atomic(
            output_path,
            to_output_dict(
                gen_data,
                {
                    "mode": "resume_start",
                    "input_path": str(input_path) if input_path else None,
                    "output_path": str(output_path),
                    "started_with_samples": len(existing_samples),
                    "target_n": args.n,
                    "replayed_samples": replayed_samples,
                    "new_accepted_samples": 0,
                    "new_rejected_samples": 0,
                    "elapsed_seconds": 0.0,
                    "replay_elapsed_seconds": replay_elapsed,
                },
            ),
        )

    pattern = re.compile(f"^(?:{regex})$")
    started = perf_counter()
    old_elapsed = gen_data.elapsed_seconds or 0.0
    old_rejected = gen_data.rejected_samples or 0
    old_attempted_batches = gen_data.attempted_batches or 0

    progress_bar = None
    if not args.no_progress and not args.stdout_only:
        progress_bar = tqdm(
            total=args.n,
            initial=len(gen_data.samples),
            desc=f"{args.grammar}:resume",
            unit="sample",
            dynamic_ncols=True,
        )

    try:
        while len(gen_data.samples) < args.n:
            remaining = args.n - len(gen_data.samples)
            curr_batch_size = min(args.batch_size, remaining)
            gen_data.attempted_batches = (gen_data.attempted_batches or 0) + 1
            batch_texts = generator.generate_batch(
                prompt,
                n=curr_batch_size,
                max_tokens=args.max_tokens,
            )
            for text in batch_texts:
                if not pattern.match(text):
                    gen_data.rejected_samples = (gen_data.rejected_samples or 0) + 1
                    print(f"[resume] rejected: {text!r}")
                    continue
                gen_data.samples.append(text)
                gen_data.accepted_samples = len(gen_data.samples)
                if progress_bar is not None:
                    progress_bar.update(1)
                print(f"[resume] accepted {len(gen_data.samples)}/{args.n}: {text}")
                if len(gen_data.samples) >= args.n:
                    break

            continuation_elapsed = perf_counter() - started
            gen_data.elapsed_seconds = old_elapsed + continuation_elapsed
            if progress_bar is not None:
                accepted = len(gen_data.samples)
                rejected = gen_data.rejected_samples or 0
                progress_bar.set_postfix(
                    {
                        "accepted": accepted,
                        "rejected": rejected,
                        "elapsed_s": f"{gen_data.elapsed_seconds:.1f}",
                    }
                )
            if output_path is not None:
                new_rejected = (gen_data.rejected_samples or 0) - old_rejected
                write_json_atomic(
                    output_path,
                    to_output_dict(
                        gen_data,
                        {
                            "mode": "resume_progress",
                            "input_path": str(input_path) if input_path else None,
                            "output_path": str(output_path),
                            "started_with_samples": len(existing_samples),
                            "target_n": args.n,
                            "replayed_samples": replayed_samples,
                            "new_accepted_samples": len(gen_data.samples)
                            - len(existing_samples),
                            "new_rejected_samples": new_rejected,
                            "elapsed_seconds": continuation_elapsed,
                            "replay_elapsed_seconds": replay_elapsed,
                        },
                    ),
                )
    finally:
        if progress_bar is not None:
            progress_bar.close()

    continuation_elapsed = perf_counter() - started
    gen_data.elapsed_seconds = old_elapsed + continuation_elapsed
    gen_data.accepted_samples = len(gen_data.samples)
    new_rejected = (gen_data.rejected_samples or 0) - old_rejected
    new_attempted_batches = (gen_data.attempted_batches or 0) - old_attempted_batches
    resume_record = {
        "mode": "resume_complete",
        "input_path": str(input_path) if input_path else None,
        "output_path": str(output_path) if output_path else None,
        "started_with_samples": len(existing_samples),
        "target_n": args.n,
        "replayed_samples": replayed_samples,
        "new_accepted_samples": len(gen_data.samples) - len(existing_samples),
        "new_rejected_samples": new_rejected,
        "new_attempted_batches": new_attempted_batches,
        "elapsed_seconds": continuation_elapsed,
        "replay_elapsed_seconds": replay_elapsed,
    }
    if output_path is not None:
        write_json_atomic(output_path, to_output_dict(gen_data, resume_record))

    print(
        "[resume] done "
        f"accepted={len(gen_data.samples)} new_accepted={resume_record['new_accepted_samples']} "
        f"new_rejected={new_rejected} continuation_elapsed={continuation_elapsed:.2f}s"
    )


if __name__ == "__main__":
    main()
