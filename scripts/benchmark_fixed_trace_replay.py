"""Fixed-trace replay benchmark for DFA-guided generation.

The generation benchmarks measure realistic end-to-end latency, but each method
can visit different DFA states. This script separates two phases:

1. Collect token traces once, using the current generator.
2. Replay those exact token prefixes and measure processor/model components.

The replay path does not sample. It feeds the saved generated-token prefixes to
the logits processor and then commits the saved complete token sequence, so
different implementations can be compared on the same state trajectory.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import random
import subprocess
import sys
from pathlib import Path
from statistics import mean, pstdev
from time import perf_counter
from typing import Any

import torch
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.logits_process import LogitsProcessorList


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Collect and replay fixed token traces for engineering benchmarks."
    )
    parser.add_argument("grammar")
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--repo", type=Path, default=repo_root)
    parser.add_argument("--grammar-source-repo", type=Path, default=repo_root)
    parser.add_argument("--trace-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--collect-traces", action="store_true")
    parser.add_argument("--force-trace", action="store_true")
    parser.add_argument("--trace-count", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=24)
    parser.add_argument("--replay-runs", type=int, default=5)
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gamma", type=float, default=0.3)
    parser.add_argument("--completion-bias-eta", type=float, default=0.1)
    parser.add_argument(
        "--path-frequency-estimator",
        choices=["min", "harmonic"],
        default="harmonic",
    )
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--include-model-forward", action="store_true")
    parser.add_argument(
        "--rayon-threads",
        type=int,
        default=None,
        help="Set RAYON_NUM_THREADS before loading the Rust extension.",
    )
    parser.add_argument(
        "--parallel-threshold",
        type=int,
        default=None,
        help=(
            "Set DIVERSE_GUIDE_PARALLEL_ADJUSTMENT_THRESHOLD before loading "
            "the Rust extension."
        ),
    )
    parser.add_argument(
        "--torch-dtype",
        choices=["auto", "float16", "bfloat16", "float32"],
        default="auto",
    )
    parser.add_argument(
        "--fake-score-device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Device used for processor-only fake scores.",
    )
    args = parser.parse_args(argv)
    if args.trace_count <= 0:
        parser.error("--trace-count must be positive")
    if args.max_tokens <= 0:
        parser.error("--max-tokens must be positive")
    if args.replay_runs <= 0:
        parser.error("--replay-runs must be positive")
    if args.warmup_runs < 0:
        parser.error("--warmup-runs cannot be negative")
    if args.rayon_threads is not None and args.rayon_threads <= 0:
        parser.error("--rayon-threads must be positive")
    if args.parallel_threshold is not None and args.parallel_threshold <= 0:
        parser.error("--parallel-threshold must be positive")
    return args


def apply_runtime_env(args: argparse.Namespace) -> None:
    if args.rayon_threads is not None:
        os.environ["RAYON_NUM_THREADS"] = str(args.rayon_threads)
    if args.parallel_threshold is not None:
        os.environ["DIVERSE_GUIDE_PARALLEL_ADJUSTMENT_THRESHOLD"] = str(
            args.parallel_threshold
        )


def git_commit(repo: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        text=True,
    ).strip()


def load_generate_re(repo: Path) -> Any:
    module_path = repo / "scripts" / "generate_re.py"
    spec = importlib.util.spec_from_file_location(
        "_fixed_trace_generate_re", module_path
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load grammar source from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_current_guide(repo: Path) -> Any:
    repo = repo.resolve()
    for name in list(sys.modules):
        if name == "diverse_guide" or name.startswith("diverse_guide."):
            sys.modules.pop(name, None)
        if name == "regex_dfa_guide" or name.startswith("regex_dfa_guide."):
            sys.modules.pop(name, None)
    sys.path.insert(0, str(repo))
    sys.path.insert(0, str(repo / "src"))
    return importlib.import_module("diverse_guide.guide_rust")


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def synchronize_if_cuda(device: torch.device | str | None = None) -> None:
    if torch.cuda.is_available():
        if device is None:
            torch.cuda.synchronize()
        else:
            device_obj = torch.device(device)
            if device_obj.type == "cuda":
                torch.cuda.synchronize(device_obj)


def time_call(fn, *, sync_device: torch.device | str | None = None):  # noqa: ANN001
    synchronize_if_cuda(sync_device)
    started = perf_counter()
    result = fn()
    synchronize_if_cuda(sync_device)
    return result, (perf_counter() - started) * 1000.0


def select_dtype(name: str, device: str) -> torch.dtype:
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float32":
        return torch.float32
    return torch.float16 if device == "cuda" else torch.float32


def resolve_model_ref(model: str, local_files_only: bool) -> str:
    if not local_files_only:
        return model
    return snapshot_download(model, local_files_only=True)


def score_width(tokenizer: Any) -> int:
    vocab = tokenizer.get_vocab()
    ids = [int(token_id) for token_id in vocab.values()]
    eos = getattr(tokenizer, "eos_token_id", None)
    if eos is not None:
        ids.append(int(eos))
    return max(ids) + 1


def generation_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    kwargs: dict[str, Any] = {}
    if args.top_k is not None:
        kwargs["top_k"] = args.top_k
    if args.top_p is not None:
        kwargs["top_p"] = args.top_p
    if args.temperature is not None:
        kwargs["temperature"] = args.temperature
    return kwargs


def load_model_and_tokenizer(
    args: argparse.Namespace,
) -> tuple[Any, Any, dict[str, Any]]:
    cuda_available = torch.cuda.is_available()
    if args.require_cuda and not cuda_available:
        raise RuntimeError("CUDA is required, but torch.cuda.is_available() is false.")
    device = "cuda" if cuda_available else "cpu"
    dtype = select_dtype(args.torch_dtype, device)
    model_ref = resolve_model_ref(args.model, args.local_files_only)

    model_started = perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        model_ref,
        dtype=dtype,
        local_files_only=args.local_files_only,
        trust_remote_code=args.trust_remote_code,
    ).to(device)
    model.eval()
    model_load_seconds = perf_counter() - model_started

    tokenizer_started = perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(
        model_ref,
        local_files_only=args.local_files_only,
        trust_remote_code=args.trust_remote_code,
    )
    tokenizer_load_seconds = perf_counter() - tokenizer_started

    return model, tokenizer, {
        "model_ref": model_ref,
        "device": device,
        "torch_dtype": str(dtype).removeprefix("torch."),
        "model_load_seconds": model_load_seconds,
        "tokenizer_load_seconds": tokenizer_load_seconds,
        "model_parameter_device": str(next(model.parameters()).device),
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
    }


def collect_traces(
    *,
    args: argparse.Namespace,
    guide_rust: Any,
    model: Any,
    tokenizer: Any,
    regex: str,
    prompt: str,
) -> dict[str, Any]:
    set_seed(args.seed)
    processor = guide_rust.DiverseRegexLogitsProcessor(
        regex,
        tokenizer,
        gamma=args.gamma,
        completion_bias_eta=args.completion_bias_eta,
        path_frequency_estimator=args.path_frequency_estimator,
    )
    gen_kwargs = generation_kwargs(args)
    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    if pad_token_id is None:
        pad_token_id = getattr(tokenizer, "eos_token_id", None)
    if pad_token_id is not None:
        gen_kwargs.setdefault("pad_token_id", pad_token_id)

    traces = []
    for trace_index in range(args.trace_count):
        processor.dfa.reset_local_state_counter()
        processor.reset()
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        prompt_len = int(inputs["input_ids"].shape[1])
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                logits_processor=LogitsProcessorList([processor]),
                do_sample=True,
                max_new_tokens=args.max_tokens,
                **gen_kwargs,
            )
        generated_ids = [int(x) for x in output_ids[0, prompt_len:].tolist()]
        processor.update_generated_token_ids(
            torch.tensor(generated_ids, dtype=torch.long)
        )
        text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        traces.append(
            {
                "trace_index": trace_index,
                "prompt_token_ids": [int(x) for x in inputs["input_ids"][0].tolist()],
                "generated_token_ids": generated_ids,
                "generated_token_count": len(generated_ids),
                "text": text,
            }
        )

    return {
        "schema_version": 1,
        "experiment": "fixed_trace_collection",
        "model": args.model,
        "grammar": args.grammar,
        "regex": regex,
        "prompt": prompt,
        "parameters": {
            "trace_count": args.trace_count,
            "max_tokens": args.max_tokens,
            "seed": args.seed,
            "gamma": args.gamma,
            "completion_bias_eta": args.completion_bias_eta,
            "path_frequency_estimator": args.path_frequency_estimator,
            "generation_kwargs": gen_kwargs,
        },
        "traces": traces,
    }


def fake_score_device(
    args: argparse.Namespace, model_device: str | None
) -> torch.device:
    if args.fake_score_device == "cuda":
        return torch.device("cuda")
    if args.fake_score_device == "cpu":
        return torch.device("cpu")
    if args.include_model_forward and model_device is not None:
        return torch.device(model_device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def profile_adjustments(
    *,
    processor: Any,
    state: int,
    device: torch.device,
) -> dict[str, Any]:
    if state == -1:
        started = perf_counter()
        allowed_tensor = torch.tensor(
            [processor.guide.eos_token_id], dtype=torch.long, device=device
        )
        adjustments_tensor = processor._compute_adjustments_from_counts([0], [1]).to(
            device
        )
        return {
            "path": "terminal",
            "allowed_count": 1,
            "rust_compute_ms": 0.0,
            "pyo3_transfer_ms": 0.0,
            "allowed_tensor_ms": (perf_counter() - started) * 1000.0,
            "adjustment_tensor_ms": 0.0,
            "allowed_token_tensor": allowed_tensor,
            "adjustments_tensor": adjustments_tensor,
        }

    cache_key = (state, (device.type, device.index))
    allowed_tensor = processor._allowed_token_tensor_cache.get(cache_key)
    if allowed_tensor is None:
        started = perf_counter()
        token_ids, adjustments, rust_compute_ms = (
            processor.dfa.profile_compute_adjustments_with_estimator(
                state,
                processor.no_reward,
                processor.no_penalty,
                processor.completion_bias_eta,
                processor.path_frequency_estimator,
            )
        )
        total_call_ms = (perf_counter() - started) * 1000.0
        allowed_tensor, allowed_tensor_ms = time_call(
            lambda: torch.tensor(token_ids, dtype=torch.long, device=device),
            sync_device=device,
        )
        processor._allowed_token_tensor_cache[cache_key] = allowed_tensor
        path = "full-cache-miss"
    else:
        started = perf_counter()
        adjustments, rust_compute_ms = (
            processor.dfa.profile_compute_adjustment_values_with_estimator(
                state,
                processor.no_reward,
                processor.no_penalty,
                processor.completion_bias_eta,
                processor.path_frequency_estimator,
            )
        )
        total_call_ms = (perf_counter() - started) * 1000.0
        allowed_tensor_ms = 0.0
        path = "values-cache-hit"

    adjustments_tensor, adjustment_tensor_ms = time_call(
        lambda: torch.tensor(adjustments, dtype=torch.float, device=device),
        sync_device=device,
    )
    return {
        "path": path,
        "allowed_count": int(allowed_tensor.numel()),
        "rust_compute_ms": rust_compute_ms,
        "pyo3_transfer_ms": max(0.0, total_call_ms - rust_compute_ms),
        "allowed_tensor_ms": allowed_tensor_ms,
        "adjustment_tensor_ms": adjustment_tensor_ms,
        "allowed_token_tensor": allowed_tensor,
        "adjustments_tensor": adjustments_tensor,
    }


def profile_counts(
    *,
    processor: Any,
    state: int,
) -> dict[str, Any] | None:
    if state == -1:
        return None
    started = perf_counter()
    token_ids, reward_counts, penalty_counts, rust_compute_ms = (
        processor.dfa.profile_compute_counts_with_estimator(
            state,
            processor.path_frequency_estimator,
        )
    )
    total_call_ms = (perf_counter() - started) * 1000.0
    return {
        "allowed_count": len(token_ids),
        "reward_count_len": len(reward_counts),
        "penalty_count_len": len(penalty_counts),
        "rust_compute_ms": rust_compute_ms,
        "pyo3_transfer_ms": max(0.0, total_call_ms - rust_compute_ms),
    }


def scatter_scores(
    *,
    scores: torch.Tensor,
    allowed_tokens: torch.Tensor,
    adjustments: torch.Tensor,
    gamma: float,
    no_range_scaling: bool,
) -> torch.Tensor:
    batch_indices = torch.zeros_like(allowed_tokens)
    allowed_scores = scores[batch_indices, allowed_tokens]
    logits_range = allowed_scores.max() - allowed_scores.min()
    if no_range_scaling:
        final_adjustment = gamma * adjustments
    else:
        final_adjustment = gamma * logits_range * adjustments
    final_adjustment = final_adjustment.to(dtype=allowed_scores.dtype)
    output = scores.clone()
    output.fill_(float("-inf"))
    output[batch_indices, allowed_tokens] = allowed_scores + final_adjustment
    return output


def summarize(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "mean": None, "std": None, "min": None, "max": None}
    return {
        "count": len(values),
        "mean": mean(values),
        "std": pstdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
    }


def replay_once(
    *,
    args: argparse.Namespace,
    guide_rust: Any,
    tokenizer: Any,
    trace_data: dict[str, Any],
    model: Any | None,
    score_device: torch.device,
    run_index: int,
    warmup: bool,
) -> dict[str, Any]:
    processor = guide_rust.DiverseRegexLogitsProcessor(
        trace_data["regex"],
        tokenizer,
        gamma=args.gamma,
        completion_bias_eta=args.completion_bias_eta,
        path_frequency_estimator=args.path_frequency_estimator,
    )
    width = score_width(tokenizer)
    metrics: dict[str, list[float]] = {
        "advance_ms": [],
        "counts_rust_compute_ms": [],
        "counts_pyo3_transfer_ms": [],
        "adjust_rust_compute_ms": [],
        "adjust_pyo3_transfer_ms": [],
        "allowed_tensor_ms": [],
        "adjustment_tensor_ms": [],
        "torch_scatter_ms": [],
        "processor_step_ms": [],
        "model_forward_full_prefix_ms": [],
        "commit_valid_token_path_ms": [],
    }
    path_counts: dict[str, int] = {}
    allowed_counts: list[int] = []
    trace_lengths: list[int] = []

    for trace in trace_data["traces"]:
        processor.dfa.reset_local_state_counter()
        processor.reset()
        prompt_ids = [int(x) for x in trace["prompt_token_ids"]]
        generated_ids = [int(x) for x in trace["generated_token_ids"]]
        trace_lengths.append(len(generated_ids))
        processor._seq_start_idx = len(prompt_ids)
        processor._ensure_rows(1)

        for step_idx in range(len(generated_ids)):
            prefix_ids = prompt_ids + generated_ids[:step_idx]
            input_ids = torch.tensor([prefix_ids], dtype=torch.long)
            gen_ids = input_ids[0, processor._seq_start_idx :]

            if model is not None:
                model_input = input_ids.to(model.device)
                with torch.no_grad():
                    outputs, model_ms = time_call(
                        lambda: model(input_ids=model_input, use_cache=False),
                        sync_device=model.device,
                    )
                scores = outputs.logits[:, -1, :].detach()
                metrics["model_forward_full_prefix_ms"].append(model_ms)
                score_device = scores.device
            else:
                scores = torch.zeros(
                    (1, width), dtype=torch.float32, device=score_device
                )

            _, advance_ms = time_call(
                lambda: processor._advance_row(0, gen_ids),
                sync_device=None,
            )
            state = processor._guide_states_by_row[0]
            metrics["advance_ms"].append(advance_ms)

            counts_profile = profile_counts(processor=processor, state=state)
            if counts_profile is not None:
                metrics["counts_rust_compute_ms"].append(
                    float(counts_profile["rust_compute_ms"])
                )
                metrics["counts_pyo3_transfer_ms"].append(
                    float(counts_profile["pyo3_transfer_ms"])
                )

            profile, profile_ms = time_call(
                lambda: profile_adjustments(
                    processor=processor,
                    state=state,
                    device=score_device,
                ),
                sync_device=score_device,
            )
            path_counts[profile["path"]] = path_counts.get(profile["path"], 0) + 1
            allowed_counts.append(profile["allowed_count"])
            metrics["adjust_rust_compute_ms"].append(profile["rust_compute_ms"])
            metrics["adjust_pyo3_transfer_ms"].append(profile["pyo3_transfer_ms"])
            metrics["allowed_tensor_ms"].append(profile["allowed_tensor_ms"])
            metrics["adjustment_tensor_ms"].append(profile["adjustment_tensor_ms"])

            _, scatter_ms = time_call(
                lambda: scatter_scores(
                    scores=scores,
                    allowed_tokens=profile["allowed_token_tensor"],
                    adjustments=profile["adjustments_tensor"],
                    gamma=processor.gamma,
                    no_range_scaling=processor.no_range_scaling,
                ),
                sync_device=score_device,
            )
            metrics["torch_scatter_ms"].append(scatter_ms)
            metrics["processor_step_ms"].append(advance_ms + profile_ms + scatter_ms)

        _, commit_ms = time_call(
            lambda ids=generated_ids: processor.dfa.commit_valid_token_path(ids),
            sync_device=None,
        )
        metrics["commit_valid_token_path_ms"].append(commit_ms)

    return {
        "run_index": run_index,
        "warmup": warmup,
        "trace_count": len(trace_data["traces"]),
        "step_count": sum(trace_lengths),
        "trace_lengths": trace_lengths,
        "allowed_count": summarize([float(x) for x in allowed_counts]),
        "path_counts": path_counts,
        "metrics": {name: summarize(values) for name, values in metrics.items()},
    }


def aggregate_runs(runs: list[dict[str, Any]]) -> dict[str, Any]:
    metric_names = tuple(runs[0]["metrics"].keys())
    return {
        "trace_count": runs[0]["trace_count"],
        "step_count": runs[0]["step_count"],
        "allowed_count": summarize(
            [float(run["allowed_count"]["mean"]) for run in runs]
        ),
        "path_counts": runs[0]["path_counts"],
        "metrics": {
            name: summarize(
                [
                    float(run["metrics"][name]["mean"])
                    for run in runs
                    if run["metrics"][name]["mean"] is not None
                ]
            )
            for name in metric_names
        },
    }


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = parse_args(argv)
    apply_runtime_env(args)
    repo = args.repo.resolve()
    guide_rust = load_current_guide(repo)
    generate_re = load_generate_re(args.grammar_source_repo.resolve())
    if args.grammar not in generate_re.GRAMMAR_REGEX:
        raise ValueError(f"unknown grammar: {args.grammar}")
    regex = generate_re.GRAMMAR_REGEX[args.grammar]
    prompt = generate_re.GRAMMAR_PROMPT[args.grammar]

    needs_model = args.collect_traces or args.include_model_forward
    model = None
    model_loading: dict[str, Any] = {}
    if needs_model:
        model, tokenizer, model_loading = load_model_and_tokenizer(args)
    else:
        model_ref = resolve_model_ref(args.model, args.local_files_only)
        tokenizer = AutoTokenizer.from_pretrained(
            model_ref,
            local_files_only=args.local_files_only,
            trust_remote_code=args.trust_remote_code,
        )

    if args.collect_traces or args.force_trace or not args.trace_path.exists():
        if model is None:
            model, tokenizer, model_loading = load_model_and_tokenizer(args)
        trace_data = collect_traces(
            args=args,
            guide_rust=guide_rust,
            model=model,
            tokenizer=tokenizer,
            regex=regex,
            prompt=prompt,
        )
        args.trace_path.parent.mkdir(parents=True, exist_ok=True)
        args.trace_path.write_text(
            json.dumps(trace_data, indent=2, sort_keys=True) + "\n"
        )
        print(f"[trace] wrote {args.trace_path}")
    else:
        trace_data = json.loads(args.trace_path.read_text())

    score_device = fake_score_device(
        args,
        str(next(model.parameters()).device) if model is not None else None,
    )
    warmups = [
        replay_once(
            args=args,
            guide_rust=guide_rust,
            tokenizer=tokenizer,
            trace_data=trace_data,
            model=model if args.include_model_forward else None,
            score_device=score_device,
            run_index=i,
            warmup=True,
        )
        for i in range(args.warmup_runs)
    ]
    runs = [
        replay_once(
            args=args,
            guide_rust=guide_rust,
            tokenizer=tokenizer,
            trace_data=trace_data,
            model=model if args.include_model_forward else None,
            score_device=score_device,
            run_index=i,
            warmup=False,
        )
        for i in range(args.replay_runs)
    ]

    result = {
        "schema_version": 1,
        "experiment": "fixed_trace_replay",
        "repo": str(repo),
        "git_commit": git_commit(repo),
        "model": args.model,
        "grammar": args.grammar,
        "regex": regex,
        "prompt": prompt,
        "trace_path": str(args.trace_path),
        "trace_schema_version": trace_data.get("schema_version"),
        "model_loading": model_loading,
        "tokenizer": {
            "vocab_size": len(tokenizer.get_vocab()),
            "score_width": score_width(tokenizer),
            "eos_token_id": getattr(tokenizer, "eos_token_id", None),
        },
        "parameters": {
            "replay_runs": args.replay_runs,
            "warmup_runs": args.warmup_runs,
            "include_model_forward": args.include_model_forward,
            "fake_score_device": str(score_device),
            "gamma": args.gamma,
            "completion_bias_eta": args.completion_bias_eta,
            "path_frequency_estimator": args.path_frequency_estimator,
            "seed": args.seed,
            "rayon_threads": args.rayon_threads,
            "parallel_threshold": args.parallel_threshold,
            "rayon_num_threads_env": os.environ.get("RAYON_NUM_THREADS"),
            "parallel_threshold_env": os.environ.get(
                "DIVERSE_GUIDE_PARALLEL_ADJUSTMENT_THRESHOLD"
            ),
        },
        "warmups": warmups,
        "runs": runs,
        "summary": aggregate_runs(runs),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(f"[replay] wrote {args.output}")
    print(json.dumps(result["summary"], indent=2, sort_keys=True))
    return result


if __name__ == "__main__":
    main()
