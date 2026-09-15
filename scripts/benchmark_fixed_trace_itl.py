"""Fixed-trace inter-token latency benchmark for DiverseGuide variants.

This benchmark removes sampling/path variance by replaying saved token traces.
For every saved prefix, it runs the model with KV cache and applies the selected
logits processor, but it takes the next token from the trace instead of sampling.
The first token step is reported separately; ITL is computed from cached decode
steps after the first generated token.
"""

from __future__ import annotations

import argparse
import ast
import importlib
import importlib.util
import inspect
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

METHODS = ("dgplus", "old-dg")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Replay fixed token traces and measure cached decode ITL."
    )
    parser.add_argument("grammar")
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--method", choices=METHODS, default="dgplus")
    parser.add_argument("--repo", type=Path, default=repo_root)
    parser.add_argument(
        "--old-repo",
        type=Path,
        default=repo_root / ".worktrees" / "origin-master-engineering",
    )
    parser.add_argument("--grammar-source-repo", type=Path, default=repo_root)
    parser.add_argument("--trace-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--sample-artifact",
        type=Path,
        default=None,
        help="Existing generation artifact JSON whose samples are replayed once.",
    )
    parser.add_argument(
        "--target-replay-tokens",
        type=int,
        default=256,
        help=(
            "Use unique artifact samples until at least this many tokens are "
            "replayed."
        ),
    )
    parser.add_argument(
        "--sample-limit",
        type=int,
        default=None,
        help="Maximum number of unique artifact samples to replay.",
    )
    parser.add_argument(
        "--allow-duplicate-samples",
        action="store_true",
        help="Keep duplicate sample strings from the artifact instead of deduping.",
    )
    parser.add_argument("--collect-traces", action="store_true")
    parser.add_argument("--force-trace", action="store_true")
    parser.add_argument("--trace-count", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=24)
    parser.add_argument("--replay-runs", type=int, default=1)
    parser.add_argument("--warmup-runs", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gamma", type=float, default=None)
    parser.add_argument("--beta", type=float, default=3.0)
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
    parser.add_argument(
        "--torch-dtype",
        choices=["auto", "float16", "bfloat16", "float32"],
        default="auto",
    )
    parser.add_argument(
        "--rayon-threads",
        type=int,
        default=None,
        help="Set RAYON_NUM_THREADS before loading the current Rust extension.",
    )
    parser.add_argument(
        "--parallel-threshold",
        type=int,
        default=None,
        help="Set DIVERSE_GUIDE_PARALLEL_ADJUSTMENT_THRESHOLD before loading Rust.",
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
    if args.target_replay_tokens <= 0:
        parser.error("--target-replay-tokens must be positive")
    if args.sample_limit is not None and args.sample_limit <= 0:
        parser.error("--sample-limit must be positive")
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


def path_is_under(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def _literal_eval_grammar_node(node: ast.AST) -> Any:
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Dict):
        return {
            _literal_eval_grammar_node(key): _literal_eval_grammar_node(value)
            for key, value in zip(node.keys, node.values, strict=True)
        }
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        return _literal_eval_grammar_node(node.left) + _literal_eval_grammar_node(
            node.right
        )
    return ast.literal_eval(node)


def load_generate_re(repo: Path) -> Any:
    module_path = repo / "scripts" / "generate_re.py"
    tree = ast.parse(module_path.read_text(), filename=str(module_path))
    values: dict[str, Any] = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id in {
                "GRAMMAR_REGEX",
                "GRAMMAR_PROMPT",
            }:
                values[target.id] = _literal_eval_grammar_node(node.value)
    if "GRAMMAR_REGEX" not in values or "GRAMMAR_PROMPT" not in values:
        raise ImportError(f"could not load grammar constants from {module_path}")
    return type("GrammarModule", (), values)


def clear_project_modules() -> None:
    for name in list(sys.modules):
        if name == "diverse_guide" or name.startswith("diverse_guide."):
            sys.modules.pop(name, None)
        if name == "regex_dfa_guide" or name.startswith("regex_dfa_guide."):
            sys.modules.pop(name, None)


def load_guide_modules(repo: Path) -> dict[str, Any]:
    clear_project_modules()
    repo = repo.resolve()
    sys.path.insert(0, str(repo))
    sys.path.insert(0, str(repo / "src"))
    guide_rust = importlib.import_module("diverse_guide.guide_rust")
    regex_dfa_guide = importlib.import_module("regex_dfa_guide")
    rs_module = importlib.import_module("regex_dfa_guide.regex_dfa_guide_rs")
    return {
        "guide_rust": guide_rust,
        "regex_dfa_guide": regex_dfa_guide,
        "rs_module": rs_module,
    }


def validate_implementation(
    *, method: str, repo: Path, modules: dict[str, Any]
) -> dict[str, Any]:
    repo = repo.resolve()
    guide_rust = modules["guide_rust"]
    regex_pkg = modules["regex_dfa_guide"]
    rs_module = modules["rs_module"]
    guide_rust_file = Path(guide_rust.__file__).resolve()
    regex_pkg_file = Path(regex_pkg.__file__).resolve()
    rs_file = Path(rs_module.__file__).resolve()
    checks = {
        "guide_rust_under_repo": path_is_under(guide_rust_file, repo),
        "regex_package_under_repo": path_is_under(regex_pkg_file, repo),
        "rust_extension_under_repo": path_is_under(rs_file, repo),
    }
    failed = [name for name, ok in checks.items() if not ok]
    if failed:
        raise RuntimeError(
            "implementation path validation failed: "
            + ", ".join(failed)
            + (
                f"; guide_rust={guide_rust_file}; "
                f"regex_pkg={regex_pkg_file}; rs={rs_file}"
            )
        )

    processor_sig = inspect.signature(guide_rust.DiverseRegexLogitsProcessor.__init__)
    has_eta_arg = "completion_bias_eta" in processor_sig.parameters
    has_estimator_arg = "path_frequency_estimator" in processor_sig.parameters
    has_values_api = hasattr(rs_module.DiverseGuideDFA, "compute_adjustment_values")
    if method == "dgplus" and not (has_eta_arg and has_estimator_arg):
        raise RuntimeError("loaded implementation is not DG+")
    if method == "old-dg" and (has_eta_arg or has_estimator_arg or has_values_api):
        raise RuntimeError("loaded implementation is not origin/master old DG")

    return {
        "repo": str(repo),
        "git_commit": git_commit(repo),
        "python_executable": sys.executable,
        "guide_rust_file": str(guide_rust_file),
        "regex_dfa_guide_file": str(regex_pkg_file),
        "rust_extension_file": str(rs_file),
        "processor_signature": str(processor_sig),
        "checks": checks,
        "has_eta_arg": has_eta_arg,
        "has_estimator_arg": has_estimator_arg,
        "has_values_api": has_values_api,
    }


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def synchronize_if_cuda(device: torch.device | str | None = None) -> None:
    if not torch.cuda.is_available():
        return
    if device is None:
        torch.cuda.synchronize()
        return
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

    parameter_device = str(next(model.parameters()).device)
    if args.require_cuda and not parameter_device.startswith("cuda"):
        raise RuntimeError(f"model is not on CUDA: {parameter_device}")

    return model, tokenizer, {
        "model_ref": model_ref,
        "device": device,
        "torch_dtype": str(dtype).removeprefix("torch."),
        "model_load_seconds": model_load_seconds,
        "tokenizer_load_seconds": tokenizer_load_seconds,
        "model_parameter_device": parameter_device,
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
    }


def tokenizer_metadata(tokenizer: Any) -> dict[str, Any]:
    vocab = tokenizer.get_vocab()
    ids = [int(token_id) for token_id in vocab.values()]
    eos = getattr(tokenizer, "eos_token_id", None)
    if eos is not None:
        ids.append(int(eos))
    return {
        "class": type(tokenizer).__name__,
        "vocab_size": len(vocab),
        "score_width": max(ids) + 1 if ids else 0,
        "eos_token_id": eos,
        "pad_token_id": getattr(tokenizer, "pad_token_id", None),
    }


def dgplus_processor(
    guide_rust: Any, regex: str, tokenizer: Any, args: argparse.Namespace
):
    gamma = 0.3 if args.gamma is None else args.gamma
    return guide_rust.DiverseRegexLogitsProcessor(
        regex,
        tokenizer,
        gamma=gamma,
        completion_bias_eta=args.completion_bias_eta,
        path_frequency_estimator=args.path_frequency_estimator,
    )


def old_dg_processor(
    guide_rust: Any, regex: str, tokenizer: Any, args: argparse.Namespace
):
    gamma = (1.0 / 6.0) if args.gamma is None else args.gamma
    return guide_rust.DiverseRegexLogitsProcessor(
        regex,
        tokenizer,
        gamma=gamma,
        beta=args.beta,
    )


def build_processor(
    *,
    method: str,
    guide_rust: Any,
    regex: str,
    tokenizer: Any,
    args: argparse.Namespace,
):
    if method == "dgplus":
        return dgplus_processor(guide_rust, regex, tokenizer, args)
    if method == "old-dg":
        return old_dg_processor(guide_rust, regex, tokenizer, args)
    raise ValueError(f"unsupported method: {method}")


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
    processor = dgplus_processor(guide_rust, regex, tokenizer, args)
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
        if hasattr(processor, "update_generated_token_ids"):
            processor.update_generated_token_ids(
                torch.tensor(generated_ids, dtype=torch.long)
            )
        text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        if hasattr(processor, "update_generated_content"):
            processor.update_generated_content(text)
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
        "collector_method": "dgplus",
        "model": args.model,
        "grammar": args.grammar,
        "regex": regex,
        "prompt": prompt,
        "parameters": {
            "trace_count": args.trace_count,
            "max_tokens": args.max_tokens,
            "seed": args.seed,
            "gamma": 0.3 if args.gamma is None else args.gamma,
            "completion_bias_eta": args.completion_bias_eta,
            "path_frequency_estimator": args.path_frequency_estimator,
            "generation_kwargs": gen_kwargs,
        },
        "traces": traces,
    }


def trace_data_from_sample_artifact(
    *,
    artifact_path: Path,
    tokenizer: Any,
    args: argparse.Namespace,
) -> dict[str, Any]:
    artifact = json.loads(artifact_path.read_text())
    samples = artifact.get("samples")
    if not isinstance(samples, list):
        raise ValueError(f"sample artifact has no list-valued samples: {artifact_path}")
    artifact_model = artifact.get("model")
    artifact_grammar = artifact.get("grammar")
    if artifact_model is not None and artifact_model != args.model:
        raise ValueError(
            "sample artifact model mismatch: "
            f"expected {args.model}, got {artifact_model}"
        )
    if artifact_grammar is not None and artifact_grammar != args.grammar:
        raise ValueError(
            "sample artifact grammar mismatch: "
            f"expected {args.grammar}, got {artifact_grammar}"
        )

    prompt = artifact.get("prompt")
    regex = artifact.get("regex")
    if not isinstance(prompt, str) or not isinstance(regex, str):
        raise ValueError(
            f"sample artifact must contain prompt and regex: {artifact_path}"
        )

    prompt_ids = [
        int(token_id)
        for token_id in tokenizer(prompt, return_tensors="pt")["input_ids"][0].tolist()
    ]
    traces: list[dict[str, Any]] = []
    seen_samples: set[str] = set()
    total_tokens = 0
    duplicate_count = 0
    empty_tokenized_count = 0

    for artifact_index, sample in enumerate(samples):
        if not isinstance(sample, str):
            continue
        if not args.allow_duplicate_samples and sample in seen_samples:
            duplicate_count += 1
            continue
        seen_samples.add(sample)
        try:
            generated_ids = tokenizer.encode(sample, add_special_tokens=False)
        except TypeError:
            generated_ids = tokenizer.encode(sample)
        generated_ids = [int(token_id) for token_id in generated_ids]
        if not generated_ids:
            empty_tokenized_count += 1
            continue
        traces.append(
            {
                "trace_index": len(traces),
                "artifact_sample_index": artifact_index,
                "prompt_token_ids": prompt_ids,
                "generated_token_ids": generated_ids,
                "generated_token_count": len(generated_ids),
                "text": sample,
            }
        )
        total_tokens += len(generated_ids)
        if args.sample_limit is not None and len(traces) >= args.sample_limit:
            break
        if total_tokens >= args.target_replay_tokens:
            break

    if not traces:
        raise ValueError(f"no usable samples found in artifact: {artifact_path}")

    return {
        "schema_version": 2,
        "experiment": "fixed_trace_from_sample_artifact",
        "collector_method": "sample-artifact",
        "sample_artifact": str(artifact_path),
        "sample_artifact_method": artifact.get("baseline_backend")
        or artifact.get("method"),
        "model": args.model,
        "grammar": args.grammar,
        "regex": regex,
        "prompt": prompt,
        "parameters": {
            "target_replay_tokens": args.target_replay_tokens,
            "sample_limit": args.sample_limit,
            "allow_duplicate_samples": args.allow_duplicate_samples,
            "artifact_sample_count": len(samples),
            "used_sample_count": len(traces),
            "used_token_count": total_tokens,
            "skipped_duplicate_count": duplicate_count,
            "empty_tokenized_count": empty_tokenized_count,
        },
        "traces": traces,
    }


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


def commit_trace(processor: Any, tokenizer: Any, generated_ids: list[int]) -> None:
    if hasattr(processor, "update_generated_token_ids"):
        processor.update_generated_token_ids(
            torch.tensor(generated_ids, dtype=torch.long)
        )
        return
    text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    if hasattr(processor, "update_generated_content"):
        processor.update_generated_content(text)
        return
    dfa = getattr(processor, "dfa", None)
    if dfa is not None and hasattr(dfa, "update_path_counter"):
        dfa.update_path_counter(text)


def replay_once(
    *,
    args: argparse.Namespace,
    guide_rust: Any,
    tokenizer: Any,
    model: Any,
    trace_data: dict[str, Any],
    run_index: int,
    warmup: bool,
) -> dict[str, Any]:
    processor = build_processor(
        method=args.method,
        guide_rust=guide_rust,
        regex=trace_data["regex"],
        tokenizer=tokenizer,
        args=args,
    )
    metrics: dict[str, list[float]] = {
        "first_token_step_ms": [],
        "cached_decode_itl_ms": [],
        "all_token_step_ms": [],
        "model_forward_ms": [],
        "model_forward_first_token_ms": [],
        "model_forward_cached_decode_ms": [],
        "score_prepare_ms": [],
        "processor_ms": [],
        "processor_first_token_ms": [],
        "processor_cached_decode_ms": [],
        "commit_ms": [],
    }
    trace_lengths: list[int] = []

    for trace in trace_data["traces"]:
        processor.dfa.reset_local_state_counter()
        processor.reset()
        prompt_ids = [int(x) for x in trace["prompt_token_ids"]]
        generated_ids = [int(x) for x in trace["generated_token_ids"]]
        trace_lengths.append(len(generated_ids))
        if not generated_ids:
            continue

        past_key_values = None
        for step_idx, _token_id in enumerate(generated_ids):
            if step_idx == 0:
                model_input_ids = torch.tensor(
                    [prompt_ids], dtype=torch.long, device=model.device
                )
                processor_input_ids = model_input_ids
            else:
                model_input_ids = torch.tensor(
                    [[generated_ids[step_idx - 1]]],
                    dtype=torch.long,
                    device=model.device,
                )
                processor_input_ids = torch.tensor(
                    [prompt_ids + generated_ids[:step_idx]],
                    dtype=torch.long,
                    device=model.device,
                )

            with torch.no_grad():
                outputs, model_ms = time_call(
                    lambda ids=model_input_ids, past=past_key_values: model(
                        input_ids=ids,
                        past_key_values=past,
                        use_cache=True,
                    ),
                    sync_device=model.device,
                )
            past_key_values = outputs.past_key_values
            scores, score_prepare_ms = time_call(
                lambda out=outputs: out.logits[:, -1, :].detach().float().clone(),
                sync_device=model.device,
            )

            _, processor_ms = time_call(
                lambda ids=processor_input_ids, logits=scores: processor(ids, logits),
                sync_device=model.device,
            )
            step_ms = model_ms + score_prepare_ms + processor_ms

            metrics["model_forward_ms"].append(model_ms)
            metrics["score_prepare_ms"].append(score_prepare_ms)
            metrics["processor_ms"].append(processor_ms)
            metrics["all_token_step_ms"].append(step_ms)
            if step_idx == 0:
                metrics["first_token_step_ms"].append(step_ms)
                metrics["model_forward_first_token_ms"].append(model_ms)
                metrics["processor_first_token_ms"].append(processor_ms)
            else:
                metrics["cached_decode_itl_ms"].append(step_ms)
                metrics["model_forward_cached_decode_ms"].append(model_ms)
                metrics["processor_cached_decode_ms"].append(processor_ms)

        _, commit_ms = time_call(
            lambda ids=generated_ids: commit_trace(processor, tokenizer, ids),
            sync_device=None,
        )
        metrics["commit_ms"].append(commit_ms)

    return {
        "run_index": run_index,
        "warmup": warmup,
        "trace_count": len(trace_data["traces"]),
        "step_count": sum(trace_lengths),
        "trace_lengths": trace_lengths,
        "metrics": {name: summarize(values) for name, values in metrics.items()},
    }


def aggregate_runs(runs: list[dict[str, Any]]) -> dict[str, Any]:
    metric_names = tuple(runs[0]["metrics"].keys())
    return {
        "trace_count": runs[0]["trace_count"],
        "step_count": runs[0]["step_count"],
        "trace_lengths": runs[0]["trace_lengths"],
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
    replay_repo = (
        args.repo.resolve() if args.method == "dgplus" else args.old_repo.resolve()
    )
    collector_repo = args.repo.resolve()
    grammar_source_repo = args.grammar_source_repo.resolve()

    replay_modules = load_guide_modules(replay_repo)
    implementation = validate_implementation(
        method=args.method,
        repo=replay_repo,
        modules=replay_modules,
    )
    guide_rust = replay_modules["guide_rust"]
    generate_re = load_generate_re(grammar_source_repo)
    if args.grammar not in generate_re.GRAMMAR_REGEX:
        raise ValueError(f"unknown grammar: {args.grammar}")
    regex = generate_re.GRAMMAR_REGEX[args.grammar]
    prompt = generate_re.GRAMMAR_PROMPT[args.grammar]

    model, tokenizer, model_loading = load_model_and_tokenizer(args)
    if args.sample_artifact is not None:
        trace_data = trace_data_from_sample_artifact(
            artifact_path=args.sample_artifact.resolve(),
            tokenizer=tokenizer,
            args=args,
        )
        args.trace_path.parent.mkdir(parents=True, exist_ok=True)
        args.trace_path.write_text(
            json.dumps(trace_data, indent=2, sort_keys=True) + "\n"
        )
        print(f"[trace-from-artifact] wrote {args.trace_path}")
    elif args.collect_traces or args.force_trace or not args.trace_path.exists():
        if args.method != "dgplus":
            raise RuntimeError(
                "trace collection must be run with --method dgplus; replay old-dg "
                "against the saved trace afterward."
            )
        collector_modules = replay_modules
        if collector_repo != replay_repo:
            collector_modules = load_guide_modules(collector_repo)
        trace_data = collect_traces(
            args=args,
            guide_rust=collector_modules["guide_rust"],
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

    warmups = [
        replay_once(
            args=args,
            guide_rust=guide_rust,
            tokenizer=tokenizer,
            model=model,
            trace_data=trace_data,
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
            model=model,
            trace_data=trace_data,
            run_index=i,
            warmup=False,
        )
        for i in range(args.replay_runs)
    ]

    result = {
        "schema_version": 1,
        "experiment": "fixed_trace_itl_replay",
        "method": args.method,
        "implementation": implementation,
        "grammar_source_repo": str(grammar_source_repo),
        "grammar_source_commit": git_commit(grammar_source_repo),
        "model": args.model,
        "grammar": args.grammar,
        "regex": regex,
        "prompt": prompt,
        "trace_path": str(args.trace_path),
        "trace_schema_version": trace_data.get("schema_version"),
        "trace_source_experiment": trace_data.get("experiment"),
        "sample_artifact": trace_data.get("sample_artifact"),
        "model_loading": model_loading,
        "tokenizer": tokenizer_metadata(tokenizer),
        "parameters": {
            "replay_runs": args.replay_runs,
            "warmup_runs": args.warmup_runs,
            "trace_count": args.trace_count,
            "max_tokens": args.max_tokens,
            "seed": args.seed,
            "gamma": (0.3 if args.method == "dgplus" else 1.0 / 6.0)
            if args.gamma is None
            else args.gamma,
            "beta": args.beta if args.method == "old-dg" else None,
            "completion_bias_eta": (
                args.completion_bias_eta if args.method == "dgplus" else None
            ),
            "path_frequency_estimator": (
                args.path_frequency_estimator if args.method == "dgplus" else None
            ),
            "local_files_only": args.local_files_only,
            "trust_remote_code": args.trust_remote_code,
            "require_cuda": args.require_cuda,
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
    print(f"[fixed-trace-itl] wrote {args.output}")
    print(json.dumps(result["summary"], indent=2, sort_keys=True))
    return result


if __name__ == "__main__":
    main()
