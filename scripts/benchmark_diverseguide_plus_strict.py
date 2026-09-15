"""Strict cumulative-token benchmark for current diverse-guide+."""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import inspect
import json
import sys
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_shared_benchmark_module(repo: Path) -> Any:
    module_path = repo / "scripts" / "benchmark_origin_master_diverseguide_strict.py"
    spec = importlib.util.spec_from_file_location(
        "_strict_benchmark_shared",
        module_path,
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load shared benchmark helpers from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Strict cumulative-token benchmark for current diverse-guide+."
    )
    parser.add_argument("grammar")
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--repo", type=Path, default=repo_root)
    parser.add_argument("--grammar-source-repo", type=Path, default=repo_root)
    parser.add_argument("--expected-current-commit", default=None)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--target-generated-tokens", type=int, default=256)
    parser.add_argument("--max-tokens", type=int, default=24)
    parser.add_argument(
        "--build-only",
        action="store_true",
        help="Measure construction only; do not load a full model or run generation.",
    )
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
    parser.add_argument(
        "--torch-dtype",
        choices=["auto", "float16", "bfloat16", "float32"],
        default="auto",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.runs <= 0:
        parser.error("--runs must be positive")
    if args.warmup_runs < 0:
        parser.error("--warmup-runs cannot be negative")
    if args.target_generated_tokens <= 0:
        parser.error("--target-generated-tokens must be positive")
    if args.max_tokens <= 0:
        parser.error("--max-tokens must be positive")
    if args.gamma < 0.0:
        parser.error("--gamma must be non-negative")
    if args.completion_bias_eta < 0.0:
        parser.error("--completion-bias-eta must be non-negative")
    return args


def load_current_modules(repo: Path) -> dict[str, Any]:
    for name in list(sys.modules):
        if name == "diverse_guide" or name.startswith("diverse_guide."):
            sys.modules.pop(name, None)
        if name == "regex_dfa_guide" or name.startswith("regex_dfa_guide."):
            sys.modules.pop(name, None)

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


def validate_current_implementation(
    *,
    repo: Path,
    expected_commit: str | None,
    modules: dict[str, Any],
    shared: Any,
) -> dict[str, Any]:
    repo = repo.resolve()
    commit = shared.git_commit(repo)
    if expected_commit is not None and commit != expected_commit:
        raise RuntimeError(
            f"current repo commit mismatch: expected {expected_commit}, got {commit}"
        )

    guide_rust = modules["guide_rust"]
    regex_dfa_guide = modules["regex_dfa_guide"]
    rs_module = modules["rs_module"]
    guide_rust_file = Path(guide_rust.__file__).resolve()
    regex_pkg_file = Path(regex_dfa_guide.__file__).resolve()
    rs_file = Path(rs_module.__file__).resolve()

    checks = {
        "guide_rust_under_repo": shared.path_is_under(guide_rust_file, repo),
        "regex_package_under_repo": shared.path_is_under(regex_pkg_file, repo),
        "rust_extension_under_repo": shared.path_is_under(rs_file, repo),
    }
    failed = [name for name, ok in checks.items() if not ok]
    if failed:
        raise RuntimeError(
            "current implementation path validation failed: "
            + ", ".join(failed)
            + (
                f"; guide_rust={guide_rust_file}; "
                f"regex_pkg={regex_pkg_file}; rs={rs_file}"
            )
        )

    processor_sig = inspect.signature(guide_rust.DiverseRegexLogitsProcessor.__init__)
    has_eta_arg = "completion_bias_eta" in processor_sig.parameters
    has_estimator_arg = "path_frequency_estimator" in processor_sig.parameters
    has_adjustment_api = hasattr(
        rs_module.DiverseGuideDFA,
        "compute_adjustments_with_estimator",
    )
    if not (has_eta_arg and has_estimator_arg and has_adjustment_api):
        raise RuntimeError("loaded implementation is not current diverse-guide+ API")

    return {
        "repo": str(repo),
        "git_commit": commit,
        "python_executable": sys.executable,
        "guide_rust_file": str(guide_rust_file),
        "regex_dfa_guide_file": str(regex_pkg_file),
        "rust_extension_file": str(rs_file),
        "processor_signature": str(processor_sig),
        "checks": checks,
        "has_eta_arg": has_eta_arg,
        "has_estimator_arg": has_estimator_arg,
        "has_adjustment_api": has_adjustment_api,
    }


def generation_kwargs_from_args(args: argparse.Namespace) -> dict[str, Any]:
    kwargs: dict[str, Any] = {}
    if args.top_k is not None:
        kwargs["top_k"] = args.top_k
    if args.top_p is not None:
        kwargs["top_p"] = args.top_p
    if args.temperature is not None:
        kwargs["temperature"] = args.temperature
    return kwargs


def build_generator(
    *,
    guide_rust: Any,
    model: Any,
    tokenizer: Any,
    regex: str,
    args: argparse.Namespace,
) -> Any:
    return guide_rust.DiverseGuide(
        model,
        tokenizer,
        regex,
        gamma=args.gamma,
        completion_bias_eta=args.completion_bias_eta,
        path_frequency_estimator=args.path_frequency_estimator,
        **generation_kwargs_from_args(args),
    )


def run_one_repetition(
    *,
    shared: Any,
    guide_rust: Any,
    model: Any,
    tokenizer: Any,
    regex: str,
    prompt: str,
    args: argparse.Namespace,
    run_index: int,
    warmup: bool,
) -> dict[str, Any]:
    shared.set_generation_seed(args.seed)
    started = perf_counter()
    generator = build_generator(
        guide_rust=guide_rust,
        model=model,
        tokenizer=tokenizer,
        regex=regex,
        args=args,
    )
    build_seconds = perf_counter() - started
    generation = (
        shared.build_only_generation_metrics()
        if args.build_only
        else shared.run_until_target_tokens(
            generator=generator,
            tokenizer=tokenizer,
            prompt=prompt,
            max_tokens=args.max_tokens,
            target_generated_tokens=args.target_generated_tokens,
        )
    )
    return {
        "run_index": run_index,
        "warmup": warmup,
        "build_seconds": build_seconds,
        "method_dfa": shared.extract_method_dfa_metrics(generator),
        **generation,
    }


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = parse_args(argv)
    repo = args.repo.resolve()
    shared = load_shared_benchmark_module(repo)
    modules = load_current_modules(repo)
    implementation = validate_current_implementation(
        repo=repo,
        expected_commit=args.expected_current_commit,
        modules=modules,
        shared=shared,
    )

    grammar_source_repo = args.grammar_source_repo.resolve()
    generate_re = shared.load_generate_re(grammar_source_repo)
    if args.grammar not in generate_re.GRAMMAR_REGEX:
        raise ValueError(f"unknown grammar: {args.grammar}")
    regex = generate_re.GRAMMAR_REGEX[args.grammar]
    prompt = generate_re.GRAMMAR_PROMPT[args.grammar]

    model_ref = shared.resolve_model_ref(args.model, args.local_files_only)

    if args.build_only:
        model = SimpleNamespace(device=torch.device("cpu"))
        device = "not-loaded"
        dtype = None
        model_load_seconds = 0.0
        parameter_device = "not-loaded"
    else:
        cuda_available = torch.cuda.is_available()
        if args.require_cuda and not cuda_available:
            raise RuntimeError(
                "CUDA is required, but torch.cuda.is_available() is false."
            )
        device = "cuda" if cuda_available else "cpu"
        dtype = shared.select_dtype(args.torch_dtype, device)
        model_started = perf_counter()
        model = AutoModelForCausalLM.from_pretrained(
            model_ref,
            dtype=dtype,
            local_files_only=args.local_files_only,
            trust_remote_code=args.trust_remote_code,
        ).to(device)
        model.eval()
        model_load_seconds = perf_counter() - model_started
        parameter_device = str(next(model.parameters()).device)
        if args.require_cuda and not parameter_device.startswith("cuda"):
            raise RuntimeError(f"model is not on CUDA: {parameter_device}")

    tokenizer_started = perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(
        model_ref,
        local_files_only=args.local_files_only,
        trust_remote_code=args.trust_remote_code,
    )
    tokenizer_load_seconds = perf_counter() - tokenizer_started

    grammar_dfa = shared.build_grammar_dfa_metrics(modules["guide_rust"], regex)
    warmups = [
        run_one_repetition(
            shared=shared,
            guide_rust=modules["guide_rust"],
            model=model,
            tokenizer=tokenizer,
            regex=regex,
            prompt=prompt,
            args=args,
            run_index=i,
            warmup=True,
        )
        for i in range(args.warmup_runs)
    ]
    runs = [
        run_one_repetition(
            shared=shared,
            guide_rust=modules["guide_rust"],
            model=model,
            tokenizer=tokenizer,
            regex=regex,
            prompt=prompt,
            args=args,
            run_index=i,
            warmup=False,
        )
        for i in range(args.runs)
    ]

    result = {
        "schema_version": 1,
        "experiment": "strict_current_diverseguide_plus_benchmark",
        "implementation": implementation,
        "grammar_source_repo": str(grammar_source_repo),
        "grammar_source_commit": shared.git_commit(grammar_source_repo),
        "grammar": args.grammar,
        "regex": regex,
        "prompt": prompt,
        "grammar_dfa": grammar_dfa,
        "method_dfa": runs[0].get("method_dfa") if runs else None,
        "method": "diverse-guide-plus",
        "method_config": {
            "gamma": args.gamma,
            "completion_bias_eta": args.completion_bias_eta,
            "path_frequency_estimator": args.path_frequency_estimator,
            "engine": "current-rust-harmonic-eta",
        },
        "model": args.model,
        "model_loading": {
            "model_ref": model_ref,
            "device": device,
            "torch_dtype": (
                None if dtype is None else str(dtype).removeprefix("torch.")
            ),
            "model_load_seconds": model_load_seconds,
            "tokenizer_load_seconds": tokenizer_load_seconds,
            "model_parameter_device": parameter_device,
            "model_load_skipped": args.build_only,
            "torch_version": torch.__version__,
            "torch_cuda_version": torch.version.cuda,
        },
        "tokenizer": shared.tokenizer_metadata(tokenizer),
        "parameters": {
            "runs": args.runs,
            "warmup_runs": args.warmup_runs,
            "target_generated_tokens": args.target_generated_tokens,
            "max_tokens": args.max_tokens,
            "build_only": args.build_only,
            "seed": args.seed,
            "local_files_only": args.local_files_only,
            "require_cuda": args.require_cuda,
            "top_k": args.top_k,
            "top_p": args.top_p,
            "temperature": args.temperature,
            "trust_remote_code": args.trust_remote_code,
        },
        "warmups": warmups,
        "runs": runs,
        "summary": shared.summarize_runs(runs),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(f"[benchmark] wrote {args.output}")
    print(json.dumps(result["summary"], indent=2, sort_keys=True))
    return result


if __name__ == "__main__":
    main()
