"""Benchmark runtime, TTFT, and average inter-token latency for regex generation."""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path
from statistics import mean, pstdev
from time import perf_counter
from typing import Any

import torch
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from transformers.generation.streamers import BaseStreamer
except ModuleNotFoundError:
    from transformers.generation.streamers import BaseStreamer


class TokenTimingStreamer(BaseStreamer):
    """Measure prompt-skipping generated-token timings from HF generate()."""

    def __init__(self, prompt_token_count: int):
        self.prompt_token_count = prompt_token_count
        self.started_at = 0.0
        self._seen_total = 0
        self.generated_token_count = 0
        self.first_token_delay_ms: float | None = None
        self.inter_token_delays_ms: list[float] = []
        self._last_generated_at: float | None = None

    def start(self, started_at: float) -> None:
        self.started_at = started_at
        self._seen_total = 0
        self.generated_token_count = 0
        self.first_token_delay_ms = None
        self.inter_token_delays_ms = []
        self._last_generated_at = None

    def put(self, value) -> None:  # noqa: ANN001
        if isinstance(value, torch.Tensor):
            token_ids = value.reshape(-1).tolist()
        else:
            token_ids = list(value)

        for _ in token_ids:
            self._seen_total += 1
            if self._seen_total <= self.prompt_token_count:
                continue

            now = perf_counter()
            self.generated_token_count += 1
            if self.first_token_delay_ms is None:
                self.first_token_delay_ms = (now - self.started_at) * 1000.0
            elif self._last_generated_at is not None:
                self.inter_token_delays_ms.append(
                    (now - self._last_generated_at) * 1000.0
                )
            self._last_generated_at = now

    def end(self) -> None:
        return None


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Benchmark regex runtime, TTFT, and inter-token latency."
    )
    parser.add_argument(
        "--repo-path",
        type=Path,
        default=repo_root,
        help="Repository root containing src/diverse_guide and scripts/generate_re.py.",
    )
    parser.add_argument("grammar", help="Grammar to benchmark.")
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument(
        "--mode",
        choices=["diverse", "baseline_mask"],
        default="diverse",
    )
    parser.add_argument("-n", type=int, default=100, help="Target generated tokens.")
    parser.add_argument(
        "--max-tokens", type=int, default=20, help="Max new tokens per call."
    )
    parser.add_argument(
        "--runs", type=int, default=5, help="Measured runs to include in statistics."
    )
    parser.add_argument(
        "--warmup-runs", type=int, default=1, help="Warmup runs to execute first."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed reset before each run to keep outputs comparable.",
    )
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument(
        "--require-cuda",
        action="store_true",
        help="Fail unless torch sees CUDA and the loaded model is on a CUDA device.",
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)
    if args.n <= 0:
        parser.error("-n must be positive")
    if args.max_tokens <= 0:
        parser.error("--max-tokens must be positive")
    if args.runs <= 0:
        parser.error("--runs must be positive")
    if args.warmup_runs < 0:
        parser.error("--warmup-runs cannot be negative")
    return args


def _load_repo_modules(repo_path: Path) -> dict[str, Any]:
    repo_root = str(repo_path.resolve())
    src_root = str((repo_path / "src").resolve())
    if src_root not in sys.path:
        sys.path.insert(0, src_root)
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    for module_name in [
        "diverse_guide",
        "diverse_guide.guide_rust",
        "scripts.generate_re",
    ]:
        sys.modules.pop(module_name, None)

    diverse_guide = importlib.import_module("diverse_guide")
    guide_rust = importlib.import_module("diverse_guide.guide_rust")
    generate_re = importlib.import_module("scripts.generate_re")
    return {
        "diverse_guide": diverse_guide,
        "guide_rust": guide_rust,
        "generate_re": generate_re,
    }


def _resolve_model_ref(model: str, local_files_only: bool) -> str:
    if not local_files_only:
        return model
    return snapshot_download(model, local_files_only=True)


def _build_generator(
    *,
    modules: dict[str, Any],
    model,
    tokenizer,
    regex: str,
    mode: str,
):
    guide_rust = modules["guide_rust"]
    if mode == "baseline_mask":
        return modules["diverse_guide"].baseline_regex(model, tokenizer, regex)
    if mode == "diverse":
        return modules["diverse_guide"].DiverseGuide(model, tokenizer, regex)
    raise ValueError(f"unsupported mode: {mode}")


def _run_once(
    *,
    modules: dict[str, Any],
    model,
    tokenizer,
    grammar: str,
    mode: str,
    n: int,
    max_tokens: int,
    seed: int | None,
) -> dict[str, Any]:
    generate_re = modules["generate_re"]
    regex = generate_re.GRAMMAR_REGEX[grammar]
    prompt = generate_re.GRAMMAR_PROMPT[grammar]
    generate_re.set_generation_seed(seed)

    generator = _build_generator(
        modules=modules,
        model=model,
        tokenizer=tokenizer,
        regex=regex,
        mode=mode,
    )
    prompt_token_count = tokenizer(prompt, return_tensors="pt")["input_ids"].shape[1]

    total_generated_tokens = 0
    total_elapsed_seconds = 0.0
    ttft_ms: list[float] = []
    inter_token_ms: list[float] = []
    call_count = 0

    while total_generated_tokens < n:
        streamer = TokenTimingStreamer(prompt_token_count=prompt_token_count)
        started = perf_counter()
        streamer.start(started)
        generated_text = generator(prompt, max_tokens=max_tokens, streamer=streamer)
        elapsed_seconds = perf_counter() - started
        generated_tokens = len(tokenizer.encode(generated_text))

        total_generated_tokens += generated_tokens
        total_elapsed_seconds += elapsed_seconds
        call_count += 1
        if streamer.first_token_delay_ms is not None:
            ttft_ms.append(streamer.first_token_delay_ms)
        inter_token_ms.extend(streamer.inter_token_delays_ms)

    tokens_per_second = total_generated_tokens / total_elapsed_seconds
    return {
        "tokens_generated": total_generated_tokens,
        "seconds": total_elapsed_seconds,
        "tokens_per_second": tokens_per_second,
        "call_count": call_count,
        "ttft_ms_mean": mean(ttft_ms) if ttft_ms else None,
        "avg_inter_token_latency_ms": mean(inter_token_ms) if inter_token_ms else None,
    }


def _summary(values: list[float | None]) -> dict[str, float] | None:
    numeric_values = [float(value) for value in values if value is not None]
    if not numeric_values:
        return None
    return {
        "mean": mean(numeric_values),
        "std": pstdev(numeric_values),
        "min": min(numeric_values),
        "max": max(numeric_values),
    }


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = parse_args(argv)
    modules = _load_repo_modules(args.repo_path)
    if args.grammar not in modules["generate_re"].GRAMMAR_REGEX:
        raise ValueError(f"unknown grammar: {args.grammar}")

    model_ref = _resolve_model_ref(args.model, args.local_files_only)
    cuda_available_initial = torch.cuda.is_available()
    device = "cuda" if cuda_available_initial else "cpu"
    if args.require_cuda and not cuda_available_initial:
        raise RuntimeError(
            "CUDA is required for this benchmark run, but "
            "torch.cuda.is_available() returned False."
        )
    dtype = torch.float16 if device == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_ref,
        dtype=dtype,
        local_files_only=args.local_files_only,
    ).to(device)
    model.eval()
    model_device = str(next(model.parameters()).device)
    if args.require_cuda and not next(model.parameters()).is_cuda:
        raise RuntimeError(
            "CUDA is required for this benchmark run, but the loaded model "
            f"parameters are on {model_device}."
        )
    tokenizer = AutoTokenizer.from_pretrained(
        model_ref,
        local_files_only=args.local_files_only,
    )

    warmups = [
        _run_once(
            modules=modules,
            model=model,
            tokenizer=tokenizer,
            grammar=args.grammar,
            mode=args.mode,
            n=args.n,
            max_tokens=args.max_tokens,
            seed=args.seed,
        )
        for _ in range(args.warmup_runs)
    ]
    runs = [
        _run_once(
            modules=modules,
            model=model,
            tokenizer=tokenizer,
            grammar=args.grammar,
            mode=args.mode,
            n=args.n,
            max_tokens=args.max_tokens,
            seed=args.seed,
        )
        for _ in range(args.runs)
    ]

    result = {
        "schema_version": 1,
        "experiment": "runtime_latency_benchmark",
        "repo_path": str(args.repo_path.resolve()),
        "grammar": args.grammar,
        "model": args.model,
        "mode": args.mode,
        "python_executable": sys.executable,
        "torch_cuda_available_initial": cuda_available_initial,
        "parameters": {
            "n": args.n,
            "max_tokens": args.max_tokens,
            "runs": args.runs,
            "warmup_runs": args.warmup_runs,
            "seed": args.seed,
            "local_files_only": args.local_files_only,
            "require_cuda": args.require_cuda,
        },
        "device": device,
        "model_parameter_device": model_device,
        "warmups": warmups,
        "runs": runs,
        "summary": {
            "tokens_per_second": _summary([run["tokens_per_second"] for run in runs]),
            "ttft_ms": _summary([run["ttft_ms_mean"] for run in runs]),
            "avg_inter_token_latency_ms": _summary(
                [run["avg_inter_token_latency_ms"] for run in runs]
            ),
        },
    }

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True))
    print(json.dumps(result["summary"], indent=2, sort_keys=True))
    return result


if __name__ == "__main__":
    main()
