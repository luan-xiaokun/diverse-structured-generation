"""Strict benchmark for the origin/master DiverseGuide implementation.

This script is intentionally narrow. It benchmarks only the old
``diverse_guide.guide_rust.DiverseGuide`` implementation from a separate
origin/master worktree, while grammar definitions may come from a newer
worktree so that later grammars such as semver and api-route can be included.

Run it with ``uv run`` from the old worktree so the old PyO3 extension in that
worktree's venv is loaded.
"""

from __future__ import annotations

import argparse
import ast
import importlib
import importlib.util
import inspect
import json
import random
import subprocess
import sys
from pathlib import Path
from statistics import mean, pstdev
from time import perf_counter
from types import SimpleNamespace
from typing import Any

import torch
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from transformers.generation.streamers import BaseStreamer
except ModuleNotFoundError:  # pragma: no cover - transformers layout fallback
    from transformers.generation.streamers import BaseStreamer


class TokenTimingStreamer(BaseStreamer):
    """Measure generated-token timing while ignoring prompt tokens."""

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
        description="Strict benchmark for origin/master DiverseGuide."
    )
    parser.add_argument("grammar")
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument(
        "--old-repo",
        type=Path,
        required=True,
        help="origin/master worktree whose implementation is benchmarked.",
    )
    parser.add_argument(
        "--grammar-source-repo",
        type=Path,
        default=repo_root,
        help="Repo used only for GRAMMAR_REGEX and GRAMMAR_PROMPT.",
    )
    parser.add_argument("--expected-old-commit", default=None)
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
    parser.add_argument("--gamma", type=float, default=1.0 / 6.0)
    parser.add_argument("--beta", type=float, default=3.0)
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
    if args.beta <= 0.0:
        parser.error("--beta must be positive")
    return args


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
    return SimpleNamespace(**values)


def load_old_modules(old_repo: Path) -> dict[str, Any]:
    for name in list(sys.modules):
        if name == "diverse_guide" or name.startswith("diverse_guide."):
            sys.modules.pop(name, None)
        if name == "regex_dfa_guide" or name.startswith("regex_dfa_guide."):
            sys.modules.pop(name, None)

    old_repo = old_repo.resolve()
    old_src = old_repo / "src"
    sys.path.insert(0, str(old_repo))
    sys.path.insert(0, str(old_src))

    guide_rust = importlib.import_module("diverse_guide.guide_rust")
    regex_dfa_guide = importlib.import_module("regex_dfa_guide")
    rs_module = importlib.import_module("regex_dfa_guide.regex_dfa_guide_rs")
    return {
        "guide_rust": guide_rust,
        "regex_dfa_guide": regex_dfa_guide,
        "rs_module": rs_module,
    }


def validate_old_implementation(
    *,
    old_repo: Path,
    expected_commit: str | None,
    modules: dict[str, Any],
) -> dict[str, Any]:
    old_repo = old_repo.resolve()
    commit = git_commit(old_repo)
    if expected_commit is not None and commit != expected_commit:
        raise RuntimeError(
            f"old repo commit mismatch: expected {expected_commit}, got {commit}"
        )

    guide_rust = modules["guide_rust"]
    regex_dfa_guide = modules["regex_dfa_guide"]
    rs_module = modules["rs_module"]

    guide_rust_file = Path(guide_rust.__file__).resolve()
    regex_pkg_file = Path(regex_dfa_guide.__file__).resolve()
    rs_file = Path(rs_module.__file__).resolve()

    checks = {
        "guide_rust_under_old_repo": path_is_under(guide_rust_file, old_repo),
        "regex_package_under_old_repo": path_is_under(regex_pkg_file, old_repo),
        "rust_extension_under_old_repo": path_is_under(rs_file, old_repo),
    }
    failed = [name for name, ok in checks.items() if not ok]
    if failed:
        raise RuntimeError(
            "old implementation path validation failed: "
            + ", ".join(failed)
            + (
                f"; guide_rust={guide_rust_file}; "
                f"regex_pkg={regex_pkg_file}; rs={rs_file}"
            )
        )

    processor_sig = inspect.signature(guide_rust.DiverseRegexLogitsProcessor.__init__)
    has_current_eta_arg = "completion_bias_eta" in processor_sig.parameters
    has_current_estimator_arg = "path_frequency_estimator" in processor_sig.parameters
    has_current_adjustment_api = hasattr(
        rs_module.DiverseGuideDFA,
        "compute_adjustments_with_estimator",
    )
    if has_current_eta_arg or has_current_estimator_arg or has_current_adjustment_api:
        raise RuntimeError(
            "loaded implementation looks like the current optimized API, not "
            "origin/master DiverseGuide"
        )

    return {
        "old_repo": str(old_repo),
        "old_git_commit": commit,
        "python_executable": sys.executable,
        "guide_rust_file": str(guide_rust_file),
        "regex_dfa_guide_file": str(regex_pkg_file),
        "rust_extension_file": str(rs_file),
        "processor_signature": str(processor_sig),
        "checks": checks,
        "has_current_eta_arg": has_current_eta_arg,
        "has_current_estimator_arg": has_current_estimator_arg,
        "has_current_adjustment_api": has_current_adjustment_api,
    }


def resolve_model_ref(model: str, local_files_only: bool) -> str:
    if not local_files_only:
        return model
    return snapshot_download(model, local_files_only=True)


def select_dtype(name: str, device: str) -> torch.dtype:
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float32":
        return torch.float32
    return torch.float16 if device == "cuda" else torch.float32


def generation_kwargs_from_args(args: argparse.Namespace) -> dict[str, Any]:
    kwargs: dict[str, Any] = {}
    if args.top_k is not None:
        kwargs["top_k"] = args.top_k
    if args.top_p is not None:
        kwargs["top_p"] = args.top_p
    if args.temperature is not None:
        kwargs["temperature"] = args.temperature
    return kwargs


def set_generation_seed(seed: int | None) -> None:
    if seed is None:
        return
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def score_width(tokenizer: Any) -> int:
    vocab = tokenizer.get_vocab()
    ids = [int(token_id) for token_id in vocab.values()]
    eos = getattr(tokenizer, "eos_token_id", None)
    if eos is not None:
        ids.append(int(eos))
    return max(ids) + 1 if ids else 0


def tokenizer_metadata(tokenizer: Any) -> dict[str, Any]:
    vocab = tokenizer.get_vocab()
    return {
        "class": type(tokenizer).__name__,
        "vocab_size": len(vocab),
        "score_width": score_width(tokenizer),
        "eos_token_id": getattr(tokenizer, "eos_token_id", None),
        "pad_token_id": getattr(tokenizer, "pad_token_id", None),
    }


def dfa_metrics_from_dfa(dfa: Any) -> dict[str, Any]:
    states = list(dfa.get_states())
    transitions = dfa.get_transitions()
    token_transition_count = 0
    token_transition_state_count = 0
    token_transition_missing_state_count = 0
    for state in states:
        try:
            token_transition_count += len(dfa.get_allowed_token_ids(state) or [])
            token_transition_state_count += 1
        except Exception:
            token_transition_missing_state_count += 1
    return {
        "state_count": len(states),
        "byte_transition_count": sum(len(edges) for edges in transitions.values()),
        "token_transition_count": token_transition_count,
        "token_transition_state_count": token_transition_state_count,
        "token_transition_missing_state_count": token_transition_missing_state_count,
    }


def build_grammar_dfa_metrics(guide_rust: Any, regex: str) -> dict[str, Any]:
    started = perf_counter()
    dfa = guide_rust.DiverseGuideDFA(f"(?:{regex})$", 2**32 - 1, {})
    return {
        **dfa_metrics_from_dfa(dfa),
        "build_seconds": perf_counter() - started,
    }


def extract_method_dfa_metrics(generator: Any) -> dict[str, Any] | None:
    processor = getattr(generator, "logits_processor", None)
    dfa = getattr(processor, "dfa", None)
    if dfa is None:
        return None
    return dfa_metrics_from_dfa(dfa)


def encode_prompt_len(tokenizer: Any, prompt: str) -> int:
    encoded = tokenizer(prompt, return_tensors="pt")
    return int(encoded["input_ids"].shape[1])


def generated_token_fallback_count(tokenizer: Any, text: str) -> int:
    try:
        return len(tokenizer.encode(text, add_special_tokens=False))
    except TypeError:
        return len(tokenizer.encode(text))


def synchronize_if_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def run_generation_call(
    *,
    generator: Any,
    tokenizer: Any,
    prompt: str,
    max_tokens: int,
) -> dict[str, Any]:
    streamer = TokenTimingStreamer(
        prompt_token_count=encode_prompt_len(tokenizer, prompt)
    )
    synchronize_if_cuda()
    started = perf_counter()
    streamer.start(started)
    text = generator(prompt, max_tokens=max_tokens, streamer=streamer)
    synchronize_if_cuda()
    elapsed_seconds = perf_counter() - started
    generated_tokens = streamer.generated_token_count
    if generated_tokens == 0:
        generated_tokens = generated_token_fallback_count(tokenizer, text)
    return {
        "generation_seconds": elapsed_seconds,
        "generated_token_count": generated_tokens,
        "ttft_ms": streamer.first_token_delay_ms,
        "inter_token_delays_ms": streamer.inter_token_delays_ms,
        "inter_token_count": len(streamer.inter_token_delays_ms),
        "output_char_length": len(text),
    }


def percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    idx = round((len(ordered) - 1) * pct)
    return ordered[idx]


def summarize_numeric(values: list[float | int | None]) -> dict[str, float] | None:
    numeric = [float(value) for value in values if value is not None]
    if not numeric:
        return None
    return {
        "count": float(len(numeric)),
        "mean": mean(numeric),
        "std": pstdev(numeric),
        "min": min(numeric),
        "max": max(numeric),
    }


def summarize_intervals(values: list[float]) -> dict[str, float | None]:
    summary = summarize_numeric(values) or {
        "count": 0.0,
        "mean": None,
        "std": None,
        "min": None,
        "max": None,
    }
    summary.update(
        {
            "p50": percentile(values, 0.50),
            "p90": percentile(values, 0.90),
            "p95": percentile(values, 0.95),
        }
    )
    return summary


def build_only_generation_metrics() -> dict[str, Any]:
    return {
        "generation_seconds": None,
        "generated_token_count": None,
        "call_count": None,
        "tokens_per_second": None,
        "ttft_ms": summarize_intervals([]),
        "inter_token_latency_ms": summarize_intervals([]),
        "calls": [],
    }


def run_until_target_tokens(
    *,
    generator: Any,
    tokenizer: Any,
    prompt: str,
    max_tokens: int,
    target_generated_tokens: int,
) -> dict[str, Any]:
    total_generated = 0
    total_seconds = 0.0
    calls = []
    ttft_values: list[float] = []
    inter_token_delays: list[float] = []

    while total_generated < target_generated_tokens:
        call = run_generation_call(
            generator=generator,
            tokenizer=tokenizer,
            prompt=prompt,
            max_tokens=max_tokens,
        )
        generated = int(call["generated_token_count"])
        if generated <= 0:
            raise RuntimeError("generation made no token progress")
        total_generated += generated
        total_seconds += float(call["generation_seconds"])
        if call["ttft_ms"] is not None:
            ttft_values.append(float(call["ttft_ms"]))
        inter_token_delays.extend(float(x) for x in call["inter_token_delays_ms"])
        calls.append(
            {
                "generation_seconds": call["generation_seconds"],
                "generated_token_count": generated,
                "ttft_ms": call["ttft_ms"],
                "inter_token_count": call["inter_token_count"],
                "output_char_length": call["output_char_length"],
            }
        )

    return {
        "generation_seconds": total_seconds,
        "generated_token_count": total_generated,
        "call_count": len(calls),
        "tokens_per_second": (
            total_generated / total_seconds if total_seconds > 0.0 else None
        ),
        "ttft_ms": summarize_intervals(ttft_values),
        "inter_token_latency_ms": summarize_intervals(inter_token_delays),
        "calls": calls,
    }


def build_generator(
    *,
    guide_rust: Any,
    model: Any,
    tokenizer: Any,
    regex: str,
    gamma: float,
    beta: float,
    generation_kwargs: dict[str, Any],
) -> Any:
    if hasattr(guide_rust, "DiverseGuide"):
        return guide_rust.DiverseGuide(
            model,
            tokenizer,
            regex,
            gamma=gamma,
            beta=beta,
            **generation_kwargs,
        )
    return guide_rust.diverse_regex(
        model,
        tokenizer,
        regex,
        gamma=gamma,
        beta=beta,
        **generation_kwargs,
    )


def run_one_repetition(
    *,
    guide_rust: Any,
    model: Any,
    tokenizer: Any,
    regex: str,
    prompt: str,
    args: argparse.Namespace,
    run_index: int,
    warmup: bool,
) -> dict[str, Any]:
    set_generation_seed(args.seed)
    generation_kwargs = generation_kwargs_from_args(args)
    started = perf_counter()
    generator = build_generator(
        guide_rust=guide_rust,
        model=model,
        tokenizer=tokenizer,
        regex=regex,
        gamma=args.gamma,
        beta=args.beta,
        generation_kwargs=generation_kwargs,
    )
    build_seconds = perf_counter() - started
    generation = (
        build_only_generation_metrics()
        if args.build_only
        else run_until_target_tokens(
            generator=generator,
            tokenizer=tokenizer,
            prompt=prompt,
            max_tokens=args.max_tokens,
            target_generated_tokens=args.target_generated_tokens,
        )
    )
    method_dfa = extract_method_dfa_metrics(generator)
    return {
        "run_index": run_index,
        "warmup": warmup,
        "build_seconds": build_seconds,
        "method_dfa": method_dfa,
        **generation,
    }


def summarize_runs(runs: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "build_seconds": summarize_numeric([run["build_seconds"] for run in runs]),
        "generation_seconds": summarize_numeric(
            [run["generation_seconds"] for run in runs]
        ),
        "generated_token_count": summarize_numeric(
            [run["generated_token_count"] for run in runs]
        ),
        "call_count": summarize_numeric([run["call_count"] for run in runs]),
        "tokens_per_second": summarize_numeric(
            [run["tokens_per_second"] for run in runs]
        ),
        "avg_inter_token_latency_ms": summarize_numeric(
            [run["inter_token_latency_ms"]["mean"] for run in runs]
        ),
        "inter_token_latency_p50_ms": summarize_numeric(
            [run["inter_token_latency_ms"]["p50"] for run in runs]
        ),
        "inter_token_latency_p90_ms": summarize_numeric(
            [run["inter_token_latency_ms"]["p90"] for run in runs]
        ),
        "inter_token_latency_p95_ms": summarize_numeric(
            [run["inter_token_latency_ms"]["p95"] for run in runs]
        ),
        "ttft_ms": summarize_numeric([run["ttft_ms"]["mean"] for run in runs]),
        "total_inter_token_count": sum(
            int(run["inter_token_latency_ms"]["count"]) for run in runs
        ),
    }


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = parse_args(argv)
    old_repo = args.old_repo.resolve()
    grammar_source_repo = args.grammar_source_repo.resolve()

    modules = load_old_modules(old_repo)
    implementation = validate_old_implementation(
        old_repo=old_repo,
        expected_commit=args.expected_old_commit,
        modules=modules,
    )
    generate_re = load_generate_re(grammar_source_repo)
    if args.grammar not in generate_re.GRAMMAR_REGEX:
        raise ValueError(f"unknown grammar: {args.grammar}")
    regex = generate_re.GRAMMAR_REGEX[args.grammar]
    prompt = generate_re.GRAMMAR_PROMPT[args.grammar]

    model_ref = resolve_model_ref(args.model, args.local_files_only)

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
        dtype = select_dtype(args.torch_dtype, device)
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

    grammar_dfa = build_grammar_dfa_metrics(modules["guide_rust"], regex)

    warmups = [
        run_one_repetition(
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
        "experiment": "strict_origin_master_diverseguide_benchmark",
        "implementation": implementation,
        "grammar_source_repo": str(grammar_source_repo),
        "grammar_source_commit": git_commit(grammar_source_repo),
        "grammar": args.grammar,
        "regex": regex,
        "prompt": prompt,
        "grammar_dfa": grammar_dfa,
        "method_dfa": runs[0].get("method_dfa") if runs else None,
        "method": "origin-master-diverse-guide",
        "method_config": {
            "gamma": args.gamma,
            "beta": args.beta,
            "engine": "origin-master-rust-python",
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
        "tokenizer": tokenizer_metadata(tokenizer),
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
        "summary": summarize_runs(runs),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(f"[benchmark] wrote {args.output}")
    print(json.dumps(result["summary"], indent=2, sort_keys=True))
    return result


if __name__ == "__main__":
    main()
