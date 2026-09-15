"""Benchmark constrained-generation engineering overhead.

The benchmark separates the expensive phases that matter for implementation
optimization:

1. model/tokenizer loading;
2. constraint/generator build after the model is already loaded;
3. generation-time token latency after ``model.generate`` starts.

The default method presets match the main matrix:

* ``diverse-guide-plus``: optimized Rust/harmonic guide with eta enabled.
* ``diverse-guide``: Rust-only legacy min-estimator preset, without the current
  public API's Python fallback.
* ``outlines-baseline``: Outlines regex baseline.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import inspect
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, pstdev
from time import perf_counter
from typing import Any

import torch
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from transformers.generation.streamers import BaseStreamer
except ModuleNotFoundError:  # pragma: no cover - transformers layout fallback
    from transformers.generation.streamers import BaseStreamer


METHOD_PRESETS: dict[str, dict[str, Any]] = {
    "diverse-guide-plus": {
        "gamma": 0.3,
        "completion_bias_eta": 0.1,
        "path_frequency_estimator": "harmonic",
        "engine": "rust-current",
        "baseline_backend": "internal",
    },
    "diverse-guide": {
        "gamma": 0.16666666666666666,
        "completion_bias_eta": 0.0,
        "path_frequency_estimator": "min",
        "engine": "rust-only-legacy-min",
        "baseline_backend": "internal",
    },
    "outlines-baseline": {
        "gamma": 0.0,
        "completion_bias_eta": 0.0,
        "path_frequency_estimator": "min",
        "engine": "outlines",
        "baseline_backend": "outlines",
    },
    "mask-baseline": {
        "gamma": 0.0,
        "completion_bias_eta": 0.0,
        "path_frequency_estimator": "min",
        "engine": "internal-mask",
        "baseline_backend": "internal",
    },
}


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


@dataclass(frozen=True)
class MethodConfig:
    method: str
    gamma: float
    completion_bias_eta: float
    path_frequency_estimator: str
    engine: str
    baseline_backend: str
    ablation_component: str | None


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Benchmark guide build time and inter-token latency."
    )
    parser.add_argument("grammar", help="Grammar name from scripts/generate_re.py.")
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument(
        "--method",
        choices=sorted(METHOD_PRESETS),
        default="diverse-guide-plus",
    )
    parser.add_argument(
        "--repo-path",
        type=Path,
        default=repo_root,
        help="Repository root whose Python modules should be benchmarked.",
    )
    parser.add_argument(
        "--grammar-source-repo",
        type=Path,
        default=repo_root,
        help=(
            "Repository root used only for grammar definitions. This lets old "
            "worktrees be benchmarked on grammars added later."
        ),
    )
    parser.add_argument(
        "--runs",
        "-m",
        type=int,
        default=5,
        help="Measured repetitions to aggregate.",
    )
    parser.add_argument(
        "--warmup-runs",
        type=int,
        default=1,
        help="Warmup repetitions excluded from aggregate statistics.",
    )
    parser.add_argument("--max-tokens", type=int, default=24)
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--gamma", type=float, default=None)
    parser.add_argument("--completion-bias-eta", type=float, default=None)
    parser.add_argument(
        "--path-frequency-estimator",
        choices=["min", "harmonic"],
        default=None,
    )
    parser.add_argument(
        "--ablation-component",
        choices=["reward", "penalty", "range_scaling"],
        default=None,
    )
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Device for model loading. 'auto' chooses CUDA when visible.",
    )
    parser.add_argument(
        "--torch-dtype",
        choices=["auto", "float16", "bfloat16", "float32"],
        default="auto",
    )
    parser.add_argument(
        "--require-cuda",
        action="store_true",
        help="Fail if CUDA is unavailable or the loaded model is not on CUDA.",
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)
    if args.runs <= 0:
        parser.error("--runs must be positive")
    if args.warmup_runs < 0:
        parser.error("--warmup-runs cannot be negative")
    if args.max_tokens <= 0:
        parser.error("--max-tokens must be positive")
    if args.gamma is not None and args.gamma < 0.0:
        parser.error("--gamma must be non-negative")
    if args.completion_bias_eta is not None and args.completion_bias_eta < 0.0:
        parser.error("--completion-bias-eta must be non-negative")
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
        "scripts.eval_runtime",
    ]:
        sys.modules.pop(module_name, None)

    return {
        "diverse_guide": importlib.import_module("diverse_guide"),
        "guide_rust": importlib.import_module("diverse_guide.guide_rust"),
        "generate_re": importlib.import_module("scripts.generate_re"),
        "eval_runtime": importlib.import_module("scripts.eval_runtime"),
    }


def _load_generate_re_from_path(repo_path: Path) -> Any:
    module_path = repo_path / "scripts" / "generate_re.py"
    spec = importlib.util.spec_from_file_location(
        f"_benchmark_generate_re_{abs(hash(module_path.resolve()))}",
        module_path,
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load grammar source from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def resolve_grammar_definition(
    modules: dict[str, Any],
    args: argparse.Namespace,
) -> tuple[str, str, str]:
    grammar_modules = [modules["generate_re"]]
    source_repo = args.grammar_source_repo.resolve()
    target_repo = args.repo_path.resolve()
    if source_repo != target_repo:
        grammar_modules.append(_load_generate_re_from_path(source_repo))

    for grammar_module in grammar_modules:
        grammar_regex = getattr(grammar_module, "GRAMMAR_REGEX", {})
        if args.grammar not in grammar_regex:
            continue
        grammar_prompt = getattr(grammar_module, "GRAMMAR_PROMPT", {})
        prompt = args.prompt or grammar_prompt.get(args.grammar)
        if prompt is None:
            raise ValueError(f"missing prompt for grammar: {args.grammar}")
        return args.grammar, grammar_regex[args.grammar], prompt
    raise ValueError(f"unknown grammar: {args.grammar}")


def resolve_method_config(args: argparse.Namespace) -> MethodConfig:
    preset = METHOD_PRESETS[args.method]
    return MethodConfig(
        method=args.method,
        gamma=preset["gamma"] if args.gamma is None else args.gamma,
        completion_bias_eta=(
            preset["completion_bias_eta"]
            if args.completion_bias_eta is None
            else args.completion_bias_eta
        ),
        path_frequency_estimator=(
            preset["path_frequency_estimator"]
            if args.path_frequency_estimator is None
            else args.path_frequency_estimator
        ),
        engine=preset["engine"],
        baseline_backend=preset["baseline_backend"],
        ablation_component=args.ablation_component,
    )


def _resolve_model_ref(model: str, local_files_only: bool) -> str:
    if not local_files_only:
        return model
    return snapshot_download(model, local_files_only=True)


def _torch_dtype(name: str, device: str) -> torch.dtype:
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float32":
        return torch.float32
    return torch.float16 if device == "cuda" else torch.float32


def _select_device(args: argparse.Namespace) -> str:
    cuda_available = torch.cuda.is_available()
    if args.require_cuda and not cuda_available:
        raise RuntimeError("CUDA is required, but torch.cuda.is_available() is false.")
    if args.device == "auto":
        return "cuda" if cuda_available else "cpu"
    if args.device == "cuda" and not cuda_available:
        raise RuntimeError("--device cuda requested, but CUDA is unavailable.")
    return args.device


def _model_parameter_device(model) -> str | None:  # noqa: ANN001
    try:
        return str(next(model.parameters()).device)
    except StopIteration:
        return None


def _score_width(tokenizer: Any) -> int:
    vocab = getattr(tokenizer, "get_vocab", None)
    if callable(vocab):
        ids = [int(token_id) for token_id in vocab().values()]
        eos = getattr(tokenizer, "eos_token_id", None)
        if eos is not None:
            ids.append(int(eos))
        return max(ids) + 1 if ids else 0
    vocab_size = getattr(tokenizer, "vocab_size", None)
    if vocab_size is not None:
        return int(vocab_size)
    raise TypeError("tokenizer must provide get_vocab() or vocab_size")


def tokenizer_metadata(tokenizer: Any) -> dict[str, Any]:
    vocab = tokenizer.get_vocab()
    return {
        "class": type(tokenizer).__name__,
        "vocab_size": len(vocab),
        "score_width": _score_width(tokenizer),
        "eos_token_id": getattr(tokenizer, "eos_token_id", None),
        "pad_token_id": getattr(tokenizer, "pad_token_id", None),
    }


def generation_kwargs_from_args(args: argparse.Namespace) -> dict[str, Any]:
    kwargs = {}
    if args.top_k is not None:
        kwargs["top_k"] = args.top_k
    if args.top_p is not None:
        kwargs["top_p"] = args.top_p
    if args.temperature is not None:
        kwargs["temperature"] = args.temperature
    return kwargs


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


def build_grammar_dfa_metrics(
    modules: dict[str, Any],
    regex: str,
) -> dict[str, Any]:
    dfa_cls = modules["guide_rust"].DiverseGuideDFA
    started = perf_counter()
    dfa = dfa_cls(f"(?:{regex})$", 2**32 - 1, {})
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


def build_rust_only_processor_class(guide_rust_module: Any):
    class RustOnlyDiverseRegexLogitsProcessor(
        guide_rust_module.DiverseRegexLogitsProcessor
    ):
        """Benchmark wrapper that forces Rust adjustments for min/harmonic modes."""

        def __init__(
            self,
            regex_string: str,
            tokenizer: Any,
            *,
            gamma: float,
            completion_bias_eta: float,
            path_frequency_estimator: str,
            no_reward: bool = False,
            no_penalty: bool = False,
            no_range_scaling: bool = False,
        ):
            super().__init__(
                regex_string,
                tokenizer,
                gamma=gamma,
                completion_bias_eta=completion_bias_eta,
                path_frequency_estimator="harmonic",
                no_reward=no_reward,
                no_penalty=no_penalty,
                no_range_scaling=no_range_scaling,
            )
            self.path_frequency_estimator = path_frequency_estimator
            self._python_fallback = None

        def _compute_row_allowed_tokens_and_adjustments(
            self, guide_state: int, device: torch.device
        ) -> tuple[torch.LongTensor, torch.FloatTensor]:
            if guide_state == -1:
                allowed_tokens = [self.guide.eos_token_id]
                adjustments = self._compute_adjustments_from_counts([0], [1])
                return (
                    self._allowed_token_tensor(guide_state, allowed_tokens, device),
                    adjustments,
                )

            if not hasattr(self.dfa, "compute_adjustments_with_estimator"):
                raise RuntimeError(
                    "Rust-only benchmark requires "
                    "DiverseGuideDFA.compute_adjustments_with_estimator()."
                )
            allowed_tokens, adjustments = self.dfa.compute_adjustments_with_estimator(
                guide_state,
                self.no_reward,
                self.no_penalty,
                self.completion_bias_eta,
                self.path_frequency_estimator,
            )
            return (
                self._allowed_token_tensor(guide_state, allowed_tokens, device),
                torch.tensor(adjustments, dtype=torch.float),
            )

    return RustOnlyDiverseRegexLogitsProcessor


def supports_current_rust_adjustments(guide_rust_module: Any) -> bool:
    try:
        signature = inspect.signature(
            guide_rust_module.DiverseRegexLogitsProcessor.__init__
        )
    except (TypeError, ValueError):
        return False
    return (
        "completion_bias_eta" in signature.parameters
        and "path_frequency_estimator" in signature.parameters
        and hasattr(
            guide_rust_module.DiverseGuideDFA,
            "compute_adjustments_with_estimator",
        )
    )


def build_generator(
    *,
    modules: dict[str, Any],
    method_config: MethodConfig,
    model: Any,
    tokenizer: Any,
    regex: str,
    generation_kwargs: dict[str, Any],
):
    guide_rust = modules["guide_rust"]
    no_reward = method_config.ablation_component == "reward"
    no_penalty = method_config.ablation_component == "penalty"
    no_range_scaling = method_config.ablation_component == "range_scaling"

    if method_config.engine == "rust-current":
        return guide_rust.DiverseGuide(
            model,
            tokenizer,
            regex,
            gamma=method_config.gamma,
            completion_bias_eta=method_config.completion_bias_eta,
            path_frequency_estimator=method_config.path_frequency_estimator,
            ablation_component=method_config.ablation_component,
            **generation_kwargs,
        )
    if method_config.engine == "rust-only-legacy-min":
        if not supports_current_rust_adjustments(guide_rust):
            return guide_rust.DiverseGuide(
                model,
                tokenizer,
                regex,
                gamma=method_config.gamma,
                ablation_component=method_config.ablation_component,
                **generation_kwargs,
            )
        processor_cls = build_rust_only_processor_class(guide_rust)
        processor = processor_cls(
            regex,
            tokenizer,
            gamma=method_config.gamma,
            completion_bias_eta=method_config.completion_bias_eta,
            path_frequency_estimator=method_config.path_frequency_estimator,
            no_reward=no_reward,
            no_penalty=no_penalty,
            no_range_scaling=no_range_scaling,
        )
        return guide_rust.StatefulSequenceGeneratorAdapter(
            model,
            tokenizer,
            processor,
            **generation_kwargs,
        )
    if method_config.engine == "internal-mask":
        return guide_rust.baseline_regex(model, tokenizer, regex, **generation_kwargs)
    if method_config.engine == "outlines":
        outlines_kwargs = {**generation_kwargs, "do_sample": True}
        if getattr(tokenizer, "pad_token_id", None) is None:
            outlines_kwargs["pad_token_id"] = tokenizer.eos_token_id
        return modules["eval_runtime"].make_outlines_regex_generator(
            model,
            tokenizer,
            regex,
            **outlines_kwargs,
        )
    raise ValueError(f"unsupported method engine: {method_config.engine}")


def encode_prompt_len(tokenizer: Any, prompt: str) -> int:
    encoded = tokenizer(prompt, return_tensors="pt")
    return int(encoded["input_ids"].shape[1])


def generated_token_fallback_count(tokenizer: Any, text: str) -> int:
    try:
        return len(tokenizer.encode(text, add_special_tokens=False))
    except TypeError:
        return len(tokenizer.encode(text))


def run_generation_once(
    *,
    generator: Any,
    tokenizer: Any,
    prompt: str,
    max_tokens: int,
) -> dict[str, Any]:
    streamer = TokenTimingStreamer(
        prompt_token_count=encode_prompt_len(tokenizer, prompt)
    )
    started = perf_counter()
    streamer.start(started)
    streamer_supported = True
    try:
        text = generator(prompt, max_tokens=max_tokens, streamer=streamer)
    except TypeError as exc:
        if "streamer" not in str(exc):
            raise
        streamer_supported = False
        started = perf_counter()
        text = generator(prompt, max_tokens=max_tokens)
    elapsed_seconds = perf_counter() - started
    generated_tokens = streamer.generated_token_count
    if generated_tokens == 0:
        generated_tokens = generated_token_fallback_count(tokenizer, text)
    return {
        "streamer_supported": streamer_supported,
        "generation_seconds": elapsed_seconds,
        "generated_token_count": generated_tokens,
        "tokens_per_second": (
            generated_tokens / elapsed_seconds if elapsed_seconds > 0 else None
        ),
        "ttft_ms": streamer.first_token_delay_ms,
        "avg_inter_token_latency_ms": (
            mean(streamer.inter_token_delays_ms)
            if streamer.inter_token_delays_ms
            else None
        ),
        "inter_token_latency_std_ms": (
            pstdev(streamer.inter_token_delays_ms)
            if len(streamer.inter_token_delays_ms) > 1
            else 0.0
        ),
        "inter_token_count": len(streamer.inter_token_delays_ms),
        "output_char_length": len(text),
    }


def _summary(values: list[float | int | None]) -> dict[str, float] | None:
    numeric_values = [float(value) for value in values if value is not None]
    if not numeric_values:
        return None
    return {
        "count": float(len(numeric_values)),
        "mean": mean(numeric_values),
        "std": pstdev(numeric_values),
        "min": min(numeric_values),
        "max": max(numeric_values),
    }


def summarize_runs(runs: list[dict[str, Any]]) -> dict[str, Any]:
    fields = [
        "build_seconds",
        "generation_seconds",
        "generated_token_count",
        "tokens_per_second",
        "ttft_ms",
        "avg_inter_token_latency_ms",
        "inter_token_count",
        "output_char_length",
    ]
    return {field: _summary([run.get(field) for run in runs]) for field in fields}


def run_one_repetition(
    *,
    modules: dict[str, Any],
    method_config: MethodConfig,
    model: Any,
    tokenizer: Any,
    regex: str,
    prompt: str,
    args: argparse.Namespace,
    run_index: int,
    warmup: bool,
) -> dict[str, Any]:
    modules["generate_re"].set_generation_seed(args.seed)
    generation_kwargs = generation_kwargs_from_args(args)
    started = perf_counter()
    generator = build_generator(
        modules=modules,
        method_config=method_config,
        model=model,
        tokenizer=tokenizer,
        regex=regex,
        generation_kwargs=generation_kwargs,
    )
    build_seconds = perf_counter() - started
    generation = run_generation_once(
        generator=generator,
        tokenizer=tokenizer,
        prompt=prompt,
        max_tokens=args.max_tokens,
    )
    return {
        "run_index": run_index,
        "warmup": warmup,
        "build_seconds": build_seconds,
        "method_dfa": extract_method_dfa_metrics(generator),
        **generation,
    }


def load_model_and_tokenizer(
    args: argparse.Namespace,
) -> tuple[Any, Any, dict[str, Any]]:
    model_ref = _resolve_model_ref(args.model, args.local_files_only)
    device = _select_device(args)
    dtype = _torch_dtype(args.torch_dtype, device)
    load_started = perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        model_ref,
        dtype=dtype,
        local_files_only=args.local_files_only,
        trust_remote_code=args.trust_remote_code,
    ).to(device)
    model.eval()
    model_load_seconds = perf_counter() - load_started

    tokenizer_started = perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(
        model_ref,
        local_files_only=args.local_files_only,
        trust_remote_code=args.trust_remote_code,
    )
    tokenizer_load_seconds = perf_counter() - tokenizer_started

    parameter_device = _model_parameter_device(model)
    if args.require_cuda and not str(parameter_device).startswith("cuda"):
        raise RuntimeError(
            "CUDA is required, but loaded model parameters are on "
            f"{parameter_device}."
        )

    metadata = {
        "model_ref": model_ref,
        "device": device,
        "torch_dtype": str(dtype).removeprefix("torch."),
        "model_load_seconds": model_load_seconds,
        "tokenizer_load_seconds": tokenizer_load_seconds,
        "model_parameter_device": parameter_device,
    }
    return model, tokenizer, metadata


def build_result(
    *,
    args: argparse.Namespace,
    modules: dict[str, Any],
    method_config: MethodConfig,
    model_metadata: dict[str, Any],
    tokenizer: Any,
    grammar_dfa: dict[str, Any],
    prompt: str,
    regex: str,
    warmups: list[dict[str, Any]],
    runs: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "experiment": "generation_engineering_benchmark",
        "repo_path": str(args.repo_path.resolve()),
        "grammar_source_repo": str(args.grammar_source_repo.resolve()),
        "grammar": args.grammar,
        "model": args.model,
        "method": method_config.method,
        "method_config": {
            "engine": method_config.engine,
            "gamma": method_config.gamma,
            "completion_bias_eta": method_config.completion_bias_eta,
            "path_frequency_estimator": method_config.path_frequency_estimator,
            "baseline_backend": method_config.baseline_backend,
            "ablation_component": method_config.ablation_component,
            "python_fallback_used": False,
        },
        "regex": regex,
        "prompt": prompt,
        "parameters": {
            "runs": args.runs,
            "warmup_runs": args.warmup_runs,
            "max_tokens": args.max_tokens,
            "seed": args.seed,
            "top_k": args.top_k,
            "top_p": args.top_p,
            "temperature": args.temperature,
            "local_files_only": args.local_files_only,
            "trust_remote_code": args.trust_remote_code,
            "require_cuda": args.require_cuda,
        },
        "model_loading": model_metadata,
        "tokenizer": tokenizer_metadata(tokenizer),
        "grammar_dfa": grammar_dfa,
        "warmups": warmups,
        "runs": runs,
        "summary": summarize_runs(runs),
        "module_versions": {
            "diverse_guide": getattr(modules["diverse_guide"], "__version__", None),
        },
    }


def run_benchmark(argv: list[str] | None = None) -> dict[str, Any]:
    args = parse_args(argv)
    modules = _load_repo_modules(args.repo_path)
    _, regex, prompt = resolve_grammar_definition(modules, args)
    method_config = resolve_method_config(args)

    model, tokenizer, model_metadata = load_model_and_tokenizer(args)
    grammar_dfa = build_grammar_dfa_metrics(modules, regex)

    warmups = [
        run_one_repetition(
            modules=modules,
            method_config=method_config,
            model=model,
            tokenizer=tokenizer,
            regex=regex,
            prompt=prompt,
            args=args,
            run_index=run_index,
            warmup=True,
        )
        for run_index in range(args.warmup_runs)
    ]
    runs = [
        run_one_repetition(
            modules=modules,
            method_config=method_config,
            model=model,
            tokenizer=tokenizer,
            regex=regex,
            prompt=prompt,
            args=args,
            run_index=run_index,
            warmup=False,
        )
        for run_index in range(args.runs)
    ]
    return build_result(
        args=args,
        modules=modules,
        method_config=method_config,
        model_metadata=model_metadata,
        tokenizer=tokenizer,
        grammar_dfa=grammar_dfa,
        prompt=prompt,
        regex=regex,
        warmups=warmups,
        runs=runs,
    )


def main(argv: list[str] | None = None) -> dict[str, Any]:
    result = run_benchmark(argv)
    output = parse_args(argv).output
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2, sort_keys=True))
        print(f"[benchmark] wrote {output}")
    print(json.dumps(result["summary"], indent=2, sort_keys=True))
    return result


if __name__ == "__main__":
    main()
