"""Run a small gamma/eta grid search for diverse regex generation."""

import argparse
import csv
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from diverse_guide import DiverseGuide
from regex_dfa_guide import DiverseGuideDFA

try:
    from generate_re import GRAMMAR_PROMPT, GRAMMAR_REGEX, set_generation_seed
    from metrics_eval import compute_metrics
    from repro_results import build_metadata, write_json
except ModuleNotFoundError:
    from scripts.generate_re import GRAMMAR_PROMPT, GRAMMAR_REGEX, set_generation_seed
    from scripts.metrics_eval import compute_metrics
    from scripts.repro_results import build_metadata, write_json

DEFAULT_GRAMMARS = ["css-color", "json"]
DEFAULT_GAMMAS = [0.1, 0.15, 0.2, 0.3]
DEFAULT_ETAS = [0.0, 0.1, 0.15, 0.2, 0.3]
DEFAULT_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
DEFAULT_MAX_TOKENS = {
    "css-color": 18,
    "json": 54,
}


@dataclass
class GridConditionResult:
    grammar: str
    gamma: float
    eta: float
    sample_count: int
    accepted_samples: int
    rejected_samples: int
    attempted_samples: int
    validity_rate: float
    elapsed_seconds: float
    samples_per_second: float
    generated_tokens: int
    tokens_per_second: float
    average_length: float
    state_coverage: float
    transition_coverage: float
    path_coverage: float
    distinct_2gram_count: int
    distinct_3gram_count: int
    vendi_score: float
    samples: list[str]


def get_model_parameter_device(model: AutoModelForCausalLM) -> str | None:
    try:
        parameter = next(model.parameters())
    except (AttributeError, StopIteration, TypeError):
        return None
    return str(parameter.device)


def parse_float_list(raw: str) -> list[float]:
    return [float(item.strip()) for item in raw.split(",") if item.strip()]


def parse_str_list(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Grid-search gamma and completion-bias eta for diverse generation."
    )
    parser.add_argument(
        "--grammars",
        type=parse_str_list,
        default=DEFAULT_GRAMMARS,
        help="Comma-separated grammar list.",
    )
    parser.add_argument(
        "--gammas",
        type=parse_float_list,
        default=DEFAULT_GAMMAS,
        help="Comma-separated gamma values.",
    )
    parser.add_argument(
        "--etas",
        type=parse_float_list,
        default=DEFAULT_ETAS,
        help="Comma-separated completion-bias eta values.",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("-n", "--samples", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="Override grammar-specific max token defaults.",
    )
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--path-frequency-estimator",
        choices=["min", "harmonic"],
        default="harmonic",
    )
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/gamma_eta_grid"),
        help="Directory for grid_search.json and grid_search.csv.",
    )
    parser.add_argument(
        "--max-attempt-factor",
        type=int,
        default=5,
        help="Stop after samples * factor attempts if too many invalid outputs occur.",
    )
    args = parser.parse_args()
    if args.samples <= 0:
        parser.error("--samples must be positive")
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    if args.max_attempt_factor <= 0:
        parser.error("--max-attempt-factor must be positive")
    for grammar in args.grammars:
        if grammar not in GRAMMAR_REGEX:
            parser.error(f"Unknown grammar {grammar!r}")
    return args


def generation_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    kwargs: dict[str, Any] = {}
    if args.top_k is not None:
        kwargs["top_k"] = args.top_k
    if args.top_p is not None:
        kwargs["top_p"] = args.top_p
    if args.temperature is not None:
        kwargs["temperature"] = args.temperature
    return kwargs


def generate_condition(
    *,
    args: argparse.Namespace,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    grammar: str,
    gamma: float,
    eta: float,
    condition_index: int,
) -> GridConditionResult:
    regex = GRAMMAR_REGEX[grammar]
    prompt = GRAMMAR_PROMPT[grammar]
    pattern = re.compile("^(?:" + regex + ")$")
    max_tokens = args.max_tokens or DEFAULT_MAX_TOKENS.get(grammar, 18)
    max_attempts = args.samples * args.max_attempt_factor

    set_generation_seed(None if args.seed is None else args.seed + condition_index)
    generator = DiverseGuide(
        model,
        tokenizer,
        regex,
        gamma=gamma,
        completion_bias_eta=eta,
        path_frequency_estimator=args.path_frequency_estimator,
        **generation_kwargs(args),
    )

    samples: list[str] = []
    accepted = 0
    rejected = 0
    attempted = 0
    started = perf_counter()

    while accepted < args.samples and attempted < max_attempts:
        batch_size = min(args.batch_size, args.samples - accepted)
        batch_texts = generator.generate_batch(
            prompt, n=batch_size, max_tokens=max_tokens
        )
        attempted += len(batch_texts)
        for text in batch_texts:
            if pattern.match(text):
                samples.append(text)
                accepted += 1
            else:
                rejected += 1
            if accepted >= args.samples:
                break

    elapsed = perf_counter() - started
    token_count = sum(len(tokenizer.encode(sample)) for sample in samples)

    dfa = DiverseGuideDFA("(?:" + regex + ")$", 2**32 - 1, {})
    metrics = compute_metrics(dfa, samples)
    metric_values = metrics["metrics"]

    return GridConditionResult(
        grammar=grammar,
        gamma=gamma,
        eta=eta,
        sample_count=len(samples),
        accepted_samples=accepted,
        rejected_samples=rejected,
        attempted_samples=attempted,
        validity_rate=accepted / attempted if attempted else 0.0,
        elapsed_seconds=elapsed,
        samples_per_second=accepted / elapsed if elapsed > 0 else 0.0,
        generated_tokens=token_count,
        tokens_per_second=token_count / elapsed if elapsed > 0 else 0.0,
        average_length=metrics["average_length"],
        state_coverage=metric_values["state_coverage"],
        transition_coverage=metric_values["transition_coverage"],
        path_coverage=metric_values["path_coverage"],
        distinct_2gram_count=metric_values["distinct_2gram"][0],
        distinct_3gram_count=metric_values["distinct_3gram"][0],
        vendi_score=metric_values["vendi_score"],
        samples=samples,
    )


def write_csv(path: Path, rows: list[GridConditionResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        name for name in GridConditionResult.__dataclass_fields__ if name != "samples"
    ]
    with path.open("w", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            data = asdict(row)
            data.pop("samples")
            writer.writerow(data)


def print_summary(rows: list[GridConditionResult]) -> None:
    header = (
        "grammar gamma eta accepted/rejected seconds samples/s tokens/s "
        "state_cov trans_cov path_cov vendi"
    )
    print(header)
    for row in rows:
        print(
            f"{row.grammar} {row.gamma:.2f} {row.eta:.2f} "
            f"{row.accepted_samples}/{row.rejected_samples} "
            f"{row.elapsed_seconds:.2f} {row.samples_per_second:.3f} "
            f"{row.tokens_per_second:.2f} {100 * row.state_coverage:.2f}% "
            f"{100 * row.transition_coverage:.2f}% "
            f"{100 * row.path_coverage:.2f}% {row.vendi_score:.2f}"
        )


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    set_generation_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    print(f"Device: {device}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=dtype,
        local_files_only=args.local_files_only,
    ).to(device)
    model.eval()
    model_parameter_device = get_model_parameter_device(model)
    print(f"Model parameter device: {model_parameter_device}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        local_files_only=args.local_files_only,
    )

    rows: list[GridConditionResult] = []
    condition_index = 0
    for grammar in args.grammars:
        for gamma in args.gammas:
            for eta in args.etas:
                print(
                    f"Running {grammar}: gamma={gamma}, eta={eta}, "
                    f"n={args.samples}, batch_size={args.batch_size}"
                )
                row = generate_condition(
                    args=args,
                    model=model,
                    tokenizer=tokenizer,
                    grammar=grammar,
                    gamma=gamma,
                    eta=eta,
                    condition_index=condition_index,
                )
                rows.append(row)
                condition_index += 1
                write_json(
                    args.output_dir / "grid_search.json",
                    {
                        "schema_version": 1,
                        "experiment": "gamma_eta_grid_search",
                        "model": args.model,
                        "parameters": {
                            "grammars": args.grammars,
                            "gammas": args.gammas,
                            "etas": args.etas,
                            "samples": args.samples,
                            "batch_size": args.batch_size,
                            "path_frequency_estimator": args.path_frequency_estimator,
                            "seed": args.seed,
                            "top_k": args.top_k,
                            "top_p": args.top_p,
                            "temperature": args.temperature,
                            "device": device,
                            "model_parameter_device": model_parameter_device,
                        },
                        "metadata": build_metadata(),
                        "results": [asdict(item) for item in rows],
                    },
                )
                write_csv(args.output_dir / "grid_search.csv", rows)

    print_summary(rows)
    print(f"JSON written to: {args.output_dir / 'grid_search.json'}")
    print(f"CSV written to: {args.output_dir / 'grid_search.csv'}")


if __name__ == "__main__":
    main()
