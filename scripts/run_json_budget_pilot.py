"""Run the Phase 0 JSON max-token budget pilot.

This script intentionally keeps a narrow scope:

* model: one local Hugging Face causal LM
* grammar: json
* methods: DG baseline and one representative DG+ candidate
* budgets: a small max-token grid

Unlike ``scripts/generate_re.py``, rejected samples are retained with a coarse
reason so we can distinguish likely max-token exhaustion from other regex
mismatches. Global diversity counters are committed only after the decoded text
passes the regex validator.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from time import perf_counter
from typing import Any

import numpy as np
import torch
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.logits_process import LogitsProcessorList

try:
    from generate_re import GRAMMAR_PROMPT, GRAMMAR_REGEX
except ModuleNotFoundError:
    from scripts.generate_re import GRAMMAR_PROMPT, GRAMMAR_REGEX

from diverse_guide.guide_rust import DiverseRegexLogitsProcessor


@dataclass(frozen=True)
class MethodConfig:
    name: str
    gamma: float
    completion_bias_eta: float
    path_frequency_estimator: str


METHODS = (
    MethodConfig(
        name="dg_baseline_min_eta0",
        gamma=1.0 / 6.0,
        completion_bias_eta=0.0,
        path_frequency_estimator="min",
    ),
    MethodConfig(
        name="dgplus_harmonic_eta0.1_gamma0.3",
        gamma=0.3,
        completion_bias_eta=0.1,
        path_frequency_estimator="harmonic",
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run JSON budget pilot.")
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument(
        "--n",
        type=int,
        default=50,
        help="Number of generation attempts per cell.",
    )
    parser.add_argument(
        "--budgets",
        type=int,
        nargs="+",
        default=[54, 72, 96, 128],
        help="max_new_tokens budgets to test.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-attempts", type=int, default=50)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("results/extension/refinement_budget_pilot_n50"),
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data/extension/refinement_budget_pilot_n50"),
    )
    args = parser.parse_args()
    if args.n <= 0:
        parser.error("--n must be positive")
    if args.max_attempts <= 0:
        parser.error("--max-attempts must be positive")
    if any(budget <= 0 for budget in args.budgets):
        parser.error("--budgets must all be positive")
    return args


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def model_slug(model: str) -> str:
    return model.split("/")[-1].lower()


def get_model_parameter_device(model: Any) -> str | None:
    try:
        return str(next(model.parameters()).device)
    except (AttributeError, StopIteration, TypeError):
        return None


def token_count(tokenizer: Any, text: str) -> int:
    try:
        return len(tokenizer.encode(text, add_special_tokens=False))
    except TypeError:
        return len(tokenizer.encode(text))


def classify_rejection(
    *,
    generated_token_ids: list[int],
    eos_token_id: int | None,
    max_tokens: int,
) -> str:
    if len(generated_token_ids) >= max_tokens and (
        eos_token_id is None or eos_token_id not in generated_token_ids
    ):
        return "likely_max_token_exhaustion"
    return "regex_mismatch_before_budget"


def commit_valid_sample(
    processor: DiverseRegexLogitsProcessor,
    generated_token_ids: list[int],
    text: str,
) -> None:
    if hasattr(processor, "update_generated_token_ids"):
        processor.update_generated_token_ids(
            torch.tensor(generated_token_ids, dtype=torch.long)
        )
    if hasattr(processor, "update_generated_content"):
        processor.update_generated_content(text)


def generate_cell(
    *,
    model: Any,
    tokenizer: Any,
    regex: str,
    prompt: str,
    pattern: re.Pattern[str],
    method: MethodConfig,
    budget: int,
    attempt_count: int,
    max_attempts: int,
    seed: int,
    progress: bool,
) -> dict[str, Any]:
    set_seed(seed)
    processor = DiverseRegexLogitsProcessor(
        regex,
        tokenizer,
        gamma=method.gamma,
        completion_bias_eta=method.completion_bias_eta,
        path_frequency_estimator=method.path_frequency_estimator,
    )

    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    if pad_token_id is None:
        pad_token_id = getattr(tokenizer, "eos_token_id", None)

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    prompt_len = int(inputs["input_ids"].shape[1])

    accepted: list[str] = []
    accepted_records: list[dict[str, Any]] = []
    rejected_records: list[dict[str, Any]] = []
    started = perf_counter()
    planned_attempts = min(attempt_count, max_attempts)

    pbar = None
    if progress:
        pbar = tqdm(
            total=planned_attempts,
            desc=f"{method.name}:json:max{budget}",
            unit="attempt",
            dynamic_ncols=True,
        )

    try:
        attempts = 0
        while attempts < planned_attempts:
            attempts += 1
            processor.dfa.reset_local_state_counter()
            processor.reset()
            with torch.no_grad():
                output_ids = model.generate(
                    **inputs,
                    logits_processor=LogitsProcessorList([processor]),
                    do_sample=True,
                    max_new_tokens=budget,
                    pad_token_id=pad_token_id,
                )
            generated_ids = [int(x) for x in output_ids[0, prompt_len:].tolist()]
            text = tokenizer.decode(generated_ids, skip_special_tokens=True)
            generated_token_count = len(generated_ids)
            decoded_token_count = token_count(tokenizer, text)
            is_valid = pattern.fullmatch(text) is not None
            if is_valid:
                commit_valid_sample(processor, generated_ids, text)
                accepted.append(text)
                accepted_records.append(
                    {
                        "sample_index": len(accepted) - 1,
                        "attempt_index": attempts - 1,
                        "text": text,
                        "generated_token_count": generated_token_count,
                        "decoded_token_count": decoded_token_count,
                        "char_length": len(text),
                    }
                )
                if pbar is not None:
                    pbar.update(1)
            else:
                rejected_records.append(
                    {
                        "attempt_index": attempts - 1,
                        "reason": classify_rejection(
                            generated_token_ids=generated_ids,
                            eos_token_id=getattr(tokenizer, "eos_token_id", None),
                            max_tokens=budget,
                        ),
                        "text": text,
                        "generated_token_count": generated_token_count,
                        "decoded_token_count": decoded_token_count,
                        "char_length": len(text),
                    }
                )
            if pbar is not None:
                if not is_valid:
                    pbar.update(1)
                pbar.set_postfix(
                    {
                        "accepted": len(accepted),
                        "rejected": len(rejected_records),
                        "attempts": attempts,
                    }
                )
    finally:
        if pbar is not None:
            pbar.close()

    elapsed_seconds = perf_counter() - started
    return {
        "schema_version": 1,
        "experiment": "json_budget_pilot",
        "grammar": "json",
        "regex": regex,
        "prompt": prompt,
        "method": method.name,
        "model": None,
        "parameters": {
            "attempt_count": attempt_count,
            "max_tokens": budget,
            "max_attempts": max_attempts,
            "seed": seed,
            "gamma": method.gamma,
            "completion_bias_eta": method.completion_bias_eta,
            "path_frequency_estimator": method.path_frequency_estimator,
        },
        "samples": accepted,
        "accepted_records": accepted_records,
        "rejected_records": rejected_records,
        "accepted_samples": len(accepted),
        "rejected_samples": len(rejected_records),
        "attempted_batches": len(accepted) + len(rejected_records),
        "elapsed_seconds": elapsed_seconds,
        "complete": attempts >= planned_attempts,
    }


def summarize_cell(cell: dict[str, Any], tokenizer: Any) -> dict[str, Any]:
    accepted = cell["accepted_records"]
    rejected = cell["rejected_records"]
    attempts = cell["attempted_batches"]
    elapsed = cell["elapsed_seconds"]
    exhaustion = [
        item for item in rejected if item["reason"] == "likely_max_token_exhaustion"
    ]

    def avg(records: list[dict[str, Any]], key: str) -> float | None:
        if not records:
            return None
        return mean(float(item[key]) for item in records)

    return {
        "method": cell["method"],
        "model": cell["model"],
        "grammar": "json",
        "max_tokens": cell["parameters"]["max_tokens"],
        "gamma": cell["parameters"]["gamma"],
        "completion_bias_eta": cell["parameters"]["completion_bias_eta"],
        "path_frequency_estimator": cell["parameters"]["path_frequency_estimator"],
        "accepted_samples": cell["accepted_samples"],
        "rejected_samples": cell["rejected_samples"],
        "attempts": attempts,
        "complete": cell["complete"],
        "validity_rate": cell["accepted_samples"] / attempts if attempts else 0.0,
        "rejection_rate": cell["rejected_samples"] / attempts if attempts else 0.0,
        "max_token_exhaustion_rejections": len(exhaustion),
        "max_token_exhaustion_rate": len(exhaustion) / attempts if attempts else 0.0,
        "elapsed_seconds": elapsed,
        "accepted_per_second": cell["accepted_samples"] / elapsed
        if elapsed > 0.0
        else None,
        "attempts_per_second": attempts / elapsed if elapsed > 0.0 else None,
        "avg_accepted_generated_tokens": avg(accepted, "generated_token_count"),
        "avg_accepted_decoded_tokens": avg(accepted, "decoded_token_count"),
        "avg_accepted_chars": avg(accepted, "char_length"),
        "avg_rejected_generated_tokens": avg(rejected, "generated_token_count"),
        "avg_rejected_decoded_tokens": avg(rejected, "decoded_token_count"),
        "eos_token_id": getattr(tokenizer, "eos_token_id", None),
    }


def write_summary(output_root: Path, rows: list[dict[str, Any]]) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    metrics_json = output_root / "json_budget_pilot_metrics.json"
    metrics_csv = output_root / "json_budget_pilot_metrics.csv"
    summary_md = output_root / "json_budget_pilot_summary.md"
    metrics_json.write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n")
    with metrics_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# JSON Budget Pilot",
        "",
        (
            "| Method | Max tokens | Accepted | Rejected | Rejection rate | "
            "Max-token exhaustion rate | Accepted/s | Avg accepted toks | "
            "Avg rejected toks |"
        ),
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {method} | {max_tokens} | {accepted} | {rejected} | {reject:.3f} | "
            "{exhaust:.3f} | {aps:.3f} | {acc_tok:.1f} | {rej_tok} |".format(
                method=row["method"],
                max_tokens=row["max_tokens"],
                accepted=row["accepted_samples"],
                rejected=row["rejected_samples"],
                reject=row["rejection_rate"],
                exhaust=row["max_token_exhaustion_rate"],
                aps=row["accepted_per_second"] or 0.0,
                acc_tok=row["avg_accepted_generated_tokens"] or 0.0,
                rej_tok=(
                    f"{row['avg_rejected_generated_tokens']:.1f}"
                    if row["avg_rejected_generated_tokens"] is not None
                    else "-"
                ),
            )
        )
    summary_md.write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    cuda_available = torch.cuda.is_available()
    if args.require_cuda and not cuda_available:
        raise RuntimeError("CUDA is required, but torch.cuda.is_available() is false.")

    device = "cuda" if cuda_available else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    model_started = perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=dtype,
        local_files_only=args.local_files_only,
    ).to(device)
    model.eval()
    model_load_seconds = perf_counter() - model_started
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        local_files_only=args.local_files_only,
    )
    parameter_device = get_model_parameter_device(model)
    if args.require_cuda and parameter_device != "cuda:0":
        raise RuntimeError(f"expected model on cuda:0, got {parameter_device!r}")

    regex = GRAMMAR_REGEX["json"]
    prompt = GRAMMAR_PROMPT["json"]
    pattern = re.compile("^(?:" + regex + ")$")
    slug = model_slug(args.model)
    rows: list[dict[str, Any]] = []

    print(
        f"[pilot] model={args.model} device={device} "
        f"parameter_device={parameter_device} "
        f"model_load_seconds={model_load_seconds:.2f}"
    )
    print(
        f"[pilot] budgets={args.budgets} attempts_per_cell={args.n} "
        f"max_attempts={args.max_attempts}"
    )

    for method in METHODS:
        for budget in args.budgets:
            print(f"[cell] method={method.name} max_tokens={budget}")
            cell = generate_cell(
                model=model,
                tokenizer=tokenizer,
                regex=regex,
                prompt=prompt,
                pattern=pattern,
                method=method,
                budget=budget,
                attempt_count=args.n,
                max_attempts=args.max_attempts,
                seed=args.seed,
                progress=not args.no_progress,
            )
            cell["model"] = args.model
            cell["model_loading"] = {
                "device": device,
                "model_parameter_device": parameter_device,
                "model_load_seconds": model_load_seconds,
                "torch_dtype": str(dtype).removeprefix("torch."),
                "torch_version": torch.__version__,
                "torch_cuda_version": torch.version.cuda,
            }
            data_path = (
                args.data_root
                / method.name
                / slug
                / f"json_max_tokens_{budget}.json"
            )
            data_path.parent.mkdir(parents=True, exist_ok=True)
            data_path.write_text(json.dumps(cell, indent=2, sort_keys=True) + "\n")
            row = summarize_cell(cell, tokenizer)
            row["path"] = str(data_path)
            rows.append(row)
            print(
                "[result] method={method} max_tokens={budget} accepted={accepted} "
                "rejected={rejected} rejection_rate={rejection:.3f} "
                "accepted_per_second={aps:.3f}".format(
                    method=method.name,
                    budget=budget,
                    accepted=row["accepted_samples"],
                    rejected=row["rejected_samples"],
                    rejection=row["rejection_rate"],
                    aps=row["accepted_per_second"] or 0.0,
                )
            )

    write_summary(args.output_root, rows)
    print(f"[summary] wrote {args.output_root / 'json_budget_pilot_metrics.json'}")
    print(f"[summary] wrote {args.output_root / 'json_budget_pilot_metrics.csv'}")
    print(f"[summary] wrote {args.output_root / 'json_budget_pilot_summary.md'}")


if __name__ == "__main__":
    main()
