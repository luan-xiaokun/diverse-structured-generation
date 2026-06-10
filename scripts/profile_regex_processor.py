"""Profile regex/DFA logits processor overhead without loading model weights."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, pstdev
from time import perf_counter
from typing import Any

import torch
from transformers import AutoTokenizer

try:
    from generate_re import GRAMMAR_REGEX
except ModuleNotFoundError:
    from scripts.generate_re import GRAMMAR_REGEX

from diverse_guide.guide_rust import (
    DiverseRegexGuide,
    DiverseRegexLogitsProcessor,
    RegexMaskLogitsProcessor,
)


DEFAULT_PREFIXES = {
    "css-color": ("", "#", "#a", "rgb(", "red"),
    "email": ("", "a", "a@", "a@b", "a@b."),
    "json": ("", '{"name"', '{"name":', '{"name":"A"', '{"name":"A","gender":'),
    "no-bomb": ("", "a", "bo", "bom", "safe"),
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Profile regex/DFA guide and logits processor overhead."
    )
    parser.add_argument("grammar", choices=sorted(GRAMMAR_REGEX))
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument(
        "--prefixes",
        nargs="+",
        help="Generated-output prefixes to advance before profiling a state.",
    )
    parser.add_argument("--repeat", type=int, default=50)
    parser.add_argument(
        "--runs",
        type=int,
        default=1,
        help="Number of full profiler runs to execute before aggregating results.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args(argv)
    if args.repeat <= 0:
        parser.error("--repeat must be positive")
    if args.runs <= 0:
        parser.error("--runs must be positive")
    return args


def _encode_text(tokenizer: Any, text: str) -> list[int]:
    encode = getattr(tokenizer, "encode", None)
    if callable(encode):
        try:
            return [int(token_id) for token_id in encode(text, add_special_tokens=False)]
        except TypeError:
            return [int(token_id) for token_id in encode(text)]

    vocab = getattr(tokenizer, "get_vocab", None)
    if callable(vocab):
        token_to_id = vocab()
        remaining = text
        token_ids = []
        while remaining:
            choices = [
                (token, int(token_id))
                for token, token_id in token_to_id.items()
                if token and remaining.startswith(token)
            ]
            if not choices:
                raise ValueError(f"cannot encode prefix with tokenizer: {text!r}")
            token, token_id = max(choices, key=lambda item: (len(item[0]), -item[1]))
            token_ids.append(token_id)
            remaining = remaining[len(token) :]
        return token_ids

    raise TypeError("tokenizer must provide encode() or get_vocab()")


def _score_width(tokenizer: Any) -> int:
    vocab = getattr(tokenizer, "get_vocab", None)
    if callable(vocab):
        ids = [int(token_id) for token_id in vocab().values()]
        eos = getattr(tokenizer, "eos_token_id", None)
        if eos is not None:
            ids.append(int(eos))
        return max(ids) + 1
    vocab_size = getattr(tokenizer, "vocab_size", None)
    if vocab_size is not None:
        return int(vocab_size)
    raise TypeError("tokenizer must provide get_vocab() or vocab_size")


def _time_call(repeat: int, fn) -> tuple[Any, float]:  # noqa: ANN001
    result = None
    started = perf_counter()
    for _ in range(repeat):
        result = fn()
    return result, (perf_counter() - started) * 1000.0 / repeat


def _advance_prefix(guide: DiverseRegexGuide, token_ids: list[int]) -> int:
    state = guide.initial_state
    for token_id in token_ids:
        state = guide.get_next_state(state, int(token_id))
    return state


def _profile_prefix(
    *,
    prefix: str,
    token_ids: list[int],
    guide: DiverseRegexGuide,
    mask_proc: RegexMaskLogitsProcessor | None,
    diverse_proc: DiverseRegexLogitsProcessor | None,
    scores_template: torch.FloatTensor,
    repeat: int,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "prefix": prefix,
        "token_ids": token_ids,
        "token_count": len(token_ids),
    }
    try:
        state, advance_prefix_ms = _time_call(
            repeat, lambda: _advance_prefix(guide, token_ids)
        )
        allowed_tokens, allowed_ms = _time_call(
            repeat, lambda: guide.dfa.get_allowed_token_ids(state)
        )
        counts, compute_counts_ms = _time_call(
            repeat, lambda: guide.dfa.compute_counts(state)
        )
        allowed_tokens = list(allowed_tokens)
        selected_token = next(
            (
                int(token_id)
                for token_id in allowed_tokens
                if int(token_id) != int(guide.eos_token_id)
            ),
            int(allowed_tokens[0]) if allowed_tokens else None,
        )

        update_local_ms = None
        next_state_ms = None
        if selected_token is not None:
            guide.dfa.reset_local_state_counter()
            _, update_local_ms = _time_call(
                repeat,
                lambda: guide.dfa.update_local_state_counter(state, selected_token),
            )
            _, next_state_ms = _time_call(
                repeat, lambda: guide.dfa.get_next_token_state(state, selected_token)
            )
            guide.dfa.reset_local_state_counter()

        mask_ms = None
        diverse_ms = None
        processor_delta_ms = None
        if mask_proc is not None and diverse_proc is not None:
            input_ids = torch.tensor([token_ids], dtype=torch.long)
            mask_ms = _time_processor(mask_proc, input_ids, scores_template, repeat)
            diverse_ms = _time_processor(
                diverse_proc, input_ids, scores_template, repeat
            )
            processor_delta_ms = diverse_ms - mask_ms

        row.update(
            {
                "state": int(state),
                "allowed_count": len(allowed_tokens),
                "selected_token_id": selected_token,
                "advance_prefix_ms": advance_prefix_ms,
                "get_allowed_token_ids_ms": allowed_ms,
                "compute_counts_ms": compute_counts_ms,
                "update_local_state_counter_ms": update_local_ms,
                "get_next_token_state_ms": next_state_ms,
                "mask_processor_ms": mask_ms,
                "diverse_processor_ms": diverse_ms,
                "processor_diverse_minus_mask_ms": processor_delta_ms,
                "reward_count_len": len(counts[1]),
                "penalty_count_len": len(counts[2]),
                "error": None,
            }
        )
    except Exception as exc:  # pragma: no cover - exercised in real diagnostics
        row["error"] = str(exc)
    return row


def _time_processor(
    processor: RegexMaskLogitsProcessor | DiverseRegexLogitsProcessor,
    input_ids: torch.LongTensor,
    scores_template: torch.FloatTensor,
    repeat: int,
) -> float:
    started = perf_counter()
    for _ in range(repeat):
        processor.reset()
        processor._seq_start_idx = 0
        if hasattr(processor.dfa, "reset_local_state_counter"):
            processor.dfa.reset_local_state_counter()
        processor(input_ids, scores_template.clone())
    return (perf_counter() - started) * 1000.0 / repeat


def _profile_once(
    args: argparse.Namespace,
    tokenizer: Any | None = None,
) -> dict[str, Any]:
    regex = GRAMMAR_REGEX[args.grammar]

    tokenizer_started = perf_counter()
    tokenizer = tokenizer or AutoTokenizer.from_pretrained(
        args.model,
        local_files_only=args.local_files_only,
    )
    tokenizer_load_seconds = perf_counter() - tokenizer_started

    guide_started = perf_counter()
    guide = DiverseRegexGuide(regex, tokenizer)
    guide_build_seconds = perf_counter() - guide_started

    processor_started = perf_counter()
    mask_proc = RegexMaskLogitsProcessor(regex, tokenizer)
    diverse_proc = DiverseRegexLogitsProcessor(regex, tokenizer)
    processor_build_seconds = perf_counter() - processor_started

    prefixes = args.prefixes or list(DEFAULT_PREFIXES.get(args.grammar, ("",)))
    encoded_prefixes = [(prefix, _encode_text(tokenizer, prefix)) for prefix in prefixes]
    scores_template = torch.randn((1, _score_width(tokenizer)), dtype=torch.float32)

    rows = [
        _profile_prefix(
            prefix=prefix,
            token_ids=token_ids,
            guide=guide,
            mask_proc=mask_proc,
            diverse_proc=diverse_proc,
            scores_template=scores_template,
            repeat=args.repeat,
        )
        for prefix, token_ids in encoded_prefixes
    ]

    return {
        "schema_version": 1,
        "experiment": "regex_processor_profile",
        "grammar": args.grammar,
        "model": args.model,
        "repeat": args.repeat,
        "vocab_size": len(tokenizer.get_vocab()),
        "score_width": _score_width(tokenizer),
        "timing": {
            "tokenizer_load_seconds": tokenizer_load_seconds,
            "guide_build_seconds": guide_build_seconds,
            "processor_build_seconds": processor_build_seconds,
        },
        "prefixes": rows,
    }


def _numeric_summary(values: list[int | float | None]) -> dict[str, float | int] | None:
    numeric_values = [float(value) for value in values if value is not None]
    if not numeric_values:
        return None
    return {
        "count": len(numeric_values),
        "mean": mean(numeric_values),
        "min": min(numeric_values),
        "max": max(numeric_values),
        "stdev": pstdev(numeric_values),
    }


def _aggregate_profiles(runs: list[dict[str, Any]]) -> dict[str, Any]:
    timing_fields = tuple(runs[0]["timing"].keys())
    prefix_fields = (
        "token_count",
        "allowed_count",
        "advance_prefix_ms",
        "get_allowed_token_ids_ms",
        "compute_counts_ms",
        "update_local_state_counter_ms",
        "get_next_token_state_ms",
        "mask_processor_ms",
        "diverse_processor_ms",
        "processor_diverse_minus_mask_ms",
        "reward_count_len",
        "penalty_count_len",
    )

    aggregated_prefixes = []
    for prefix_index, baseline_row in enumerate(runs[0]["prefixes"]):
        row = {
            "prefix": baseline_row["prefix"],
            "token_ids": baseline_row["token_ids"],
            "selected_token_id": baseline_row.get("selected_token_id"),
            "state": baseline_row.get("state"),
            "error_count": sum(
                1 for run in runs if run["prefixes"][prefix_index].get("error") is not None
            ),
            "metrics": {},
        }
        for field in prefix_fields:
            summary = _numeric_summary(
                [run["prefixes"][prefix_index].get(field) for run in runs]
            )
            if summary is not None:
                row["metrics"][field] = summary
        aggregated_prefixes.append(row)

    timing = {}
    for field in timing_fields:
        summary = _numeric_summary([run["timing"].get(field) for run in runs])
        if summary is not None:
            timing[field] = summary

    return {"timing": timing, "prefixes": aggregated_prefixes}


def run_profile(
    args: argparse.Namespace,
    tokenizer: Any | None = None,
) -> dict[str, Any]:
    first_run = _profile_once(args, tokenizer=tokenizer)
    if args.runs == 1:
        return first_run

    runs = [dict(first_run, run_index=0)]
    for run_index in range(1, args.runs):
        runs.append(dict(_profile_once(args, tokenizer=tokenizer), run_index=run_index))

    result = dict(first_run)
    result["runs"] = runs
    result["aggregates"] = _aggregate_profiles(runs)
    return result


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = parse_args(argv)
    result = run_profile(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True))
    print(f"Wrote regex processor profile to {args.output}")
    return result


if __name__ == "__main__":
    main()
