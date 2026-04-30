"""Microbenchmark regex logits processors without loading a language model."""

import argparse
import time

import torch
from transformers import AutoTokenizer

try:
    from generate_re import GRAMMAR_PROMPT, GRAMMAR_REGEX
except ModuleNotFoundError:
    from scripts.generate_re import GRAMMAR_PROMPT, GRAMMAR_REGEX

from diverse_guide.guide_rust import (
    DiverseRegexLogitsProcessor,
    RegexMaskLogitsProcessor,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Microbenchmark internal regex logits processors."
    )
    parser.add_argument("grammar", choices=sorted(GRAMMAR_REGEX))
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--iterations", type=int, default=200)
    return parser.parse_args()


def time_processor(
    label: str,
    processor: RegexMaskLogitsProcessor | DiverseRegexLogitsProcessor,
    input_ids: torch.LongTensor,
    scores_template: torch.FloatTensor,
    iterations: int,
) -> None:
    start = time.perf_counter()
    for _ in range(iterations):
        processor.reset()
        processor(input_ids, scores_template.clone())
    elapsed = time.perf_counter() - start
    print(f"{label}: {elapsed / iterations * 1000:.3f} ms/call")


def main() -> None:
    args = parse_args()
    if args.iterations <= 0:
        raise ValueError("--iterations must be positive")

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    prompt = GRAMMAR_PROMPT[args.grammar]
    regex = GRAMMAR_REGEX[args.grammar]
    input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"]
    scores_template = torch.randn((1, len(tokenizer.get_vocab())), dtype=torch.float32)

    mask_only = RegexMaskLogitsProcessor(regex, tokenizer)
    diverse = DiverseRegexLogitsProcessor(regex, tokenizer, gamma=0.5, beta=3.0)

    time_processor(
        "mask-only processor", mask_only, input_ids, scores_template, args.iterations
    )
    time_processor(
        "diverse processor", diverse, input_ids, scores_template, args.iterations
    )


if __name__ == "__main__":
    main()
