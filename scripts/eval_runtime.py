"""
This script evaluates structured generation efficiency.
"""

import argparse
import time

import torch
from generate_re import GRAMMAR_PROMPT, GRAMMAR_REGEX
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from diverse_guide import baseline_regex, diverse_regex


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate average token generation speed."
    )
    parser.add_argument("grammar", type=str, help="The grammar to generate text for.")
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2.5-1.5B-Instruct",
        help="The model to use for generation.",
    )
    parser.add_argument(
        "-n", type=int, default=2000, help="The number of tokens to generate."
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=60,
        help="The maximum number of tokens to generate.",
    )
    parser.add_argument(
        "--top-k", type=int, default=None, help="The top-k value for sampling."
    )
    parser.add_argument(
        "--top-p", type=float, default=None, help="The top-p value for sampling."
    )
    parser.add_argument(
        "--temperature", type=float, default=None, help="The temperature for sampling."
    )
    parser.add_argument("--baseline", "-b", action="store_true", help="Use baseline")
    return parser.parse_args()


def main():
    args = parse_args()
    grammar = args.grammar
    if grammar not in GRAMMAR_REGEX:
        raise ValueError(
            f"Unknown grammar {grammar!r}. Choose from: {list(GRAMMAR_REGEX)}"
        )
    regex = GRAMMAR_REGEX[grammar]
    prompt = GRAMMAR_PROMPT[grammar]
    print(f"Grammar: {grammar}" + (" (baseline)" if args.baseline else ""))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype).to(device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    generation_kwargs = {}
    if args.top_k is not None:
        generation_kwargs["top_k"] = args.top_k
    if args.top_p is not None:
        generation_kwargs["top_p"] = args.top_p
    if args.temperature is not None:
        generation_kwargs["temperature"] = args.temperature

    if args.baseline:
        generator = baseline_regex(model, tokenizer, regex, **generation_kwargs)
    else:
        generator = diverse_regex(model, tokenizer, regex, **generation_kwargs)

    total_token_num = 0
    total_time = 0.0

    with tqdm(total=args.n) as pbar:
        while total_token_num < args.n:
            start_time = time.time()
            generated_text = generator(prompt, max_tokens=args.max_tokens)
            elapsed_time = time.time() - start_time
            token_num = len(tokenizer.encode(generated_text))
            total_token_num += token_num
            total_time += elapsed_time
            pbar.update(token_num)

    print(f"Generated {total_token_num} tokens in {total_time:.2f} seconds.")
    print(f"Tokens per second: {total_token_num / total_time:.2f}.")


if __name__ == "__main__":
    main()
