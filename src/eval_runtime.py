"""
This script evaluates structured generation efficiency.
"""

import argparse
import time

import outlines
import outlines.samplers
from outlines.models.transformers import Transformers
from tqdm import tqdm

from diverse_guide_rust import diverse_regex
from generate_re import TASK_PROMPT, TASK_REGEX


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate average token generation speed."
    )
    parser.add_argument("task", type=str, help="The task to generate text for.")
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2.5-1.5B-Instruct",
        help="The model to use for generation.",
    )
    parser.add_argument(
        "-n", type=int, default=1000, help="The number of tokens to generate."
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
    task = args.task
    regex = TASK_REGEX.get(task)
    prompt = TASK_PROMPT.get(task)
    print(f"Task: {task}" + (" (baseline)" if args.baseline else ""))

    model: Transformers = outlines.models.transformers(
        f"models/{args.model}", device="cuda"
    )
    sampler = outlines.samplers.multinomial(
        top_k=args.top_k, top_p=args.top_p, temperature=args.temperature
    )
    if args.baseline:
        generator = outlines.generate.regex(model, regex, sampler)
    else:
        generator = diverse_regex(model, regex, sampler)

    tokenizer = model.tokenizer.tokenizer
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
