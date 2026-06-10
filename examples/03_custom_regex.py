"""
Custom regex grammar: bring your own regex and prompt.

This example demonstrates how to define a new generation grammar with any
regular expression, and how the gamma / beta hyperparameters affect diversity.

Run:
    uv run python examples/03_custom_regex.py
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from diverse_guide import DiverseGuide

MODEL_PATH = "Qwen/Qwen2.5-1.5B-Instruct"


# Example: simple JSON object with name + age fields
JSON_REGEX = r'(?:\{\s*"name":\s*"(?:.+?)",\s*"age":\s*\d+\s*\})'
JSON_PROMPT = 'Give me a JSON object with fields "name" (string) and "age" (integer).'

# Example: US phone number
PHONE_REGEX = r"\(\d{3}\) \d{3}-\d{4}$"
PHONE_PROMPT = "Give me a US phone number."


def run_task(
    model,
    tokenizer,
    regex: str,
    prompt: str,
    n: int = 8,
    gamma: float = 0.5,
    beta: float = 3.0,
    max_tokens: int = 60,
):
    print(f"Regex : {regex[:80]}{'...' if len(regex) > 80 else ''}")
    print(f"Prompt: {prompt}")
    print(f"gamma={gamma}, beta={beta}\n")

    generator = DiverseGuide(model, tokenizer, regex, gamma=gamma, beta=beta)
    samples = generator.generate_batch(prompt, n=n, max_tokens=max_tokens)

    for i, s in enumerate(samples):
        print(f"  [{i + 1}] {s}")
    print(f"\nUnique: {len(set(samples))} / {n}\n")
    return samples


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, dtype=dtype).to(device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

    print("=" * 60)
    print("Grammar 1: JSON objects")
    print("=" * 60)
    run_task(model, tokenizer, JSON_REGEX, JSON_PROMPT, n=6, max_tokens=60)

    print("=" * 60)
    print("Grammar 2: Phone numbers — low gamma (less diverse)")
    print("=" * 60)
    run_task(model, tokenizer, PHONE_REGEX, PHONE_PROMPT, n=8, gamma=0.1, max_tokens=20)

    print("=" * 60)
    print("Grammar 2: Phone numbers — high gamma (more diverse)")
    print("=" * 60)
    run_task(model, tokenizer, PHONE_REGEX, PHONE_PROMPT, n=8, gamma=1.0, max_tokens=20)


if __name__ == "__main__":
    main()
