"""
Quickstart: diverse regex-constrained generation.

This example shows the minimal steps to generate diverse outputs that match a
regular expression using a locally stored HuggingFace model.

Requires:
    - A model downloaded by HuggingFace (see README for download instructions)
    - Project dependencies installed with: uv sync

Run:
    uv run python examples/01_quickstart.py
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from diverse_guide import DiverseGuide

MODEL_PATH = "Qwen/Qwen2.5-1.5B-Instruct"

# A simple regex for CSS hex color codes (#rrggbb or #rgb)
CSS_HEX_REGEX = r"#(?:[0-9a-fA-F]{6}|[0-9a-fA-F]{3})$"

PROMPT = "Give me a CSS color code."

N_SAMPLES = 10
MAX_TOKENS = 10


def main():
    print(f"Loading model from {MODEL_PATH} ...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, dtype=dtype).to(device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

    # Build the diverse generator once (compiles DFA + token-transition table)
    generator = DiverseGuide(model, tokenizer, CSS_HEX_REGEX)

    print(f"\nGenerating {N_SAMPLES} diverse CSS hex colors:\n")
    samples = []
    for i in range(N_SAMPLES):
        text = generator(PROMPT, max_tokens=MAX_TOKENS)
        samples.append(text)
        print(f"  [{i + 1:2d}] {text}")
        # Update the global path counter so the next sample is diversified
        # relative to all previously generated outputs.
        generator.update_generated_content(text)

    print(f"\nUnique samples: {len(set(samples))} / {N_SAMPLES}")


if __name__ == "__main__":
    main()
