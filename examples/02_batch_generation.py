"""
Batch generation: produce N samples in a single model.generate() call.

``generate_batch`` is more efficient than calling ``__call__`` in a loop
because all sequences are generated in parallel on the GPU.  Path counters
are updated automatically after each completed sequence within the batch,
so later sequences in the batch are diversified relative to earlier ones.

Key difference from 01_quickstart.py:
  - generate_batch  →  parallel GPU execution, counters updated per-sequence
  - loop + __call__ →  sequential execution (one sequence at a time)

Run:
    PYTHONPATH=src python examples/02_batch_generation.py
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from diverse_guide import baseline_regex, diverse_regex

MODEL_PATH = "Qwen/Qwen2.5-1.5B-Instruct"

IPV4_REGEX = (
    r"(?:(?:25[0-5]|2[0-4][0-9]|1[0-9][0-9]|[1-9][0-9]|[0-9])\.){3}"
    r"(?:25[0-5]|2[0-4][0-9]|1[0-9][0-9]|[1-9][0-9]|[0-9])$"
)

PROMPT = "Give me an IPv4 address."
N = 20
MAX_TOKENS = 20


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, dtype=dtype).to(device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

    diverse_gen = diverse_regex(model, tokenizer, IPV4_REGEX)
    baseline_gen = baseline_regex(model, tokenizer, IPV4_REGEX)

    print("=== Diverse batch ===")
    diverse_samples = diverse_gen.generate_batch(PROMPT, n=N, max_tokens=MAX_TOKENS)
    for s in diverse_samples:
        print(f"  {s}")
    print(f"Unique: {len(set(diverse_samples))} / {N}\n")

    print("=== Baseline batch ===")
    baseline_samples = baseline_gen.generate_batch(PROMPT, n=N, max_tokens=MAX_TOKENS)
    for s in baseline_samples:
        print(f"  {s}")
    print(f"Unique: {len(set(baseline_samples))} / {N}")


if __name__ == "__main__":
    main()
