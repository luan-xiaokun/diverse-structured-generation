"""
Batch generation: produce N samples in a single model.generate() call.

``generate_batch`` is more efficient than calling ``__call__`` in a loop
because all sequences are generated in parallel on the GPU.  Within the batch,
the parallel sequences share the local state counter, which penalises DFA
states that the other sequences already use heavily.  After the batch, every
completed sample that matches the regex is added to the global path counter,
so later calls are diversified relative to it.

Key difference from 01_quickstart.py:
  - generate_batch  →  parallel GPU execution, global counter updated after
                       the batch
  - loop + __call__ →  sequential execution (one sequence at a time), global
                       counter updated by update_generated_content after each

Run:
    uv run python examples/02_batch_generation.py
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from diverse_guide import DiverseGuide, baseline_regex

MODEL_PATH = "Qwen/Qwen2.5-1.5B-Instruct"

CSS_COLOR_REGEX = (
    r"#(?:[0-9a-fA-F]{6}|[0-9a-fA-F]{3})"
    r"|rgba?\(\s*(?:25[0-5]|2[0-4][0-9]|1?[0-9]{1,2})\s*,"
    r"\s*(?:25[0-5]|2[0-4][0-9]|1?[0-9]{1,2})\s*,"
    r"\s*(?:25[0-5]|2[0-4][0-9]|1?[0-9]{1,2})\s*"
    r"(?:,\s*(?:0(?:\.\d+)?|1(?:\.0+)?)\s*)?\)"
)

PROMPT = "Give me a CSS color code."
N = 20
MAX_TOKENS = 20


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, dtype=dtype).to(device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

    diverse_gen = DiverseGuide(model, tokenizer, CSS_COLOR_REGEX)
    baseline_gen = baseline_regex(model, tokenizer, CSS_COLOR_REGEX)

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
