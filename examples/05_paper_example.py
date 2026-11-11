"""
The illustrative example from Section 3 of the DiverseGuide OSP manuscript.

Creates a DiverseGuide generator and a mask-only baseline generator for the
same model, tokenizer, and regular expression, and draws ten samples from each.
Runs on a CPU (slowly) when no CUDA device is available.

Run:
    uv run python examples/05_paper_example.py
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from diverse_guide import DiverseGuide, baseline_regex

model_id = "Qwen/Qwen2.5-1.5B-Instruct"
device = "cuda" if torch.cuda.is_available() else "cpu"

model = AutoModelForCausalLM.from_pretrained(model_id).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_id)
regex = r"#[0-9a-fA-F]{6}"
prompt = "Give me a CSS color code."

generator = DiverseGuide(model, tokenizer, regex)
samples = generator.generate_batch(prompt, n=10, max_tokens=20)

baseline = baseline_regex(model, tokenizer, regex)
base_samples = baseline.generate_batch(prompt, n=10, max_tokens=20)

print("DiverseGuide:", samples)
print("Baseline:    ", base_samples)
