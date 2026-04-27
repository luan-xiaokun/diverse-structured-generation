"""
Utility for computing perplexity of text using a language model.
"""

import math
import warnings

import torch

warnings.filterwarnings("ignore", message="Using `pad_token_id` as `eos_token_id`")


def calculate_perplexity(
    text: str, model, tokenizer, device: str = "cuda", max_length: int = 128
) -> float:
    """Compute the perplexity of a given text using a language model."""
    encodings = tokenizer(
        text, return_tensors="pt", truncation=True, max_length=max_length
    )
    input_ids = encodings.input_ids.to(device)
    target_ids = input_ids.clone()

    with torch.no_grad():
        outputs = model(input_ids, labels=target_ids)
        neg_log_likelihood = outputs.loss
        if torch.isnan(neg_log_likelihood) or torch.isinf(neg_log_likelihood):
            print(f"[Warning] NNL is NaN or Inf for text: {text}")
            return math.inf
    ppl = torch.exp(neg_log_likelihood)
    return ppl.item()
