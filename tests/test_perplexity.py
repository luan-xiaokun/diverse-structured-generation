"""Tests for diverse_guide.evaluation.perplexity.

All tests use lightweight mock model and tokenizer objects so no LLM download
is required.  torch is needed (already a project dependency).
"""

import math

import pytest
import torch

from diverse_guide.evaluation.perplexity import calculate_perplexity

# ---------------------------------------------------------------------------
# Mock fixtures
# ---------------------------------------------------------------------------


class _MockTokenizer:
    """Returns a fixed input_ids tensor regardless of the input text."""

    def __call__(self, text, return_tensors=None, truncation=None, max_length=None):
        class _Encoding:
            input_ids = torch.tensor([[1, 2, 3]])

        return _Encoding()


class _MockModel:
    """Returns a fixed loss value."""

    def __init__(self, loss: float):
        self._loss = loss

    def __call__(self, input_ids, labels=None):
        loss_val = self._loss

        class _Output:
            loss = torch.tensor(loss_val)

        return _Output()


@pytest.fixture
def tokenizer():
    return _MockTokenizer()


# ---------------------------------------------------------------------------
# Basic correctness
# ---------------------------------------------------------------------------


def test_perplexity_equals_exp_loss(tokenizer):
    """perplexity should be exp(loss) for a valid loss value."""
    loss = 2.0
    model = _MockModel(loss)
    ppl = calculate_perplexity("test text", model, tokenizer, device="cpu")
    assert ppl == pytest.approx(math.exp(loss), rel=1e-5)


def test_perplexity_low_loss(tokenizer):
    """Very low loss → perplexity close to 1."""
    model = _MockModel(0.0)
    ppl = calculate_perplexity("test text", model, tokenizer, device="cpu")
    assert ppl == pytest.approx(1.0, rel=1e-5)


def test_perplexity_high_loss(tokenizer):
    """High loss → large perplexity."""
    model = _MockModel(10.0)
    ppl = calculate_perplexity("test text", model, tokenizer, device="cpu")
    assert ppl == pytest.approx(math.exp(10.0), rel=1e-4)


# ---------------------------------------------------------------------------
# NaN / Inf handling
# ---------------------------------------------------------------------------


def test_perplexity_nan_loss_returns_inf(tokenizer):
    """NaN loss should be reported as math.inf, not raise."""
    model = _MockModel(float("nan"))
    ppl = calculate_perplexity("text", model, tokenizer, device="cpu")
    assert math.isinf(ppl)


def test_perplexity_inf_loss_returns_inf(tokenizer):
    """Inf loss should be reported as math.inf."""
    model = _MockModel(float("inf"))
    ppl = calculate_perplexity("text", model, tokenizer, device="cpu")
    assert math.isinf(ppl)


# ---------------------------------------------------------------------------
# Return type
# ---------------------------------------------------------------------------


def test_perplexity_returns_float(tokenizer):
    model = _MockModel(1.5)
    ppl = calculate_perplexity("text", model, tokenizer, device="cpu")
    assert isinstance(ppl, float)
