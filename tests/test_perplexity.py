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


def test_perplexity_is_exp_of_loss(tokenizer):
    for loss in (0.0, 1.5, 2.0, 10.0):
        ppl = calculate_perplexity("test text", _MockModel(loss), tokenizer, "cpu")
        assert isinstance(ppl, float)
        assert ppl == pytest.approx(math.exp(loss), rel=1e-4)


def test_perplexity_reports_non_finite_loss_as_inf(tokenizer):
    for loss in (float("nan"), float("inf")):
        ppl = calculate_perplexity("text", _MockModel(loss), tokenizer, device="cpu")
        assert math.isinf(ppl)
