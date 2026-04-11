"""Tests for diversity evaluation metrics."""

import numpy as np
import pytest

from metrics import (
    distinct_ngram,
    get_coverage_ratio,
    path_coverage,
    state_coverage,
    transition_coverage,
    vendi_score,
)

# ---------------------------------------------------------------------------
# get_coverage_ratio
# ---------------------------------------------------------------------------


def test_coverage_ratio_all_covered():
    counts = {"x": 2, "y": 3}
    assert get_coverage_ratio(counts, threshold=1) == 1.0


def test_coverage_ratio_none_covered():
    counts = {"x": 0, "y": 0}
    assert get_coverage_ratio(counts, threshold=1) == 0.0


def test_coverage_ratio_partial():
    counts = {"x": 1, "y": 0}
    assert get_coverage_ratio(counts, threshold=1) == 0.5


def test_coverage_ratio_threshold():
    counts = {"x": 1, "y": 3}
    # threshold=2: only 'y' meets threshold
    assert get_coverage_ratio(counts, threshold=2) == 0.5


# ---------------------------------------------------------------------------
# state_coverage
# ---------------------------------------------------------------------------


def test_state_coverage_full(dfa_single):
    # Both 'a' and 'b' visit both states of (?:[ab])$ (initial and final)
    cov = state_coverage(dfa_single, ["a", "b"])
    assert cov == 1.0


def test_state_coverage_single_input(dfa_single):
    # A single input 'a' visits initial + final → 100% state coverage
    cov = state_coverage(dfa_single, ["a"])
    assert cov == 1.0


def test_state_coverage_step_size_returns_list(dfa_single):
    result = state_coverage(dfa_single, ["a", "b"], step_size=1)
    assert isinstance(result, list)
    assert len(result) == 2
    assert all(0.0 <= v <= 1.0 for v in result)


def test_state_coverage_step_size_monotone(dfa_single):
    result = state_coverage(dfa_single, ["a", "b"], step_size=1)
    # Coverage should be non-decreasing
    assert result[0] <= result[1]


# ---------------------------------------------------------------------------
# transition_coverage
# ---------------------------------------------------------------------------


def test_transition_coverage_partial(dfa_single):
    # (?:[ab])$ has two transitions: initial->(a)->final, initial->(b)->final
    # Input 'a' uses only the a-transition → 50%
    cov = transition_coverage(dfa_single, ["a"])
    assert cov == pytest.approx(0.5)


def test_transition_coverage_full(dfa_single):
    cov = transition_coverage(dfa_single, ["a", "b"])
    assert cov == pytest.approx(1.0)


def test_transition_coverage_step_size(dfa_single):
    result = transition_coverage(dfa_single, ["a", "b"], step_size=1)
    assert isinstance(result, list)
    assert len(result) == 2


# ---------------------------------------------------------------------------
# path_coverage
# ---------------------------------------------------------------------------


def test_path_coverage_full_with_one_input(dfa_single):
    # (?:[ab])$ : both 'a' and 'b' go through the same (initial, final) state pair
    # so a single input covers all paths
    cov = path_coverage(dfa_single, ["a"])
    assert cov == pytest.approx(1.0)


def test_path_coverage_step_size(dfa_single):
    result = path_coverage(dfa_single, ["a", "b"], step_size=1)
    assert isinstance(result, list)


# ---------------------------------------------------------------------------
# vendi_score
# ---------------------------------------------------------------------------


def _identity_kernel(inputs):
    """Each sequence is orthogonal to all others."""
    return np.eye(len(inputs))


def _ones_kernel(inputs):
    """All sequences are identical (minimum diversity)."""
    n = len(inputs)
    return np.ones((n, n))


def test_vendi_score_identity_kernel():
    # With identity kernel, n=2 → vendi ≈ 2
    score = vendi_score(["a", "b"], _identity_kernel)
    assert score == pytest.approx(2.0, rel=1e-3)


def test_vendi_score_ones_kernel():
    # All-ones kernel (rank-1) → vendi ≈ 1
    score = vendi_score(["a", "b"], _ones_kernel)
    assert score == pytest.approx(1.0, rel=1e-3)


def test_vendi_score_single():
    score = vendi_score(["a"], _identity_kernel)
    assert score == pytest.approx(1.0, rel=1e-3)


def test_vendi_score_step_size_returns_list():
    inputs = ["a", "b", "c", "d"]
    result = vendi_score(inputs, _identity_kernel, step_size=2)
    assert isinstance(result, list)
    # step_size=2, len=4: indices 2, 4 (but 4==len, handled by remainder check)
    assert len(result) >= 1


# ---------------------------------------------------------------------------
# distinct_ngram
# ---------------------------------------------------------------------------


def test_distinct_ngram_unigrams():
    count, n_inputs = distinct_ngram(["aa", "bb"], n=1)
    # 'a', 'a', 'b', 'b' → 2 distinct unigrams
    assert count == 2
    assert n_inputs == 2


def test_distinct_ngram_bigrams():
    count, n_inputs = distinct_ngram(["ab", "ba"], n=2)
    assert count == 2  # 'ab' and 'ba'
    assert n_inputs == 2


def test_distinct_ngram_duplicates():
    count, _ = distinct_ngram(["aa", "aa"], n=2)
    assert count == 1  # only 'aa'


def test_distinct_ngram_empty_inputs():
    count, n_inputs = distinct_ngram([], n=2)
    assert count == 0
    assert n_inputs == 0


def test_distinct_ngram_longer_than_string():
    # n=5 for a 3-char string → no ngrams
    count, _ = distinct_ngram(["abc"], n=5)
    assert count == 0
