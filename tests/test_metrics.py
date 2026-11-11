"""Tests for diversity evaluation metrics."""

import numpy as np
import pytest

from diverse_guide.evaluation.metrics import (
    distinct_ngram,
    get_coverage_ratio,
    path_coverage,
    state_coverage,
    transition_coverage,
    vendi_score,
)


def test_coverage_ratio_counts_entries_meeting_threshold():
    assert get_coverage_ratio({"x": 2, "y": 3}, threshold=1) == 1.0
    assert get_coverage_ratio({"x": 0, "y": 0}, threshold=1) == 0.0
    assert get_coverage_ratio({"x": 1, "y": 0}, threshold=1) == 0.5
    assert get_coverage_ratio({"x": 1, "y": 3}, threshold=2) == 0.5
    assert get_coverage_ratio({}, threshold=1) == 0.0


def test_dfa_coverage_metrics(dfa_single):
    # (?:[ab])$ has two states (initial, final), two transitions (a, b), and a
    # single initial-to-final state path.
    assert state_coverage(dfa_single, ["a"]) == 1.0
    assert state_coverage(dfa_single, ["a", "b"]) == 1.0
    assert transition_coverage(dfa_single, ["a"]) == pytest.approx(0.5)
    assert transition_coverage(dfa_single, ["a", "b"]) == pytest.approx(1.0)
    assert path_coverage(dfa_single, ["a"]) == pytest.approx(1.0)


def test_dfa_coverage_metrics_with_step_size(dfa_single):
    for metric in (state_coverage, transition_coverage, path_coverage):
        stepwise = metric(dfa_single, ["a", "b"], step_size=1)
        assert isinstance(stepwise, list) and len(stepwise) == 2
        assert stepwise[0] <= stepwise[1]
        # A trailing partial step is reported as an extra point.
        with_remainder = metric(dfa_single, ["a", "b", "a"], step_size=2)
        assert len(with_remainder) == 2
        assert all(0.0 <= v <= 1.0 for v in with_remainder)


def _identity_kernel(inputs):
    """Each sequence is orthogonal to all others."""
    return np.eye(len(inputs))


def _ones_kernel(inputs):
    """All sequences are identical (minimum diversity)."""
    n = len(inputs)
    return np.ones((n, n))


def test_vendi_score_extremes_and_step_size():
    assert vendi_score(["a", "b"], _identity_kernel) == pytest.approx(2.0, rel=1e-3)
    assert vendi_score(["a", "b"], _ones_kernel) == pytest.approx(1.0, rel=1e-3)
    assert vendi_score(["a"], _identity_kernel) == pytest.approx(1.0, rel=1e-3)

    stepwise = vendi_score(["a", "b", "c", "d"], _identity_kernel, step_size=2)
    assert stepwise == pytest.approx([2.0, 4.0], rel=1e-3)
    with_remainder = vendi_score(["a", "b", "c"], _identity_kernel, step_size=2)
    assert with_remainder == pytest.approx([2.0, 3.0], rel=1e-3)


def test_distinct_ngram():
    assert distinct_ngram(["aa", "bb"], n=1) == (2, 2)
    assert distinct_ngram(["ab", "ba"], n=2) == (2, 2)
    assert distinct_ngram(["aa", "aa"], n=2)[0] == 1
    assert distinct_ngram([], n=2) == (0, 0)
    # n longer than the string yields no n-grams.
    assert distinct_ngram(["abc"], n=5)[0] == 0
