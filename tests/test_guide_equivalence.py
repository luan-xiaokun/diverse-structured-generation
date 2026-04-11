"""Equivalence tests between the pure-Python and Rust-backed guide implementations.

Tests verify that guide_python and guide_rust produce identical DFA structural
results (allowed tokens, state transitions, state sequences) and equivalent
logit adjustments when counters are zero.  No LLM download is required.
"""

import torch

from diverse_guide.guide_python import DiverseRegexGuide as PyGuide
from diverse_guide.guide_python import DiverseRegexLogitsProcessor as PyProcessor
from diverse_guide.guide_rust import DiverseRegexGuide as RustGuide
from diverse_guide.guide_rust import DiverseRegexLogitsProcessor as RustProcessor

REGEX = "[ab]+"  # simple regex: one-or-more a/b


# ---------------------------------------------------------------------------
# DiverseRegexGuide equivalence
# ---------------------------------------------------------------------------


def test_guides_same_initial_state(mock_tokenizer):
    py = PyGuide(REGEX, mock_tokenizer)
    rs = RustGuide(REGEX, mock_tokenizer)
    assert py.initial_state == rs.initial_state


def test_guides_same_allowed_tokens(mock_tokenizer):
    py = PyGuide(REGEX, mock_tokenizer)
    rs = RustGuide(REGEX, mock_tokenizer)
    state = py.initial_state
    assert sorted(py.get_next_instruction(state)) == sorted(
        rs.get_next_instruction(state)
    )


def test_guides_same_next_state(mock_tokenizer):
    """Advancing by token 'a' (id=10) should reach the same state in both guides."""
    py = PyGuide(REGEX, mock_tokenizer)
    rs = RustGuide(REGEX, mock_tokenizer)
    s0 = py.initial_state
    py_next = py.get_next_state(s0, 10)  # token 'a'
    rs_next = rs.get_next_state(s0, 10)
    assert py_next == rs_next


def test_guides_same_is_final_state(mock_tokenizer):
    py = PyGuide(REGEX, mock_tokenizer)
    rs = RustGuide(REGEX, mock_tokenizer)
    s0 = py.initial_state
    # After consuming 'a', next state should be final in both
    next_state = py.get_next_state(s0, 10)
    assert py.is_final_state(next_state) == rs.is_final_state(next_state)


def test_guides_same_state_sequence_from_token(mock_tokenizer):
    """Byte-level state sequence for token 'a' should be identical."""
    py = PyGuide(REGEX, mock_tokenizer)
    s0 = py.initial_state
    py_seq = py.get_state_sequence_from_token_id(s0, 10)
    # Rust DFA exposes same via get_byte_state_sequence
    rust_seq = list(py.dfa.get_byte_state_sequence(s0, 10))
    assert py_seq == rust_seq


# ---------------------------------------------------------------------------
# Logit masking equivalence (zero counters)
# ---------------------------------------------------------------------------


def _make_scores(n_vocab: int = 100) -> torch.FloatTensor:
    torch.manual_seed(42)
    return torch.randn(1, n_vocab)


def test_masking_same_allowed_set(mock_tokenizer):
    """Both processors should allow exactly the same token ids."""
    py_proc = PyProcessor(REGEX, mock_tokenizer, gamma=0.5, beta=3.0)
    rs_proc = RustProcessor(REGEX, mock_tokenizer, gamma=0.5, beta=3.0)

    input_ids = torch.zeros(1, 0, dtype=torch.long)
    scores_py = _make_scores()
    scores_rs = scores_py.clone()

    scores_py = py_proc(input_ids, scores_py)
    scores_rs = rs_proc(input_ids, scores_rs)

    # Positions that are -inf should be identical
    py_inf = (scores_py == float("-inf")).squeeze()
    rs_inf = (scores_rs == float("-inf")).squeeze()
    assert torch.equal(py_inf, rs_inf), (
        "Masking differs between Python and Rust processors"
    )


def test_zero_counter_adjustment_zero(mock_tokenizer):
    """With no prior sequences (all counters zero), the diversity adjustment is zero,
    so both processors should produce identical final scores (same masking, no shift).
    """
    py_proc = PyProcessor(REGEX, mock_tokenizer, gamma=0.5, beta=3.0)
    rs_proc = RustProcessor(REGEX, mock_tokenizer, gamma=0.5, beta=3.0)

    input_ids = torch.zeros(1, 0, dtype=torch.long)
    scores_base = _make_scores()

    scores_py = py_proc(input_ids, scores_base.clone())
    scores_rs = rs_proc(input_ids, scores_base.clone())

    # On allowed positions both should be identical (adjustment=0 when counts=0)
    allowed = scores_py != float("-inf")
    assert torch.allclose(scores_py[allowed], scores_rs[allowed], atol=1e-5), (
        "Scores differ on allowed tokens even with zero counters"
    )


# ---------------------------------------------------------------------------
# DFA state sequence used by update_generated_content
# ---------------------------------------------------------------------------


def test_state_sequence_from_string(mock_tokenizer):
    """should include initial state as first element."""
    py = PyGuide(REGEX, mock_tokenizer)
    dfa = py.dfa
    states = list(dfa.get_state_sequence("ab"))
    # First element is the initial state
    assert states[0] == py.initial_state
    # Length equals len("ab") + 1  (one state per byte plus start)
    assert len(states) == len("ab") + 1
