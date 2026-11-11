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


def test_guides_agree_on_dfa_structure(mock_tokenizer):
    py = PyGuide(REGEX, mock_tokenizer)
    rs = RustGuide(REGEX, mock_tokenizer)
    s0 = py.initial_state
    assert s0 == rs.initial_state
    assert sorted(py.get_next_instruction(s0)) == sorted(rs.get_next_instruction(s0))

    after_a = py.get_next_state(s0, 10)
    assert after_a == rs.get_next_state(s0, 10)
    assert py.is_final_state(after_a) == rs.is_final_state(after_a)

    # Byte-level state sequence of token 'a', and of the string "ab" as used by
    # update_generated_content (initial state plus one state per byte).
    assert py.get_state_sequence_from_token_id(s0, 10) == list(
        py.dfa.get_byte_state_sequence(s0, 10)
    )
    states = list(py.dfa.get_state_sequence("ab"))
    assert states[0] == s0 and len(states) == len("ab") + 1


def test_processors_agree_with_zero_counters(mock_tokenizer):
    """With all counters zero the diversity adjustment is zero, so both processors
    must mask the same tokens and leave the allowed scores identical."""
    py_proc = PyProcessor(REGEX, mock_tokenizer, gamma=0.5, beta=3.0)
    rs_proc = RustProcessor(REGEX, mock_tokenizer, gamma=0.5, beta=3.0)
    input_ids = torch.zeros(1, 0, dtype=torch.long)
    torch.manual_seed(42)
    scores = torch.randn(1, 100)

    scores_py = py_proc(input_ids, scores.clone())
    scores_rs = rs_proc(input_ids, scores.clone())

    assert torch.equal(scores_py == float("-inf"), scores_rs == float("-inf"))
    allowed = scores_py != float("-inf")
    assert torch.allclose(scores_py[allowed], scores_rs[allowed], atol=1e-5)


def test_counter_populated_compute_counts_equivalent(mock_tokenizer):
    """Rust compute_counts should match the Python counter semantics by token id."""
    py_proc = PyProcessor(REGEX, mock_tokenizer, gamma=0.5, beta=3.0)
    rs_proc = RustProcessor(REGEX, mock_tokenizer, gamma=0.5, beta=3.0)
    state = py_proc.guide.initial_state

    path_states = list(py_proc.guide.dfa.get_state_sequence("ab"))
    for transition in zip(path_states, path_states[1:]):
        py_proc.transition_counter[transition] += 1
    rs_proc.dfa.update_path_counter("ab")

    for _ in range(2):
        local_states = py_proc.guide.get_state_sequence_from_token_id(state, 10)
        for inner_state in local_states[1:]:
            py_proc.local_state_counter[inner_state] += 1
        rs_proc.dfa.update_local_state_counter(state, 10)

    py_counts = {}
    for token_id in py_proc.guide.get_next_instruction(state):
        state_seq = py_proc.guide.get_state_sequence_from_token_id(state, token_id)
        transitions = list(zip(state_seq, state_seq[1:]))
        reward_count = (
            min(py_proc.transition_counter[t] for t in transitions)
            if transitions
            else 0
        )
        local_count = max(
            (py_proc.local_state_counter[s] for s in state_seq[1:]), default=0
        )
        py_counts[token_id] = (reward_count, max(1, local_count))

    token_ids, reward_counts, penalty_counts = rs_proc.dfa.compute_counts(state)
    rs_counts = {
        token_id: (reward_count, penalty_count)
        for token_id, reward_count, penalty_count in zip(
            token_ids, reward_counts, penalty_counts, strict=True
        )
    }

    assert rs_counts == py_counts
