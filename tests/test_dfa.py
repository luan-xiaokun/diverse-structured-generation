"""Tests for DiverseGuideDFA (Rust extension)."""

import pickle

import pytest
import torch

from diverse_guide.guide_python import DiverseRegexLogitsProcessor
from regex_dfa_guide import DiverseGuideDFA

# ---------------------------------------------------------------------------
# Construction and basic state queries
# ---------------------------------------------------------------------------


def test_construction(dfa_multi):
    assert dfa_multi is not None


def test_initial_state_is_initial(dfa_multi):
    initial = dfa_multi.get_initial_state()
    assert dfa_multi.is_initial_state(initial)


def test_initial_state_not_final(dfa_multi):
    # (?:[ab]+)$ requires at least one char, so empty string doesn't match
    initial = dfa_multi.get_initial_state()
    assert not dfa_multi.is_final_state(initial)


def test_get_states_nonempty(dfa_multi):
    states = dfa_multi.get_states()
    assert len(states) >= 2  # at least initial and one accepting state


def test_get_final_states(dfa_multi):
    finals = dfa_multi.get_final_states()
    assert len(finals) >= 1
    for s in finals:
        assert dfa_multi.is_final_state(s)


def test_initial_not_in_final_states(dfa_multi):
    initial = dfa_multi.get_initial_state()
    finals = dfa_multi.get_final_states()
    assert initial not in finals


# ---------------------------------------------------------------------------
# Token transitions
# ---------------------------------------------------------------------------


def test_allowed_token_ids_at_initial(dfa_multi):
    initial = dfa_multi.get_initial_state()
    allowed = dfa_multi.get_allowed_token_ids(initial)
    assert sorted(allowed) == [10, 20]  # tokens for 'a' and 'b'


def test_eos_allowed_at_final_state(dfa_multi):
    initial = dfa_multi.get_initial_state()
    # Transition to a final state via token 10 ('a')
    next_state = dfa_multi.get_next_token_state(initial, 10)
    assert dfa_multi.is_final_state(next_state)
    allowed = dfa_multi.get_allowed_token_ids(next_state)
    assert 99 in allowed  # EOS should be allowed at accepting state


def test_get_next_token_state(dfa_multi):
    initial = dfa_multi.get_initial_state()
    state_after_a = dfa_multi.get_next_token_state(initial, 10)
    state_after_b = dfa_multi.get_next_token_state(initial, 20)
    assert state_after_a != initial
    assert state_after_b != initial


def test_distance_to_accepting_matches_simple_regex(dfa_multi):
    initial = dfa_multi.get_initial_state()
    final_state = dfa_multi.get_next_token_state(initial, 10)

    assert dfa_multi.get_distance_to_accepting(initial) == 1
    assert dfa_multi.get_distance_to_accepting(final_state) == 0


# ---------------------------------------------------------------------------
# Byte-level transitions
# ---------------------------------------------------------------------------


def test_get_next_byte_state(dfa_multi):
    initial = dfa_multi.get_initial_state()
    state_after_a = dfa_multi.get_next_byte_state(initial, ord("a"))
    assert state_after_a != initial


def test_get_allowed_bytes_at_initial(dfa_multi):
    initial = dfa_multi.get_initial_state()
    allowed_bytes = dfa_multi.get_allowed_bytes(initial)
    assert ord("a") in allowed_bytes
    assert ord("b") in allowed_bytes


# ---------------------------------------------------------------------------
# State and transition sequences
# ---------------------------------------------------------------------------


def test_get_state_sequence_length(dfa_multi):
    seq = dfa_multi.get_state_sequence("a")
    # One state before consuming any byte + one state after consuming 'a'
    assert len(seq) == 2


def test_get_state_sequence_ends_at_final(dfa_multi):
    seq = dfa_multi.get_state_sequence("a")
    assert dfa_multi.is_final_state(seq[-1])


def test_get_state_sequence_starts_at_initial(dfa_multi):
    initial = dfa_multi.get_initial_state()
    seq = dfa_multi.get_state_sequence("a")
    assert seq[0] == initial


def test_get_state_sequence_multi_char(dfa_multi):
    seq = dfa_multi.get_state_sequence("ab")
    assert len(seq) == 3
    assert dfa_multi.is_final_state(seq[-1])


def test_get_transition_sequence(dfa_multi):
    trans = dfa_multi.get_transition_sequence("a")
    # Format: [(dummy_byte, initial_state), (ord('a'), next_state), ...]
    assert len(trans) >= 2
    # The last entry's state should be final
    _, last_state = trans[-1]
    assert dfa_multi.is_final_state(last_state)


def test_get_byte_transition_sequence(dfa_multi):
    trans = dfa_multi.get_byte_transition_sequence("a")
    assert len(trans) >= 1


def test_get_transitions_dict(dfa_multi):
    transitions = dfa_multi.get_transitions()
    assert isinstance(transitions, dict)
    assert len(transitions) >= 1
    initial = dfa_multi.get_initial_state()
    assert initial in transitions
    # Initial state must have transitions for bytes 97 ('a') and 98 ('b')
    initial_trans = transitions[initial]
    assert ord("a") in initial_trans
    assert ord("b") in initial_trans


# ---------------------------------------------------------------------------
# Path and state counters
# ---------------------------------------------------------------------------


def test_update_and_compute_path_counter(dfa_multi):
    # After updating path counter for 'a', compute_counts should reflect it
    dfa_multi.update_path_counter("a")
    initial = dfa_multi.get_initial_state()
    token_ids, reward_counts, penalty_counts = dfa_multi.compute_counts(initial)
    assert 10 in token_ids  # 'a' token
    idx = list(token_ids).index(10)
    assert reward_counts[idx] >= 1


def test_reset_path_counter(dfa_multi):
    dfa_multi.update_path_counter("a")
    dfa_multi.reset_path_counter()
    initial = dfa_multi.get_initial_state()
    token_ids, reward_counts, _ = dfa_multi.compute_counts(initial)
    assert all(c == 0 for c in reward_counts)


def test_update_and_reset_local_state_counter(dfa_multi):
    initial = dfa_multi.get_initial_state()

    # Multiple updates to raise the penalty count above the baseline of 1
    for _ in range(3):
        dfa_multi.update_local_state_counter(initial, 10)

    token_ids, _, penalty_counts = dfa_multi.compute_counts(initial)
    idx = list(token_ids).index(10)
    # compute_counts starts maximal_local_state_count at 1;
    # after 3 increments it should be > 1
    assert penalty_counts[idx] > 1

    dfa_multi.reset_local_state_counter()
    _, _, penalty_counts_after = dfa_multi.compute_counts(initial)
    # After reset, local_state_counter is empty → baseline of 1 for every token
    assert all(c == 1 for c in penalty_counts_after)


def _shifted_harmonic_mean(counts: list[int]) -> float:
    if not counts:
        return 0.0
    reciprocal_sum = sum(1.0 / (count + 1.0) for count in counts)
    return (len(counts) / reciprocal_sum) - 1.0


def _expected_adjustments_from_counts(
    reward_counts: list[float],
    penalty_counts: list[int],
    no_reward: bool = False,
    no_penalty: bool = False,
) -> torch.Tensor:
    reward_tensor = torch.tensor(reward_counts, dtype=torch.float)
    penalty_tensor = torch.tensor(penalty_counts, dtype=torch.float)
    reward = torch.log(1 + reward_tensor.sum()) / (1 + reward_tensor)
    penalty = penalty_tensor**2
    if no_reward:
        reward = torch.ones_like(reward_tensor)
    if no_penalty:
        penalty = torch.ones_like(penalty_tensor)
    return reward / penalty


def test_shifted_harmonic_mean_helper_matches_formula():
    assert DiverseRegexLogitsProcessor._shifted_harmonic_mean([2, 1]) == pytest.approx(
        _shifted_harmonic_mean([2, 1])
    )


def test_compute_adjustments_matches_python_formula(dfa_multi):
    initial = dfa_multi.get_initial_state()
    dfa_multi.update_path_counter("ab")
    dfa_multi.update_local_state_counter(initial, 10)
    dfa_multi.update_local_state_counter(initial, 10)

    token_ids, reward_counts, penalty_counts = dfa_multi.compute_counts(initial)
    adjustment_token_ids, adjustments = dfa_multi.compute_adjustments(
        initial, 1.0, False, False, 0.0
    )

    assert adjustment_token_ids == token_ids
    expected = _expected_adjustments_from_counts(
        list(reward_counts), list(penalty_counts)
    )
    actual = torch.tensor(adjustments, dtype=torch.float)
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize(
    ("no_reward", "no_penalty"),
    [(True, False), (False, True), (True, True)],
)
def test_compute_adjustments_ablation_flags_match_python_formula(
    dfa_multi, no_reward, no_penalty
):
    initial = dfa_multi.get_initial_state()
    dfa_multi.update_path_counter("ab")
    dfa_multi.update_local_state_counter(initial, 10)
    dfa_multi.update_local_state_counter(initial, 10)

    token_ids, reward_counts, penalty_counts = dfa_multi.compute_counts(initial)
    adjustment_token_ids, adjustments = dfa_multi.compute_adjustments(
        initial, 1.0, no_reward, no_penalty, 0.0
    )

    assert adjustment_token_ids == token_ids
    expected = _expected_adjustments_from_counts(
        list(reward_counts),
        list(penalty_counts),
        no_reward=no_reward,
        no_penalty=no_penalty,
    )
    actual = torch.tensor(adjustments, dtype=torch.float)
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)


def test_compute_adjustments_invalid_state_raises_value_error(dfa_multi):
    with pytest.raises(ValueError):
        dfa_multi.compute_adjustments(2**32 - 1, 1.0, False, False, 0.0)


def test_compute_adjustments_completion_bias_bonus():
    vocab = {
        99: "<eos>",
        10: "a",
        15: "b",
        20: "ab",
    }
    dfa = DiverseGuideDFA("(?:ab)$", 99, vocab)
    initial = dfa.get_initial_state()
    token_ids, adjustments = dfa.compute_adjustments(initial, 3.0, False, False, 0.2)

    by_token = dict(zip(token_ids, adjustments, strict=True))
    assert by_token[10] == pytest.approx(0.0)
    assert by_token[20] == pytest.approx(0.1, abs=1e-6)


# ---------------------------------------------------------------------------
# fork()
# ---------------------------------------------------------------------------


def test_fork_equality(dfa_multi):
    forked = dfa_multi.fork()
    assert dfa_multi == forked


def test_fork_counter_independence(dfa_multi):
    forked = dfa_multi.fork()
    dfa_multi.update_path_counter("a")
    # Counters differ now, so they should not be equal
    assert dfa_multi != forked


def test_fork_preserves_structure(dfa_multi):
    forked = dfa_multi.fork()
    initial = dfa_multi.get_initial_state()
    assert forked.get_initial_state() == initial
    assert sorted(forked.get_allowed_token_ids(initial)) == [10, 20]


def test_fork_then_update_original_does_not_affect_fork(dfa_multi, vocab_ab):
    # Both guide original and fork share the same index, structural queries should agree
    forked = dfa_multi.fork()
    dfa_multi.update_path_counter("a")
    # Structure (token transitions) should still be identical
    initial = dfa_multi.get_initial_state()
    assert sorted(forked.get_allowed_token_ids(initial)) == sorted(
        dfa_multi.get_allowed_token_ids(initial)
    )


# ---------------------------------------------------------------------------
# Equality
# ---------------------------------------------------------------------------


def test_two_identical_dfas_are_equal(vocab_ab):
    dfa1 = DiverseGuideDFA("(?:[ab]+)$", 99, vocab_ab)
    dfa2 = DiverseGuideDFA("(?:[ab]+)$", 99, vocab_ab)
    assert dfa1 == dfa2


def test_different_regex_not_equal(vocab_ab):
    dfa1 = DiverseGuideDFA("(?:[ab]+)$", 99, vocab_ab)
    dfa2 = DiverseGuideDFA("(?:[ab])$", 99, vocab_ab)
    assert dfa1 != dfa2


# ---------------------------------------------------------------------------
# Pickle (serialization round-trip)
# ---------------------------------------------------------------------------


def test_pickle_roundtrip(dfa_multi):
    data = pickle.dumps(dfa_multi)
    restored = pickle.loads(data)
    assert restored == dfa_multi


def test_pickle_roundtrip_with_counters(dfa_multi):
    initial = dfa_multi.get_initial_state()
    token_id = dfa_multi.get_allowed_token_ids(initial)[0]
    dfa_multi.update_path_counter("ab")
    dfa_multi.update_local_state_counter(initial, token_id)

    allowed_before = dfa_multi.get_allowed_token_ids(initial)
    next_state_before = dfa_multi.get_next_token_state(initial, token_id)
    byte_state_sequence_before = dfa_multi.get_byte_state_sequence(initial, token_id)
    counts_before = dfa_multi.compute_counts(initial)

    data = pickle.dumps(dfa_multi)
    restored = pickle.loads(data)

    assert restored == dfa_multi
    assert restored.get_allowed_token_ids(initial) == allowed_before
    assert restored.get_next_token_state(initial, token_id) == next_state_before
    assert (
        restored.get_byte_state_sequence(initial, token_id)
        == byte_state_sequence_before
    )
    assert restored.compute_counts(initial) == counts_before
