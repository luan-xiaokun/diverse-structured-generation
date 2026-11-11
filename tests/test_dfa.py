"""Tests for DiverseGuideDFA (Rust extension) through its Python bindings."""

import pickle

import pytest
import torch

from regex_dfa_guide import DiverseGuideDFA


def test_state_and_byte_level_queries(dfa_multi):
    # (?:[ab]+)$ requires at least one character.
    initial = dfa_multi.get_initial_state()
    assert dfa_multi.is_initial_state(initial)
    assert not dfa_multi.is_final_state(initial)
    assert len(dfa_multi.get_states()) >= 2
    finals = dfa_multi.get_final_states()
    assert finals and initial not in finals
    assert all(dfa_multi.is_final_state(s) for s in finals)

    seq = dfa_multi.get_state_sequence("a")
    assert len(seq) == 2 and seq[0] == initial
    assert dfa_multi.is_final_state(seq[-1])
    seq_ab = dfa_multi.get_state_sequence("ab")
    assert len(seq_ab) == 3 and dfa_multi.is_final_state(seq_ab[-1])

    # Format: [(dummy_byte, initial_state), (ord('a'), next_state), ...]
    trans = dfa_multi.get_transition_sequence("a")
    assert len(trans) >= 2 and dfa_multi.is_final_state(trans[-1][1])
    assert len(dfa_multi.get_byte_transition_sequence("a")) >= 1

    assert dfa_multi.get_next_byte_state(initial, ord("a")) == seq[1]
    allowed_bytes = dfa_multi.get_allowed_bytes(initial)
    transitions = dfa_multi.get_transitions()
    assert {ord("a"), ord("b")} <= set(allowed_bytes)
    assert {ord("a"), ord("b")} <= set(transitions[initial])


def test_token_transitions_and_eos(dfa_multi):
    initial = dfa_multi.get_initial_state()
    assert sorted(dfa_multi.get_allowed_token_ids(initial)) == [10, 20]
    assert dfa_multi.get_next_token_state(initial, 20) != initial

    after_a = dfa_multi.get_next_token_state(initial, 10)
    assert dfa_multi.is_final_state(after_a)
    assert 99 in dfa_multi.get_allowed_token_ids(after_a)


def test_path_and_local_counters_update_and_reset(dfa_multi):
    initial = dfa_multi.get_initial_state()
    dfa_multi.update_path_counter("a")
    token_ids, reward_counts, _ = dfa_multi.compute_counts(initial)
    assert reward_counts[list(token_ids).index(10)] >= 1
    dfa_multi.reset_path_counter()
    assert all(c == 0 for c in dfa_multi.compute_counts(initial)[1])

    for _ in range(3):
        dfa_multi.update_local_state_counter(initial, 10)
    token_ids, _, penalty_counts = dfa_multi.compute_counts(initial)
    # The penalty count starts at max(1, .), so three visits raise it above 1.
    assert penalty_counts[list(token_ids).index(10)] > 1
    dfa_multi.reset_local_state_counter()
    assert all(c == 1 for c in dfa_multi.compute_counts(initial)[2])


def _expected_adjustments_from_counts(
    reward_counts: list[int],
    penalty_counts: list[int],
    beta: float,
    no_reward: bool = False,
    no_penalty: bool = False,
) -> torch.Tensor:
    reward_tensor = torch.tensor(reward_counts, dtype=torch.float)
    penalty_tensor = torch.tensor(penalty_counts, dtype=torch.float)
    reward = torch.log(1 + reward_tensor.sum()) / (1 + reward_tensor)
    penalty = beta * penalty_tensor**2
    if no_reward:
        reward = torch.ones_like(reward_tensor)
    if no_penalty:
        penalty = torch.ones_like(penalty_tensor)
    return reward / penalty


def test_compute_adjustments_matches_python_formula(dfa_multi):
    initial = dfa_multi.get_initial_state()
    dfa_multi.update_path_counter("ab")
    dfa_multi.update_local_state_counter(initial, 10)
    dfa_multi.update_local_state_counter(initial, 10)
    token_ids, reward_counts, penalty_counts = dfa_multi.compute_counts(initial)

    for no_reward, no_penalty in [
        (False, False),
        (True, False),
        (False, True),
        (True, True),
    ]:
        adjustment_token_ids, adjustments = dfa_multi.compute_adjustments(
            initial, 3.0, no_reward, no_penalty
        )
        assert adjustment_token_ids == token_ids
        expected = _expected_adjustments_from_counts(
            list(reward_counts),
            list(penalty_counts),
            beta=3.0,
            no_reward=no_reward,
            no_penalty=no_penalty,
        )
        actual = torch.tensor(adjustments, dtype=torch.float)
        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)

    with pytest.raises(ValueError):
        dfa_multi.compute_adjustments(2**32 - 1, 3.0, False, False)


def test_fork_shares_structure_but_not_counters(dfa_multi):
    forked = dfa_multi.fork()
    assert dfa_multi == forked
    dfa_multi.update_path_counter("a")
    assert dfa_multi != forked

    initial = dfa_multi.get_initial_state()
    assert forked.get_initial_state() == initial
    assert sorted(forked.get_allowed_token_ids(initial)) == [10, 20]


def test_equality_and_pickle_roundtrip(dfa_multi, vocab_ab):
    assert DiverseGuideDFA("(?:[ab]+)$", 99, vocab_ab) == dfa_multi
    assert DiverseGuideDFA("(?:[ab])$", 99, vocab_ab) != dfa_multi

    initial = dfa_multi.get_initial_state()
    token_id = dfa_multi.get_allowed_token_ids(initial)[0]
    dfa_multi.update_path_counter("ab")
    dfa_multi.update_local_state_counter(initial, token_id)

    restored = pickle.loads(pickle.dumps(dfa_multi))

    assert restored == dfa_multi
    assert restored.get_allowed_token_ids(initial) == dfa_multi.get_allowed_token_ids(
        initial
    )
    assert restored.get_next_token_state(
        initial, token_id
    ) == dfa_multi.get_next_token_state(initial, token_id)
    assert restored.get_byte_state_sequence(
        initial, token_id
    ) == dfa_multi.get_byte_state_sequence(initial, token_id)
    assert restored.compute_counts(initial) == dfa_multi.compute_counts(initial)
