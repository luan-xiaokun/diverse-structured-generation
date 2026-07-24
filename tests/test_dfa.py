"""Tests for DiverseGuideDFA (Rust extension)."""

import pickle
import socket

import pytest

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


@pytest.mark.parametrize(
    ("value", "actual_type"),
    [("x", "str"), (bytearray(b"x"), "bytearray"), ([120], "list")],
)
def test_constructor_rejects_non_bytes_vocabulary_values(value, actual_type):
    with pytest.raises(TypeError) as exc_info:
        DiverseGuideDFA("(?:x)$", 99, {7: value})

    message = str(exc_info.value)
    assert "7" in message
    assert "bytes" in message
    assert actual_type in message


def test_token_trace_uses_exact_mixed_ascii_and_high_bytes():
    dfa = DiverseGuideDFA("(?:A\u1680)$", 99, {7: b"A\xe1\x9a\x80"})
    initial = dfa.get_initial_state()

    expected = dfa.get_byte_transition_sequence("A\u1680")
    actual = dfa.get_byte_transition_sequence_from_state(initial, 7)

    assert actual == expected
    assert [byte for byte, _ in actual] == [0, ord("A"), 0xE1, 0x9A, 0x80]
    assert dfa.get_byte_state_sequence(initial, 7) == [state for _, state in expected]
    final_state = actual[-1][1]
    assert dfa.get_byte_transition_sequence_from_state(final_state, 99) == [
        (0, final_state)
    ]


# ---------------------------------------------------------------------------
# Pickle (serialization round-trip)
# ---------------------------------------------------------------------------


def test_pickle_roundtrip(dfa_multi):
    data = pickle.dumps(dfa_multi)
    restored = pickle.loads(data)
    assert restored == dfa_multi


def test_pickle_roundtrip_with_counters(dfa_multi):
    dfa_multi.update_path_counter("ab")
    data = pickle.dumps(dfa_multi)
    restored = pickle.loads(data)
    assert restored == dfa_multi


# ---------------------------------------------------------------------------
# Strict schema-1 artifact boundary
# ---------------------------------------------------------------------------


def _artifact_bytes(dfa):
    _restore, args = dfa.__reduce__()
    return args[0]


def _assert_artifact_error(payload, prefix):
    with pytest.raises(ValueError, match=rf"^{prefix}"):
        DiverseGuideDFA.from_binary(payload)


def test_reduce_uses_schema_1_artifact_envelope(dfa_multi):
    payload = _artifact_bytes(dfa_multi)

    assert payload[:8] == b"RDGDFA\x00\x00"
    assert payload[8:10] == (1).to_bytes(2, "little")


@pytest.mark.parametrize("payload", [b"", b"not-an-artifact"])
def test_from_binary_rejects_legacy_artifacts(payload):
    _assert_artifact_error(payload, "legacy_artifact")


def test_from_binary_rejects_unknown_schema(dfa_multi):
    payload = bytearray(_artifact_bytes(dfa_multi))
    payload[8:10] = (2).to_bytes(2, "little")

    _assert_artifact_error(bytes(payload), "unsupported_schema: 2")


@pytest.mark.parametrize(
    "payload",
    [
        b"RDGDFA\x00\x00",
        b"RDGDFA\x00\x00\x01",
        b"RDGDFA\x00\x00\x01\x00",
        b"RDGDFA\x00\x00\x01\x00\xff",
    ],
)
def test_from_binary_rejects_truncated_or_malformed_artifacts(payload):
    _assert_artifact_error(payload, "corrupt_artifact")


def test_from_binary_rejects_trailing_bytes(dfa_multi):
    _assert_artifact_error(_artifact_bytes(dfa_multi) + b"\x00", "corrupt_artifact")


def test_from_binary_restores_behavior_without_network_access(dfa_multi, monkeypatch):
    initial = dfa_multi.get_initial_state()
    dfa_multi.update_path_counter("ab")
    dfa_multi.update_local_state_counter(initial, 10)
    expected = {
        "states": sorted(dfa_multi.get_states()),
        "transitions": dfa_multi.get_transitions(),
        "allowed": {
            state: sorted(dfa_multi.get_allowed_token_ids(state))
            for state in dfa_multi.get_states()
        },
        "direct_trace": dfa_multi.get_byte_transition_sequence("ab"),
        "token_trace": dfa_multi.get_byte_transition_sequence_from_state(initial, 10),
        "acceptance": dfa_multi.is_final_state(
            dfa_multi.get_next_token_state(initial, 10)
        ),
        "counts": dfa_multi.compute_counts(initial),
    }
    payload = _artifact_bytes(dfa_multi)
    del dfa_multi

    def fail_if_network_used(*_args, **_kwargs):
        raise AssertionError("artifact restore must not access the network")

    monkeypatch.setattr(socket, "create_connection", fail_if_network_used)
    restored = DiverseGuideDFA.from_binary(payload)

    assert sorted(restored.get_states()) == expected["states"]
    assert restored.get_transitions() == expected["transitions"]
    assert {
        state: sorted(restored.get_allowed_token_ids(state))
        for state in restored.get_states()
    } == expected["allowed"]
    assert restored.get_byte_transition_sequence("ab") == expected["direct_trace"]
    assert (
        restored.get_byte_transition_sequence_from_state(initial, 10)
        == expected["token_trace"]
    )
    assert (
        restored.is_final_state(restored.get_next_token_state(initial, 10))
        == expected["acceptance"]
    )
    assert restored.compute_counts(initial) == expected["counts"]
