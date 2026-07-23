from aiap.baseline import construct_one_witness
from aiap.obligations import CoverageObligation


def test_initial_state_obligation_can_return_empty_witness(dfa_empty):
    initial = dfa_empty.get_initial_state()
    obligation = CoverageObligation("state", initial)
    assert construct_one_witness(dfa_empty, obligation) == []


def test_multi_byte_token_can_match_interior_byte(dfa_multi):
    initial = dfa_multi.get_initial_state()
    next_state = dfa_multi.get_next_byte_state(initial, ord("a"))
    final_state = dfa_multi.get_next_byte_state(next_state, ord("b"))
    witness = construct_one_witness(
        dfa_multi, CoverageObligation("transition", (next_state, ord("b"), final_state))
    )
    assert witness == [10]


def test_reachable_obligation_without_accepting_completion_returns_none(dfa_non_completable):
    initial = dfa_non_completable.get_initial_state()
    a_state = dfa_non_completable.get_next_token_state(initial, 10)
    b_state = dfa_non_completable.get_next_token_state(a_state, 20)
    assert construct_one_witness(
        dfa_non_completable, CoverageObligation("state", b_state)
    ) is None
