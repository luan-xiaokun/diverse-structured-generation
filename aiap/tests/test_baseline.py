from aiap.baseline import construct_one_witness
from aiap.obligations import CoverageObligation


def test_initial_state_obligation_can_return_empty_witness(dfa_empty):
    initial = dfa_empty.get_initial_state()
    obligation = CoverageObligation("state", initial)
    result = construct_one_witness(dfa_empty, obligation)
    assert result.token_ids == []
    assert result.max_expansions is None
    assert result.limit_exhausted is False


def test_multi_byte_token_can_match_interior_byte(dfa_multi):
    initial = dfa_multi.get_initial_state()
    next_state = dfa_multi.get_next_byte_state(initial, ord("a"))
    final_state = dfa_multi.get_next_byte_state(next_state, ord("b"))
    result = construct_one_witness(
        dfa_multi, CoverageObligation("transition", (next_state, ord("b"), final_state))
    )
    assert result.token_ids == [10]
    assert result.expansions > 0


def test_exhaustive_search_without_accepting_completion_is_not_search_limit(
    dfa_non_completable,
):
    initial = dfa_non_completable.get_initial_state()
    a_state = dfa_non_completable.get_next_token_state(initial, 10)
    b_state = dfa_non_completable.get_next_token_state(a_state, 20)
    result = construct_one_witness(
        dfa_non_completable, CoverageObligation("state", b_state)
    )
    assert result.token_ids is None
    assert result.max_expansions is None
    assert result.limit_exhausted is False
    assert result.failure_reason == "no_matching_obligation"


def test_sorted_token_ids_choose_same_shortest_witness(dfa_multi):
    initial = dfa_multi.get_initial_state()
    interior = dfa_multi.get_next_byte_state(initial, ord("a"))
    final = dfa_multi.get_next_byte_state(interior, ord("b"))

    class ReversedAllowedTokens:
        def __getattr__(self, name):
            return getattr(dfa_multi, name)

        def get_allowed_token_ids(self, state):
            return list(reversed(dfa_multi.get_allowed_token_ids(state)))

    obligation = CoverageObligation("transition", (interior, ord("b"), final))
    assert construct_one_witness(dfa_multi, obligation).token_ids == [10]
    assert construct_one_witness(ReversedAllowedTokens(), obligation).token_ids == [10]


def test_explicit_exhausted_cap_is_recorded_as_search_limit_input(dfa_multi):
    initial = dfa_multi.get_initial_state()
    interior = dfa_multi.get_next_byte_state(initial, ord("a"))
    final = dfa_multi.get_next_byte_state(interior, ord("b"))

    result = construct_one_witness(
        dfa_multi,
        CoverageObligation("transition", (interior, ord("b"), final)),
        max_expansions=0,
    )

    assert result.token_ids is None
    assert result.max_expansions == 0
    assert result.expansions == 0
    assert result.limit_exhausted is True
    assert result.failure_reason == "expansion_limit"
