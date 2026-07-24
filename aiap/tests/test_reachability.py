from aiap.obligations import CoverageObligation, ObligationCatalog
from aiap.reachability import classify_reachability


def _byte_trace(dfa):
    initial = dfa.get_initial_state()
    a_state = dfa.get_next_byte_state(initial, ord("a"))
    final_state = dfa.get_next_byte_state(a_state, ord("b"))
    return initial, a_state, final_state


def test_classifies_all_obligation_kinds_and_interior_byte_steps(dfa_multi):
    initial, interior, final = _byte_trace(dfa_multi)
    obligations = [
        CoverageObligation("state", interior),
        CoverageObligation("transition", (interior, ord("b"), final)),
        CoverageObligation("path", (interior, final)),
    ]

    records = classify_reachability(dfa_multi, obligations)

    assert [record.obligation for record in records] == obligations
    assert all(record.reachable for record in records)
    assert all(record.category is None for record in records)
    assert all(
        record.evidence.algorithm == "annotated_token_graph_1" for record in records
    )
    assert all(
        record.evidence.matching_accepting_action_count > 0 for record in records
    )


def test_requires_prefix_and_accepting_suffix(dfa_non_completable):
    initial = dfa_non_completable.get_initial_state()
    blocked = dfa_non_completable.get_next_token_state(initial, 10)
    final = dfa_non_completable.get_next_token_state(blocked, 20)

    record = classify_reachability(
        dfa_non_completable, [CoverageObligation("state", final)]
    )[0]

    assert not record.reachable
    assert record.category == "byte_level_unreachable"
    assert record.evidence.forward_state_count > 0
    assert record.evidence.coreachable_state_count > 0
    assert record.evidence.matching_accepting_action_count == 0


def test_raw_byte_structure_without_token_action_is_unreachable(dfa_multi):
    initial, interior, final = _byte_trace(dfa_multi)

    class NoTokenActions:
        def __getattr__(self, name):
            return getattr(dfa_multi, name)

        def get_allowed_token_ids(self, state):
            return []

    record = classify_reachability(
        NoTokenActions(),
        [CoverageObligation("transition", (interior, ord("b"), final))],
    )[0]

    assert not record.reachable
    assert record.category == "byte_level_unreachable"
    assert record.evidence.matching_accepting_action_count == 0
    assert record.evidence.forward_state_count == 1


def test_preserves_input_order_and_never_adds_catalog_items(dfa_multi):
    initial, interior, final = _byte_trace(dfa_multi)
    covered_catalog_item = CoverageObligation("state", initial)
    supplied = [
        CoverageObligation("path", (interior, final)),
        CoverageObligation("state", interior),
    ]

    records = classify_reachability(dfa_multi, supplied)

    assert [record.obligation.obligation_id for record in records] == [
        item.obligation_id for item in supplied
    ]
    assert covered_catalog_item.obligation_id not in {
        record.obligation.obligation_id for record in records
    }
    assert len(records) == len(supplied)
    assert covered_catalog_item in ObligationCatalog.from_dfa(dfa_multi)
