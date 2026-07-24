"""ICFEM-compatible single-obligation breadth-first witness construction."""

from collections import Counter, deque

from regex_dfa_guide import DiverseGuideDFA

from .obligations import CoverageObligation, ObligationCatalog


def get_coverage_obligations(
    dfa: DiverseGuideDFA,
    samples: list[str],
    cov_threshold: int = 1,
    catalog: ObligationCatalog | None = None,
) -> list[CoverageObligation]:
    """Return catalog entries whose legacy coverage count is below a threshold."""
    catalog = catalog or ObligationCatalog.from_dfa(dfa)
    state_count: Counter[int] = Counter()
    transition_count: Counter[tuple[int, int, int]] = Counter()
    path_count: Counter[tuple[int, int]] = Counter()
    for sample in samples:
        trans_seq = dfa.get_transition_sequence(sample)
        _, previous_state = trans_seq[0]
        state_count[previous_state] += 1
        for byte, state in trans_seq[1:]:
            state_count[state] += 1
            transition_count[(previous_state, byte, state)] += 1
            path_count[(previous_state, state)] += 1
            previous_state = state
    return [
        item
        for item in catalog
        if (
            (item.state is not None and state_count[item.state] < cov_threshold)
            or (
                item.transition is not None
                and transition_count[item.transition] < cov_threshold
            )
            or (item.path is not None and path_count[item.path] < cov_threshold)
        )
    ]


def construct_one_witness(
    dfa: DiverseGuideDFA, obligation: CoverageObligation
) -> list[int] | None:
    """Find the shortest accepted token sequence meeting one obligation."""
    initial_state = dfa.get_initial_state()
    queue = deque([(initial_state, [])])
    visited_states = {initial_state}
    matched: tuple[int, list[int]] | None = None
    if obligation.is_state_obligation() and obligation.state == initial_state:
        matched = (initial_state, [])

    while queue and matched is None:
        state, token_ids = queue.popleft()
        for token_id in dfa.get_allowed_token_ids(state):
            try:
                transitions = dfa.get_byte_transition_sequence_from_state(
                    state, token_id
                )
            except ValueError:
                next_state = dfa.get_next_token_state(state, token_id)
                transitions = []
            else:
                next_state = transitions[-1][1]
            previous_state = state
            meets_obligation = False
            for byte, byte_next_state in transitions[1:]:
                if obligation.is_met_by_transition(
                    (previous_state, byte, byte_next_state)
                ):
                    meets_obligation = True
                    break
                previous_state = byte_next_state
            next_token_ids = token_ids + [token_id]
            if meets_obligation:
                matched = (next_state, next_token_ids)
                break
            if next_state not in visited_states:
                visited_states.add(next_state)
                queue.append((next_state, next_token_ids))

    if matched is None:
        return None
    middle_state, prefix_token_ids = matched
    queue = deque([(middle_state, [])])
    visited_states = {middle_state}
    while queue:
        state, suffix_token_ids = queue.popleft()
        if dfa.is_final_state(state):
            return prefix_token_ids + suffix_token_ids
        for token_id in dfa.get_allowed_token_ids(state):
            next_state = dfa.get_next_token_state(state, token_id)
            if next_state not in visited_states:
                visited_states.add(next_state)
                queue.append((next_state, suffix_token_ids + [token_id]))
    return None
