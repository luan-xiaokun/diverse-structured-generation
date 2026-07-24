"""Accepting token-action reachability for already-missing obligations."""

from __future__ import annotations

from collections import defaultdict, deque
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Literal

from regex_dfa_guide import DiverseGuideDFA

from .obligations import CoverageObligation

ReachabilityCategory = Literal["byte_level_unreachable"] | None


@dataclass(frozen=True)
class TokenGraphEdge:
    """One exact token action annotated with its interior byte transitions."""

    token_id: int
    source_state: int
    destination_state: int
    byte_steps: tuple[tuple[int, int, int], ...]


@dataclass(frozen=True)
class ReachabilityEvidence:
    """Deterministic evidence produced by the accepting token-action graph."""

    algorithm: str
    forward_state_count: int
    coreachable_state_count: int
    matching_accepting_action_count: int


@dataclass(frozen=True)
class ClassifiedObligation:
    """An already-missing obligation plus its accepting-token feasibility proof."""

    obligation: CoverageObligation
    reachable: bool
    category: ReachabilityCategory
    evidence: ReachabilityEvidence


def _token_graph_edges(dfa: DiverseGuideDFA) -> list[TokenGraphEdge]:
    edges: list[TokenGraphEdge] = []
    for source_state in sorted(int(state) for state in dfa.get_states()):
        for token_id in sorted(
            int(token) for token in dfa.get_allowed_token_ids(source_state)
        ):
            try:
                raw_steps = dfa.get_byte_transition_sequence_from_state(
                    source_state, token_id
                )
                destination_state = int(
                    dfa.get_next_token_state(source_state, token_id)
                )
            except ValueError:
                # EOS-like token actions have no byte trace but still participate in
                # accepting token paths.
                try:
                    destination_state = int(
                        dfa.get_next_token_state(source_state, token_id)
                    )
                except ValueError:
                    continue
                byte_steps: tuple[tuple[int, int, int], ...] = ()
            else:
                previous_state = source_state
                steps: list[tuple[int, int, int]] = []
                for byte, next_state in raw_steps[1:]:
                    next_state = int(next_state)
                    steps.append((previous_state, int(byte), next_state))
                    previous_state = next_state
                byte_steps = tuple(steps)
                if byte_steps and byte_steps[-1][2] != destination_state:
                    continue
            edges.append(
                TokenGraphEdge(token_id, source_state, destination_state, byte_steps)
            )
    return edges


def _reachable_states(initial_state: int, edges: Iterable[TokenGraphEdge]) -> set[int]:
    outgoing: dict[int, list[int]] = defaultdict(list)
    for edge in edges:
        outgoing[edge.source_state].append(edge.destination_state)
    seen = {initial_state}
    queue = deque([initial_state])
    while queue:
        state = queue.popleft()
        for destination in outgoing[state]:
            if destination not in seen:
                seen.add(destination)
                queue.append(destination)
    return seen


def _coreachable_states(
    final_states: Iterable[int], edges: Iterable[TokenGraphEdge]
) -> set[int]:
    incoming: dict[int, list[int]] = defaultdict(list)
    for edge in edges:
        incoming[edge.destination_state].append(edge.source_state)
    seen = {int(state) for state in final_states}
    queue = deque(seen)
    while queue:
        state = queue.popleft()
        for source in incoming[state]:
            if source not in seen:
                seen.add(source)
                queue.append(source)
    return seen


def _matching_action_counts(
    obligations: Iterable[CoverageObligation], edges: Iterable[TokenGraphEdge]
) -> dict[str, int]:
    """Count accepting actions matching each supplied missing obligation.

    The direct formulation checked every obligation against every accepting token
    action.  On a real tokenizer that makes classification quadratic in the
    number of missing DFA structures and token actions.  Index only the supplied
    missing meanings, then scan each action once.  Each action contributes at
    most one count per meaning, matching the boolean semantics of the former
    per-edge predicate.
    """

    supplied = list(obligations)
    state_ids = {item.state for item in supplied if item.is_state_obligation()}
    transition_ids = {
        item.transition for item in supplied if item.is_transition_obligation()
    }
    path_ids = {item.path for item in supplied if item.is_path_obligation()}
    counts = {item.obligation_id: 0 for item in supplied}

    for edge in edges:
        action_states = {edge.source_state, edge.destination_state}
        action_states.update(target for _, _, target in edge.byte_steps)
        for state in action_states & state_ids:
            counts[f"state:{state}"] += 1

        transitions = set(edge.byte_steps)
        for transition in transitions & transition_ids:
            source, byte, target = transition
            counts[f"transition:{source}:{byte}:{target}"] += 1

        paths = {(source, target) for source, _, target in edge.byte_steps}
        for source, target in paths & path_ids:
            counts[f"path:{source}:{target}"] += 1

    return counts


def classify_reachability(
    dfa: DiverseGuideDFA, obligations: Iterable[CoverageObligation]
) -> list[ClassifiedObligation]:
    """Classify only supplied missing obligations on accepting token actions.

    This deliberately does not inspect or construct an obligation catalog: callers
    must derive the missing set from replayed selected baseline samples first.
    """

    missing = list(obligations)
    edges = _token_graph_edges(dfa)
    forward = _reachable_states(int(dfa.get_initial_state()), edges)
    coreachable = _coreachable_states(
        (int(state) for state in dfa.get_states() if dfa.is_final_state(state)), edges
    )
    accepting_edges = [
        edge
        for edge in edges
        if edge.source_state in forward and edge.destination_state in coreachable
    ]
    matching_counts = _matching_action_counts(missing, accepting_edges)
    return [
        ClassifiedObligation(
            obligation=obligation,
            reachable=bool(matches := matching_counts[obligation.obligation_id]),
            category=None if matches else "byte_level_unreachable",
            evidence=ReachabilityEvidence(
                algorithm="annotated_token_graph_1",
                forward_state_count=len(forward),
                coreachable_state_count=len(coreachable),
                matching_accepting_action_count=matches,
            ),
        )
        for obligation in missing
    ]
