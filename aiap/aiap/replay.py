"""Canonical, byte-level replay certificates for generated token candidates."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Iterable

from regex_dfa_guide import DiverseGuideDFA

from .obligations import CoverageObligation, ObligationCatalog


@dataclass(frozen=True)
class ByteStep:
    byte: int
    from_state: int
    to_state: int


@dataclass(frozen=True)
class TokenAction:
    token_id: int
    from_state: int
    to_state: int | None
    byte_steps: tuple[ByteStep, ...] = ()


@dataclass(frozen=True)
class FailureDiagnostic:
    category: str
    message: str
    token_index: int | None = None


@dataclass(frozen=True)
class CoveredObligation:
    id: str
    kind: str
    meaning: int | list[int]

    @classmethod
    def from_obligation(cls, item: CoverageObligation) -> "CoveredObligation":
        return cls(item.obligation_id, item.kind, item.raw_meaning)

    @property
    def raw_meaning(self) -> int | list[int]:
        return self.meaning


@dataclass
class CandidateCertificate:
    """Complete replay evidence; failed certificates are never aggregate eligible."""

    token_ids: list[int]
    initial_state: int
    final_state: int | None
    token_actions: list[TokenAction]
    accepted: bool
    aggregate_eligible: bool
    covered_obligations: list[CoveredObligation] = field(default_factory=list)
    failure: FailureDiagnostic | None = None
    source_text: str | None = None
    decoded_text: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    state_events: list[int] = field(default_factory=list)
    transition_events: list[tuple[int, int, int]] = field(default_factory=list)
    path_events: list[tuple[int, int]] = field(default_factory=list)

    @property
    def covered_obligation_ids(self) -> list[str]:
        return [item.id for item in self.covered_obligations]

    @property
    def is_accepted(self) -> bool:
        return self.accepted


def _failure(category: str, error: Exception, index: int | None = None) -> FailureDiagnostic:
    return FailureDiagnostic(category, str(error), index)


def replay_candidate(
    dfa: DiverseGuideDFA,
    token_ids: Iterable[int],
    catalog: ObligationCatalog | None = None,
    *,
    source_text: str | None = None,
    decoded_text: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> CandidateCertificate:
    """Replay tokens against the DFA and derive all coverage claims from bytes."""
    ids = [int(token) for token in token_ids]
    catalog = catalog or ObligationCatalog.from_dfa(dfa)
    by_meaning = {(item.kind, item.obligation): item for item in catalog}
    initial = int(dfa.get_initial_state())
    state = initial
    actions: list[TokenAction] = []
    state_events: list[int] = []
    transition_events: list[tuple[int, int, int]] = []
    path_events: list[tuple[int, int]] = []
    failure: FailureDiagnostic | None = None
    eos_id = getattr(dfa, "eos_token_id", None)

    for index, token_id in enumerate(ids):
        from_state = state
        try:
            to_state = int(dfa.get_next_token_state(from_state, token_id))
        except Exception as error:  # parent bindings expose ValueError, but retain all diagnostics
            actions.append(TokenAction(token_id, from_state, None))
            failure = _failure("illegal_token", error, index)
            break
        if eos_id is not None and token_id == int(eos_id):
            actions.append(TokenAction(token_id, from_state, to_state, ()))
            state = to_state
            continue
        try:
            raw_steps = dfa.get_byte_transition_sequence_from_state(from_state, token_id)
        except Exception as error:
            actions.append(TokenAction(token_id, from_state, to_state))
            failure = _failure("byte_trace", error, index)
            break
        steps = [ByteStep(int(byte), from_state if n == 0 else raw_steps[n - 1][1], int(next_state))
                 for n, (byte, next_state) in enumerate(raw_steps[1:], start=1)]
        actions.append(TokenAction(token_id, from_state, to_state, tuple(steps)))
        if steps and steps[-1].to_state != to_state:
            failure = FailureDiagnostic(
                "destination_mismatch",
                f"byte trace ends at {steps[-1].to_state}, token transition ends at {to_state}",
                index,
            )
            break
        for step in steps:
            transition = (step.from_state, step.byte, step.to_state)
            transition_events.append(transition)
            path_events.append((step.from_state, step.to_state))
            state_events.append(step.to_state)
        state = to_state

    final_state = state
    accepted = failure is None and bool(dfa.is_final_state(final_state))
    if failure is None and not accepted:
        failure = FailureDiagnostic("non_final", f"replay ended in non-final state {final_state}")

    if accepted:
        state_events.insert(0, initial)
        meanings: list[tuple[str, Any]] = [("state", initial)]
        meanings.extend(("state", value) for value in state_events[1:])
        meanings.extend(("transition", value) for value in transition_events)
        meanings.extend(("path", value) for value in path_events)
        credited: list[CoveredObligation] = []
        seen: set[str] = set()
        for kind, meaning in meanings:
            item = by_meaning.get((kind, meaning))
            if item is not None and item.obligation_id not in seen:
                credited.append(CoveredObligation.from_obligation(item))
                seen.add(item.obligation_id)
    else:
        credited = []
        state_events = []
        transition_events = []
        path_events = []

    return CandidateCertificate(
        token_ids=ids,
        initial_state=initial,
        final_state=final_state,
        token_actions=actions,
        accepted=accepted,
        aggregate_eligible=accepted and failure is None,
        covered_obligations=credited,
        failure=failure,
        source_text=source_text,
        decoded_text=decoded_text,
        metadata=metadata or {},
        state_events=state_events,
        transition_events=transition_events,
        path_events=path_events,
    )


def aggregate_certificates(certificates: Iterable[CandidateCertificate]) -> dict[str, Any]:
    """Count ordered replay events from accepted certificates only."""
    states: Counter[int] = Counter()
    transitions: Counter[tuple[int, int, int]] = Counter()
    paths: Counter[tuple[int, int]] = Counter()
    for certificate in certificates:
        if not certificate.aggregate_eligible or not certificate.accepted:
            continue
        states.update(certificate.state_events)
        transitions.update(certificate.transition_events)
        paths.update(certificate.path_events)
    return {"state_counts": dict(states), "transition_counts": dict(transitions), "path_counts": dict(paths)}


def coverage_ratios(catalog: ObligationCatalog, aggregates: dict[str, Any], threshold: int = 1) -> dict[str, float]:
    def ratio(kind: str, key_fn: Any, field: str) -> float:
        items = [item for item in catalog if item.kind == kind]
        if not items:
            return 0.0
        counts = aggregates[field]
        return sum(counts.get(key_fn(item), 0) >= threshold for item in items) / len(items)
    return {
        "state": ratio("state", lambda item: item.state, "state_counts"),
        "transition": ratio("transition", lambda item: item.transition, "transition_counts"),
        "path": ratio("path", lambda item: item.path, "path_counts"),
    }


def accepted_certificates(certificates: Iterable[CandidateCertificate]):
    return (certificate for certificate in certificates if certificate.aggregate_eligible and certificate.accepted)


# Name used by the design contract for the payload carried by a candidate.
ReplayPayload = CandidateCertificate
