"""Typed DFA coverage obligations and deterministic catalog construction."""

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Literal

from regex_dfa_guide import DiverseGuideDFA

type ObligationKind = Literal["state", "transition", "path"]
type ObligationMeaning = int | tuple[int, int] | tuple[int, int, int]


@dataclass
class CoverageObligation:
    """One state, byte-transition, or ordered state-pair obligation."""

    obligation_type: ObligationKind
    obligation: ObligationMeaning
    state: int | None = None
    transition: tuple[int, int, int] | None = None
    path: tuple[int, int] | None = None

    def __post_init__(self) -> None:
        if self.obligation_type == "state" and isinstance(self.obligation, int):
            self.state = self.obligation
        elif (
            self.obligation_type == "transition"
            and isinstance(self.obligation, tuple)
            and len(self.obligation) == 3
        ):
            self.transition = self.obligation
        elif (
            self.obligation_type == "path"
            and isinstance(self.obligation, tuple)
            and len(self.obligation) == 2
        ):
            self.path = self.obligation
        else:
            raise ValueError(
                f"Invalid {self.obligation_type} obligation: {self.obligation!r}"
            )

    @property
    def kind(self) -> ObligationKind:
        return self.obligation_type

    @property
    def obligation_id(self) -> str:
        if self.obligation_type == "state":
            return f"state:{self.state}"
        if self.obligation_type == "transition":
            source, byte, target = self.transition  # type: ignore[misc]
            return f"transition:{source}:{byte}:{target}"
        source, target = self.path  # type: ignore[misc]
        return f"path:{source}:{target}"

    @property
    def raw_meaning(self) -> int | list[int]:
        """Return the semantic meaning using JSON-safe arrays for tuples."""
        if isinstance(self.obligation, tuple):
            return list(self.obligation)
        return self.obligation

    def is_state_obligation(self) -> bool:
        return self.obligation_type == "state"

    def is_transition_obligation(self) -> bool:
        return self.obligation_type == "transition"

    def is_path_obligation(self) -> bool:
        return self.obligation_type == "path"

    def is_met_by_transition(self, transition: tuple[int, int, int]) -> bool:
        previous_state, _, next_state = transition
        if self.is_transition_obligation():
            return self.transition == transition
        if self.is_path_obligation():
            return self.path == (previous_state, next_state)
        if self.is_state_obligation():
            return self.state == next_state
        raise ValueError(f"Invalid obligation type: {self.obligation_type}")


@dataclass(frozen=True)
class ObligationCatalog:
    """Stable ordered universe of obligations derived from a parent DFA."""

    items: tuple[CoverageObligation, ...]

    @classmethod
    def from_dfa(cls, dfa: DiverseGuideDFA) -> "ObligationCatalog":
        states = sorted(dfa.get_states())
        transitions: set[tuple[int, int, int]] = set()
        paths: set[tuple[int, int]] = set()
        for source, outgoing in dfa.get_transitions().items():
            for byte, target in outgoing.items():
                transitions.add((source, byte, target))
                paths.add((source, target))
        items = [CoverageObligation("state", state) for state in states]
        items.extend(
            CoverageObligation("transition", value) for value in sorted(transitions)
        )
        items.extend(CoverageObligation("path", value) for value in sorted(paths))
        return cls(tuple(items))

    def __iter__(self) -> Iterator[CoverageObligation]:
        return iter(self.items)

    def __len__(self) -> int:
        return len(self.items)

    def by_id(self, obligation_id: str) -> CoverageObligation:
        for item in self.items:
            if item.obligation_id == obligation_id:
                return item
        raise KeyError(obligation_id)

    def as_json(self) -> list[dict[str, object]]:
        return [
            {"id": item.obligation_id, "kind": item.kind, "meaning": item.raw_meaning}
            for item in self
        ]
