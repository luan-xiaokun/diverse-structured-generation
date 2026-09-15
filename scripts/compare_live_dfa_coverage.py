"""Compare raw DFA coverage with productive-live-subgraph coverage.

The existing metrics use all states/transitions exposed by ``DiverseGuideDFA``
as denominators. This script recomputes the denominators after keeping only the
states reachable from the initial state that can also reach an accepting state.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import deque
from pathlib import Path
from typing import Any

from regex_dfa_guide import DiverseGuideDFA


DEFAULT_GRAMMARS = [
    "email",
    "no-bomb",
    "css-color",
    "semver",
    "api-route",
    "json",
]


def split_csv(value: str | None, default: list[str]) -> list[str]:
    if value is None or value == "":
        return list(default)
    return [item.strip() for item in value.split(",") if item.strip()]


def model_slug(model: str) -> str:
    return model.rsplit("/", maxsplit=1)[-1].lower()


def load_generation(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def all_transitions(
    transitions: dict[int, dict[int, int]],
) -> set[tuple[int, int, int]]:
    return {
        (int(source), int(byte), int(target))
        for source, outgoing in transitions.items()
        for byte, target in outgoing.items()
    }


def all_paths(transitions: set[tuple[int, int, int]]) -> set[tuple[int, int]]:
    return {(source, target) for source, _, target in transitions}


def reachable_from_initial(
    initial_state: int,
    transitions: dict[int, dict[int, int]],
) -> set[int]:
    reachable = {initial_state}
    queue = deque([initial_state])
    while queue:
        state = queue.popleft()
        for next_state in transitions.get(state, {}).values():
            next_state = int(next_state)
            if next_state not in reachable:
                reachable.add(next_state)
                queue.append(next_state)
    return reachable


def can_reach_final(
    final_states: set[int],
    transitions: dict[int, dict[int, int]],
) -> set[int]:
    reverse: dict[int, set[int]] = {}
    for source, outgoing in transitions.items():
        for target in outgoing.values():
            reverse.setdefault(int(target), set()).add(int(source))

    productive = set(final_states)
    queue = deque(final_states)
    while queue:
        state = queue.popleft()
        for prev_state in reverse.get(state, set()):
            if prev_state not in productive:
                productive.add(prev_state)
                queue.append(prev_state)
    return productive


def productive_denominators(
    dfa: DiverseGuideDFA,
) -> tuple[set[int], set[tuple[int, int, int]], set[tuple[int, int]]]:
    transitions = {
        int(source): {int(byte): int(target) for byte, target in outgoing.items()}
        for source, outgoing in dfa.get_transitions().items()
    }
    reachable = reachable_from_initial(int(dfa.get_initial_state()), transitions)
    co_reachable = can_reach_final(set(map(int, dfa.get_final_states())), transitions)
    live_states = reachable & co_reachable
    live_transitions = {
        (source, byte, target)
        for source, outgoing in transitions.items()
        for byte, target in outgoing.items()
        if source in live_states and target in live_states
    }
    return live_states, live_transitions, all_paths(live_transitions)


def visited_sets(
    dfa: DiverseGuideDFA,
    samples: list[str],
) -> tuple[set[int], set[tuple[int, int, int]], set[tuple[int, int]]]:
    visited_states: set[int] = set()
    visited_transitions: set[tuple[int, int, int]] = set()
    visited_paths: set[tuple[int, int]] = set()

    for sample in samples:
        for state in dfa.get_state_sequence(sample):
            visited_states.add(int(state))

        transition_sequence = dfa.get_transition_sequence(sample)
        _, prev_state = transition_sequence[0]
        prev_state = int(prev_state)
        for byte, state in transition_sequence[1:]:
            state = int(state)
            byte = int(byte)
            visited_transitions.add((prev_state, byte, state))
            visited_paths.add((prev_state, state))
            prev_state = state

    return visited_states, visited_transitions, visited_paths


def ratio(visited: set[Any], denominator: set[Any]) -> float:
    if not denominator:
        return 0.0
    return len(visited & denominator) / len(denominator)


def build_record(
    grammar: str,
    data_path: Path,
    max_samples: int | None,
) -> dict[str, Any]:
    data = load_generation(data_path)
    samples = data.get("samples", [])
    if not isinstance(samples, list):
        raise ValueError(f"{data_path} field 'samples' must be a list")
    if max_samples is not None:
        samples = samples[:max_samples]

    dfa = DiverseGuideDFA(f"(?:{data['regex']})$", 2**32 - 1, {})
    raw_states = set(map(int, dfa.get_states()))
    raw_transitions = all_transitions(dfa.get_transitions())
    raw_paths = all_paths(raw_transitions)
    live_states, live_transitions, live_paths = productive_denominators(dfa)
    visited_states, visited_transitions, visited_paths = visited_sets(dfa, samples)

    return {
        "grammar": grammar,
        "sample_count": len(samples),
        "raw_state_denominator": len(raw_states),
        "live_state_denominator": len(live_states),
        "raw_transition_denominator": len(raw_transitions),
        "live_transition_denominator": len(live_transitions),
        "raw_path_denominator": len(raw_paths),
        "live_path_denominator": len(live_paths),
        "raw_state_coverage": ratio(visited_states, raw_states),
        "live_state_coverage": ratio(visited_states, live_states),
        "raw_transition_coverage": ratio(visited_transitions, raw_transitions),
        "live_transition_coverage": ratio(visited_transitions, live_transitions),
        "raw_path_coverage": ratio(visited_paths, raw_paths),
        "live_path_coverage": ratio(visited_paths, live_paths),
        "removed_states": len(raw_states - live_states),
        "removed_transitions": len(raw_transitions - live_transitions),
        "removed_paths": len(raw_paths - live_paths),
        "path": str(data_path),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare raw and productive-live DFA coverage denominators."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data/extension/eta_gamma0p3_scan_n200"),
    )
    parser.add_argument("--method", default="harmonic_g0p3_eta1")
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--grammars", default=None, help="Comma-separated list.")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "results/extension/live_coverage_comparison/"
            "qwen2p5_1p5b_harmonic_g0p3_eta1.csv"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    grammars = split_csv(args.grammars, DEFAULT_GRAMMARS)
    records = []
    for grammar in grammars:
        data_path = (
            args.data_root
            / args.method
            / model_slug(args.model)
            / f"{grammar}.json"
        )
        records.append(
            build_record(
                grammar=grammar,
                data_path=data_path,
                max_samples=args.max_samples,
            )
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(records[0].keys()) if records else []
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    print(f"[live-coverage] output={args.output}")
    print(
        "grammar\traw_state\tlive_state\traw_trans\tlive_trans\t"
        "raw_path\tlive_path\tremoved_s/t/p"
    )
    for record in records:
        print(
            f"{record['grammar']}\t"
            f"{record['raw_state_coverage']:.4f}\t"
            f"{record['live_state_coverage']:.4f}\t"
            f"{record['raw_transition_coverage']:.4f}\t"
            f"{record['live_transition_coverage']:.4f}\t"
            f"{record['raw_path_coverage']:.4f}\t"
            f"{record['live_path_coverage']:.4f}\t"
            f"{record['removed_states']}/"
            f"{record['removed_transitions']}/"
            f"{record['removed_paths']}"
        )


if __name__ == "__main__":
    main()
