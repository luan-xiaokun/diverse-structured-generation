"""Fixed SemVer grammar and DFA helpers for the SETSS experiment.

The raw expression is the one used by the repository's earlier SemVer runs.
It first appeared in commit 16ed9a84640d8e7a711951ed626d6e4265d048e5 in
``scripts/generate_re.py``.  The original regex/DFA evaluator guarantees a
full match by wrapping grammar regexes in a non-capturing group and appending
``$``; this module applies that same normalization.
"""

from __future__ import annotations

from regex_dfa_guide import DiverseGuideDFA

SEMVER_PROMPT = "Generate a semantic versioning string."

SEMVER_REGEX_SOURCE_COMMIT = "16ed9a84640d8e7a711951ed626d6e4265d048e5"
SEMVER_REGEX_SOURCE_FILE = "scripts/generate_re.py"
SEMVER_REGEX_SOURCE = (
    r"(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)"
    r"(?:-((?:0|[1-9]\d*|\d*[a-zA-Z-][0-9a-zA-Z-]*)"
    r"(?:\.(?:0|[1-9]\d*|\d*[a-zA-Z-][0-9a-zA-Z-]*))*))?"
    r"(?:\+([0-9a-zA-Z-]+(?:\.[0-9a-zA-Z-]+)*))?"
)

# Outlines 0.2.1/interegular interprets ``\d`` as ASCII digits, while the Rust
# regex-automata compiler interprets it as Unicode decimal digits.  The strict
# ASCII spelling below is therefore the compiler normalization needed to keep
# both methods on the same language intended by the historical source regex.
SEMVER_STRICT_ASCII_ALTERNATIVE = (
    r"(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)"
    r"(?:-(?:0|[1-9][0-9]*|[0-9]*[A-Za-z-][0-9A-Za-z-]*)"
    r"(?:\.(?:0|[1-9][0-9]*|[0-9]*[A-Za-z-][0-9A-Za-z-]*))*)?"
    r"(?:\+[0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*)?"
)

SEMVER_OUTLINES_REGEX = SEMVER_REGEX_SOURCE
SEMVER_DIVERSE_REGEX = SEMVER_STRICT_ASCII_ALTERNATIVE
SEMVER_COMPILER_REGEX = f"(?:{SEMVER_DIVERSE_REGEX})$"
SEMVER_NORMALIZATIONS = [
    "The source regex has no anchors.",
    (
        "Outlines 0.2.1 receives the raw source; interegular gives regex FSMs "
        "full-match semantics and rejects `$`."
    ),
    (
        "For regex-automata, replace `\\d` with `[0-9]` and remove capturing "
        "groups using the ticket's strict-ASCII form."
    ),
    (
        "Wrap the normalized Rust regex in a non-capturing group and append `$`, "
        "matching the existing evaluator."
    ),
    (
        "The Rust DFA compiler requests an anchored start state, so a leading "
        "`^` is unnecessary."
    ),
]

EOS_SENTINEL = 2**32 - 1

ACCEPTED_EXAMPLES = [
    "0.0.0",
    "1.2.3",
    "1.0.0-alpha",
    "1.0.0-alpha.1",
    "1.0.0-0.3.7",
    "1.0.0-x.7.z.92",
    "1.0.0+20130313144700",
    "1.0.0+001",
    "1.0.0-beta+exp.sha.5114f85",
    "2.7.1-rc.1+build.123",
    "1.2.3-01a",
]

REJECTED_EXAMPLES = [
    "",
    "v1.2.3",
    "1.2",
    "1.2.3.4",
    "01.2.3",
    "1.02.3",
    "1.2.03",
    "1.2.3-01",
    "1.2.3-",
    "1.2.3+",
    "1.2.3_alpha",
    "1.2.3-..",
    "1.2.3+abc..def",
]


def build_semver_dfa() -> DiverseGuideDFA:
    """Compile the exact DFA used for validation and metric evaluation."""

    return DiverseGuideDFA(SEMVER_COMPILER_REGEX, EOS_SENTINEL, {})


def dfa_fullmatch(dfa: DiverseGuideDFA, value: str) -> bool:
    """Return whether *value* reaches an accepting state in the compiled DFA."""

    try:
        states = dfa.get_state_sequence(value)
    except ValueError:
        return False
    return bool(states) and dfa.is_final_state(states[-1])


def dfa_statistics(dfa: DiverseGuideDFA) -> dict[str, int | bool]:
    """Report DFA size using the repository evaluator's byte-transition model."""

    states = set(dfa.get_states())
    transitions = dfa.get_transitions()
    accepting = set(dfa.get_final_states())

    predecessors: dict[int, set[int]] = {state: set() for state in states}
    for source, edges in transitions.items():
        for target in edges.values():
            predecessors.setdefault(target, set()).add(source)

    live = set(accepting)
    frontier = list(accepting)
    while frontier:
        target = frontier.pop()
        for source in predecessors.get(target, ()):
            if source not in live:
                live.add(source)
                frontier.append(source)

    return {
        "states": len(states),
        "transitions": sum(len(edges) for edges in transitions.values()),
        "accepting_states": len(accepting),
        "live_states": len(live),
        # regex_dfa_guide drops regex-automata dead/quit states while building
        # the exposed state and transition maps.
        "dead_state_included": False,
    }
