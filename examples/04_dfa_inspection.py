"""
DFA inspection: examine the DFA built from a regex without a language model.

This example shows how to use the Rust DFA directly (no model needed) to:
  - inspect states and transitions
  - trace byte-level state sequences
  - verify that a string matches the regex through the DFA

This is useful for understanding the DFA structure and debugging regex patterns.
The DFA can also be pickled (via bincode serialization) for caching.

Run:
    uv run python examples/04_dfa_inspection.py
"""

import pickle

from regex_dfa_guide import DiverseGuideDFA

# Minimal mock vocabulary: a small set of tokens for illustration
VOCAB: dict[int, str] = {
    0: "a",
    1: "b",
    2: "c",
    3: "ab",
    4: "bc",
    5: "abc",
    999: "<eos>",  # EOS token
}

# Regex: one or more characters from {a, b, c}
REGEX = r"(?:[abc]+)$"
EOS_TOKEN_ID = 999


def main():
    print(f"Building DFA for regex: {REGEX!r}\n")
    dfa = DiverseGuideDFA(REGEX, EOS_TOKEN_ID, VOCAB)

    states = dfa.get_states()
    final_states = dfa.get_final_states()
    init = dfa.get_initial_state()

    print(f"Total states  : {len(states)}")
    print(f"Initial state : {init}")
    print(f"Final states  : {sorted(final_states)}")

    print("\nAllowed tokens from initial state:")
    allowed = dfa.get_allowed_token_ids(init)
    for tid in sorted(allowed):
        token_str = VOCAB[tid]
        next_state = dfa.get_next_token_state(init, tid)
        print(f"  token_id={tid:3d}  token={token_str!r:6s}  → state {next_state}")

    # Trace a complete sequence "abc"
    text = "abc"
    print(f"\nByte-state trace for {text!r}:")
    transitions = dfa.get_transition_sequence(text)
    for byte_val, state in transitions:
        char = chr(byte_val) if byte_val > 0 else "<start>"
        print(f"  byte={char!r}  → state {state}")

    # Pickling (bincode serialization round-trip)
    print("\nPickle round-trip test:")
    blob = pickle.dumps(dfa)
    dfa2 = pickle.loads(blob)
    assert dfa2.get_states() == dfa.get_states()
    print(f"  Original and restored DFA agree: {dfa == dfa2}")


if __name__ == "__main__":
    main()
