from typing import Dict, List, Optional

class MinDivDFA:
    def __init__(
        self, regex: str, eos_token_id: int, tokens_to_token_ids: dict[str, set[int]]
    ):
        """Creates a minimal DFA from a regex."""
        ...

    def get_states(self) -> list[int]:
        """Get all states in the automata."""
        ...

    def get_next_state(self, state: int, char: int) -> Optional[int]:
        """Updates the state."""
        ...

    def get_next_token_state(self, state: int, token_index: int) -> Optional[int]:
        """Updates the state."""
        ...

    def is_initial_state(self, state: int) -> bool:
        """Determines whether the current state is an initial state."""
        ...

    def get_initial_state(self) -> int:
        """Get the initial state."""
        ...

    def is_final_state(self, state: int) -> bool:
        """Determines whether the current state is a final state."""
        ...

    def get_final_states(self) -> List[int]:
        """Get all final states."""
        ...

    def get_transitions(self) -> Dict[int, Dict[int, int]]:
        """Returns the minimal DFA as a Python Dict object."""
        ...

    def get_token_transitions(self) -> Dict[int, Dict[int, int]]:
        """Returns the minimal DFA as a Python Dict object."""
        ...

    def get_allowed_inputs(self, state: int) -> Optional[list[int]]:
        """Returns the allowed characters for a given state."""
        ...

    def get_allowed_token_ids(self, state: int) -> Optional[list[int]]:
        """Returns the allowed token ids for a given state."""
        ...

    def get_state_sequence_from_string(
        self, state: int, token: str
    ) -> Optional[list[int]]:
        """Returns the state sequence when reading the string."""
        ...

    def get_state_sequence(self, inputs: str) -> Optional[list[int]]:
        """Return the state sequence when reading the inputs."""
        ...

    def get_transition_sequence(self, inputs: str) -> Optional[list[tuple[int, int]]]:
        """Return the transition sequence when reading the inputs."""
        ...

    def __repr__(self) -> str:
        """Gets the debug string representation of the index."""
        ...

    def __str__(self) -> str:
        """Gets the string representation of the index."""

    def __eq__(self, other: object) -> bool:
        """Compares whether two indexes are the same."""
        ...

    def __deepcopy__(self, memo: dict) -> MinDivDFA:
        """Makes a deep copy of the Index."""
        ...

class DiverseGuideDFA:
    def __init__(
        self, regex: str, eos_token_id: int, token_id_to_token: dict[int, str]
    ):
        """Creates a DFA from a regex to guide diverse generation."""
        ...

    def is_initial_state(self, state: int) -> bool:
        """Determines whether the current state is an initial state."""
        ...

    def is_final_state(self, state: int) -> bool:
        """Determines whether the current state is a final state."""
        ...

    def get_initial_state(self) -> int:
        """Returns the initial state of the DFA."""
        ...

    def get_final_states(self) -> set[int]:
        """Returns the set of final states of the DFA."""
        ...

    def get_allowed_bytes(self, state: int) -> Optional[list[int]]:
        """Returns the allowed characters for a given state."""
        ...

    def get_allowed_token_ids(self, state: int) -> Optional[list[int]]:
        """Returns the allowed token ids for a given state."""
        ...

    def get_next_byte_state(self, state: int, char: int) -> Optional[int]:
        """Gets the next byte state from the current state."""
        ...

    def get_next_token_state(self, state: int, token_id: int) -> Optional[int]:
        """Gets the next token state from the current state."""
        ...

    def get_byte_state_sequence(self, state: int, token_id: int) -> Optional[list[int]]:
        """Returns the byte state sequence when reading the token id."""
        ...

    def get_byte_transition_sequence(
        self, string: str
    ) -> Optional[list[tuple[int, int]]]:
        """Returns the byte transition sequence when reading the string."""
        ...

    def update_path_counter(self, string: str) -> None:
        """Updates the path counter for the given string."""
        ...

    def update_local_state_counter(self, state: int, token_id: int) -> None:
        """Updates the local state counter after generating a new token."""
        ...

    def reset_path_counter(self) -> None:
        """Resets the path counter."""
        ...

    def reset_local_state_counter(self) -> None:
        """Resets the local state counter."""
        ...

    def compute_counts(self, state: int) -> tuple[list[int], list[int], list[int]]:
        """Computes the counts for the given state."""
        ...
