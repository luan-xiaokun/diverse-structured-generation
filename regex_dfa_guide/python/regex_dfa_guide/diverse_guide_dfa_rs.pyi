class DiverseGuideDFA:
    def __init__(
        self, regex: str, eos_token_id: int, token_id_to_token: dict[int, str]
    ) -> None:
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

    def get_allowed_bytes(self, state: int) -> list[int]:
        """Returns the allowed bytes for a given state."""
        ...

    def get_allowed_token_ids(self, state: int) -> list[int]:
        """Returns the allowed token ids for a given state."""
        ...

    def get_next_byte_state(self, state: int, char: int) -> int:
        """Gets the next byte state from the current state."""
        ...

    def get_next_token_state(self, state: int, token_id: int) -> int:
        """Gets the next token state from the current state."""
        ...

    def get_byte_state_sequence(self, state: int, token_id: int) -> list[int]:
        """Returns the byte state sequence when reading the token id."""
        ...

    def get_byte_transition_sequence(self, string: str) -> list[tuple[int, int]]:
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

    def __reduce__(self) -> tuple:
        """Pickle support."""
        ...

    @classmethod
    def from_binary(cls, data: bytes) -> DiverseGuideDFA:
        """Deserialize from binary data."""
        ...
