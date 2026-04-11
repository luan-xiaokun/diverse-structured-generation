import pytest

from regex_dfa_guide import DiverseGuideDFA


class MockTokenizer:
    """Minimal hashable tokenizer mock: vocab {'a': 10, 'b': 20}, eos=99."""

    eos_token_id = 99

    def __init__(self):
        # Use object identity as hash so each instance is a distinct cache key.
        self._id = id(self)

    @property
    def all_special_tokens(self):
        return ["<eos>"]

    def get_vocab(self):
        return {"a": 10, "b": 20, "<eos>": 99}

    def convert_tokens_to_string(self, tokens: list[str]) -> str:
        simple_map = {"a": "a", "b": "b"}
        return "".join(simple_map.get(t, "") for t in tokens)

    def __hash__(self):
        return self._id

    def __eq__(self, other):
        return self is other


@pytest.fixture
def mock_tokenizer():
    return MockTokenizer()


@pytest.fixture
def vocab_ab():
    """Minimal vocabulary: token 10='a', token 20='b', eos=99."""
    return {10: "a", 20: "b"}


@pytest.fixture
def dfa_multi(vocab_ab):
    """DFA for one or more a/b characters: (?:[ab]+)$"""
    return DiverseGuideDFA("(?:[ab]+)$", 99, vocab_ab)


@pytest.fixture
def dfa_single(vocab_ab):
    """DFA for exactly one a or b: (?:[ab])$"""
    return DiverseGuideDFA("(?:[ab])$", 99, vocab_ab)
