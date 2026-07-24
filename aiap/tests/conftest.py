import pytest

from regex_dfa_guide import DiverseGuideDFA


@pytest.fixture
def vocab_multi():
    return {10: b"ab", 20: b"a", 30: b"b"}


@pytest.fixture
def dfa_multi(vocab_multi):
    return DiverseGuideDFA("(?:ab)$", 99, vocab_multi)


@pytest.fixture
def dfa_empty():
    return DiverseGuideDFA("(?:a*)$", 99, {10: b"a"})


@pytest.fixture
def dfa_non_completable():
    dfa = DiverseGuideDFA("(?:ab)$", 99, {10: b"a", 20: b"b"})
    blocked_state = dfa.get_next_token_state(dfa.get_initial_state(), 10)

    class NonCompletable:
        def __getattr__(self, name):
            return getattr(dfa, name)

        def get_allowed_token_ids(self, state):
            if state == blocked_state:
                return []
            return dfa.get_allowed_token_ids(state)

    return NonCompletable()
