"""Tests for vocabulary decoding utilities."""

import pytest

from diverse_guide.vocab import (
    build_reduced_vocab,
    build_token_id_map,
    gpt2_bytes_to_unicode,
    gpt2_unicode_to_bytes,
)

# ---------------------------------------------------------------------------
# gpt2 byte/unicode roundtrip
# ---------------------------------------------------------------------------


def test_gpt2_bytes_to_unicode_is_bijection():
    b2u = gpt2_bytes_to_unicode()
    u2b = gpt2_unicode_to_bytes()
    assert len(b2u) == 256
    for byte_val, uni_char in b2u.items():
        assert u2b[uni_char] == byte_val


# ---------------------------------------------------------------------------
# build_reduced_vocab
# ---------------------------------------------------------------------------


def test_build_reduced_vocab_basic(mock_tokenizer):
    vocab = build_reduced_vocab(mock_tokenizer)
    assert b"a" in vocab
    assert b"b" in vocab
    assert 10 in vocab[b"a"]
    assert 20 in vocab[b"b"]


def test_build_reduced_vocab_excludes_special_tokens(mock_tokenizer):
    vocab = build_reduced_vocab(mock_tokenizer)
    # '<eos>' is in all_special_tokens, so its id (99) must not appear anywhere
    all_ids = {tid for tids in vocab.values() for tid in tids}
    assert 99 not in all_ids


def test_build_reduced_vocab_no_empty_strings(mock_tokenizer):
    vocab = build_reduced_vocab(mock_tokenizer)
    assert b"" not in vocab


# ---------------------------------------------------------------------------
# build_token_id_map
# ---------------------------------------------------------------------------


def test_build_token_id_map_basic(mock_tokenizer):
    token_map = build_token_id_map(mock_tokenizer)
    assert token_map[10] == b"a"
    assert token_map[20] == b"b"


def test_build_token_id_map_excludes_eos(mock_tokenizer):
    token_map = build_token_id_map(mock_tokenizer)
    assert 99 not in token_map


def test_build_token_id_map_values_are_bytes(mock_tokenizer):
    token_map = build_token_id_map(mock_tokenizer)
    for v in token_map.values():
        assert isinstance(v, bytes)


class _VocabTokenizer:
    """Tokenizer stub to trigger uncommon vocab decoding branches."""

    def __init__(self, vocab: dict, replacement_tokens: set[str] | None = None):
        self._vocab = vocab
        self._replacement_tokens = replacement_tokens or set()
        self.eos_token_id = 99
        self._id = id(self)

    @property
    def all_special_tokens(self):
        return ["<eos>"]

    def get_vocab(self):
        return self._vocab

    def convert_tokens_to_string(self, tokens):
        token = tokens[0]
        if isinstance(token, bytes):
            return token.decode("utf-8", errors="replace")
        if token in self._replacement_tokens:
            return "\ufffd"
        if token == "\ufffd":
            return "\ufffd"
        if token == "snowman":
            return "☃"
        return str(token)

    def __hash__(self):
        return self._id

    def __eq__(self, other):
        return self is other


def test_build_reduced_vocab_recovers_exact_bytes_for_synthetic_token_forms():
    inverse = gpt2_bytes_to_unicode()
    high_1 = inverse[0xE1]
    high_2 = inverse[0x9A]
    high_3 = inverse[0x80]
    vocab = {
        "ascii": 1,
        "snowman": 2,
        high_1: 3,
        high_2 + high_3: 4,
        "A" + high_1: 5,
        "<0x80>": 6,
        "\ufffd": 7,
        b"\xff": 8,
        "<eos>": 99,
    }
    tok = _VocabTokenizer(
        vocab,
        replacement_tokens={high_1, high_2 + high_3, "A" + high_1},
    )

    token_map = build_token_id_map(tok)

    assert token_map == {
        1: b"ascii",
        2: "☃".encode(),
        3: b"\xe1",
        4: b"\x9a\x80",
        5: b"A\xe1",
        6: b"\x80",
        7: b"\xef\xbf\xbd",
        8: b"\xff",
    }
    assert b"" not in build_reduced_vocab(tok)
    assert b"".join(token_map[token_id] for token_id in (3, 4)) == "\u1680".encode()


def test_build_reduced_vocab_runtime_error_for_unmappable_replacement():
    tok = _VocabTokenizer({"☃": 7, "<eos>": 99}, replacement_tokens={"☃"})
    with pytest.raises(RuntimeError):
        build_reduced_vocab(tok)
