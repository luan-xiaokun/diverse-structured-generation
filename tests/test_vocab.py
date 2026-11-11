"""Tests for vocabulary decoding utilities."""

import pytest

from diverse_guide.vocab import (
    build_reduced_vocab,
    build_token_id_map,
    byte_symbol,
    gpt2_bytes_to_unicode,
    gpt2_unicode_to_bytes,
)


def test_byte_symbol_encodes_high_bytes():
    assert [byte_symbol(ord(c)) for c in "az0"] == ["a", "z", "0"]
    # 0x7F is still ASCII; bytes >= 0x80 are encoded as \x00XX.
    assert byte_symbol(0x7F) == chr(0x7F)
    assert byte_symbol(0x80) == "\x0080"
    assert byte_symbol(0xFF) == "\x00FF"


def test_gpt2_bytes_to_unicode_is_bijection():
    b2u = gpt2_bytes_to_unicode()
    u2b = gpt2_unicode_to_bytes()
    assert len(b2u) == 256
    for byte_val, uni_char in b2u.items():
        assert u2b[uni_char] == byte_val


def test_build_reduced_vocab_and_token_id_map(mock_tokenizer):
    vocab = build_reduced_vocab(mock_tokenizer)
    assert 10 in vocab["a"] and 20 in vocab["b"]
    assert "" not in vocab
    # '<eos>' is a special token, so its id (99) must not appear anywhere.
    assert 99 not in {tid for tids in vocab.values() for tid in tids}

    token_map = build_token_id_map(mock_tokenizer)
    assert token_map[10] == "a" and token_map[20] == "b"
    assert 99 not in token_map
    assert all(isinstance(v, str) for v in token_map.values())


class _VocabTokenizer:
    """Tokenizer stub to trigger uncommon vocab decoding branches."""

    def __init__(self, vocab: dict):
        self._vocab = vocab
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
        if token in {"<0x80>", "ab", "☃"}:
            return "\ufffd"
        return str(token)

    def __hash__(self):
        return self._id

    def __eq__(self, other):
        return self is other


def test_build_reduced_vocab_bytes_and_replacement_decoding_paths():
    tok = _VocabTokenizer({b"\x80": 1, "<0x80>": 2, "ab": 3, "<eos>": 99})
    vocab = build_reduced_vocab(tok)
    all_ids = {tid for tids in vocab.values() for tid in tids}
    assert {1, 2, 3} <= all_ids

    with pytest.raises(RuntimeError):
        build_reduced_vocab(_VocabTokenizer({"☃": 7, "<eos>": 99}))
