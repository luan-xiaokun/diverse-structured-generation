"""Tests for vocabulary decoding utilities."""

import pytest

from diverse_guide.vocab import (
    build_reduced_vocab,
    build_token_id_map,
    byte_symbol,
    gpt2_bytes_to_unicode,
    gpt2_unicode_to_bytes,
)

# ---------------------------------------------------------------------------
# byte_symbol
# ---------------------------------------------------------------------------


def test_byte_symbol_ascii_printable():
    assert byte_symbol(ord("a")) == "a"
    assert byte_symbol(ord("z")) == "z"
    assert byte_symbol(ord("0")) == "0"


def test_byte_symbol_high_byte():
    # Bytes >= 0x80 are encoded as \x00XX
    result = byte_symbol(0x80)
    assert result == "\x0080"

    result = byte_symbol(0xFF)
    assert result == "\x00FF"


def test_byte_symbol_boundary():
    # 0x7F is still ASCII (< 0x80)
    assert byte_symbol(0x7F) == chr(0x7F)
    # 0x80 crosses into the high-byte encoding
    assert byte_symbol(0x80).startswith("\x00")


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
    assert "a" in vocab
    assert "b" in vocab
    assert 10 in vocab["a"]
    assert 20 in vocab["b"]


def test_build_reduced_vocab_excludes_special_tokens(mock_tokenizer):
    vocab = build_reduced_vocab(mock_tokenizer)
    # '<eos>' is in all_special_tokens, so its id (99) must not appear anywhere
    all_ids = {tid for tids in vocab.values() for tid in tids}
    assert 99 not in all_ids


def test_build_reduced_vocab_no_empty_strings(mock_tokenizer):
    vocab = build_reduced_vocab(mock_tokenizer)
    assert "" not in vocab


# ---------------------------------------------------------------------------
# build_token_id_map
# ---------------------------------------------------------------------------


def test_build_token_id_map_basic(mock_tokenizer):
    token_map = build_token_id_map(mock_tokenizer)
    assert token_map[10] == "a"
    assert token_map[20] == "b"


def test_build_token_id_map_excludes_eos(mock_tokenizer):
    token_map = build_token_id_map(mock_tokenizer)
    assert 99 not in token_map


def test_build_token_id_map_values_are_strings(mock_tokenizer):
    token_map = build_token_id_map(mock_tokenizer)
    for v in token_map.values():
        assert isinstance(v, str)


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
    assert any(tid == 1 for tids in vocab.values() for tid in tids)
    assert any(tid == 2 for tids in vocab.values() for tid in tids)
    assert any(tid == 3 for tids in vocab.values() for tid in tids)


def test_build_reduced_vocab_runtime_error_for_unmappable_replacement():
    tok = _VocabTokenizer({"☃": 7, "<eos>": 99})
    with pytest.raises(RuntimeError):
        build_reduced_vocab(tok)
