"""Vocabulary decoding utilities for HuggingFace tokenizers."""

import re
from functools import lru_cache
from typing import cast

from transformers import PreTrainedTokenizerBase

RE_LLAMA_BYTE_TOKEN = re.compile(r"^<0x[0-9A-F]{2}>$")

# The "▁*" prefix is required to handle Gemma and GPT-SW3 tokenizers.
# The "\.*" suffix is required to handle the NorwAI tokenizer.
# The "\.*" prefix is required to handle the Salamandra tokenizer.
# The "s*$" suffix is required to handle the OpenCoder tokenizer.
RE_REPLACEMENT_SEQ = re.compile(r"^▁*\.*\ufffd+\.*s*$")


def byte_symbol(byte: int) -> str:
    """Encode a single byte as a string compatible with regex-automata's
    byte representation.

    ASCII bytes (< 0x80) are returned as their Unicode character.
    High bytes (≥ 0x80) are encoded as ``\\x00XX`` (null byte followed by
    two hex digits) to avoid conflicting with valid UTF-8 sequences in the
    DFA input alphabet.
    """
    return f"\x00{byte:02X}" if byte >= 0x80 else chr(byte)


# Copied from transformers.models.gpt2.tokenization_gpt2.bytes_to_unicode
@lru_cache
def gpt2_bytes_to_unicode():
    """
    Returns list of utf-8 byte and a mapping to unicode strings.
    We specifically avoids mapping to whitespace/control characters the bpe
    code barfs on.
    """
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))


@lru_cache
def gpt2_unicode_to_bytes() -> dict[str, int]:
    """Inverse mapping of :func:`gpt2_bytes_to_unicode`: unicode char → byte value."""
    return {v: k for k, v in gpt2_bytes_to_unicode().items()}


@lru_cache
def build_reduced_vocab(
    tokenizer: PreTrainedTokenizerBase,
) -> dict[str, set[int]]:
    """decoded_str -> set[token_id], for use with DiverseGuideDFA.

    Logic mirrors reduced_vocabulary() from the reference implementation, using
    HuggingFace tokenizer API (get_vocab, all_special_tokens, convert_tokens_to_string).
    """
    vocabulary: dict[str, set[int]] = {}
    special_tokens = set(tokenizer.all_special_tokens)
    vocab = tokenizer.get_vocab()

    for token, token_idx in vocab.items():
        if token in special_tokens:
            continue

        token_str: str = tokenizer.convert_tokens_to_string([token])

        if token_str:
            if isinstance(token, bytes):
                # Handle BPE tokenizers where the tokens are directly stored as bytes
                # https://github.com/QwenLM/Qwen/blob/main/tokenization_note.md#regular-tokens
                token_str = "".join(byte_symbol(b) for b in token)

            elif "\ufffd" in token_str and not RE_REPLACEMENT_SEQ.match(token):
                # invalid utf-8 sequences are replaced with \ufffd, but there
                # might also be tokens specifically for that character
                if RE_LLAMA_BYTE_TOKEN.match(token):
                    # llama-like tokenizers have <0xXX> tokens for all
                    # bytes >= 0x80 and represent all incomplete utf-8
                    # sequences using such tokens
                    token_bytes = [int(token[3:5], 16)]
                else:
                    # gpt2-like tokenizers have multi-byte tokens that can
                    # have a mix of full and incomplete utf-8 characters
                    token_bytes = cast(
                        list[int], [gpt2_unicode_to_bytes().get(c) for c in token]
                    )
                    if None in token_bytes:
                        raise RuntimeError(
                            f"Cannot convert token `{token}` "
                            f"({token_idx}) to bytes: {token_str}"
                        )
                token_str = "".join(byte_symbol(b) for b in token_bytes)

            assert isinstance(token_str, str)
            vocabulary.setdefault(token_str, set()).add(token_idx)

    return vocabulary


@lru_cache
def build_token_id_map(
    tokenizer: PreTrainedTokenizerBase,
) -> dict[int, str]:
    """token_id -> decoded_str, for use with DiverseGuideDFA."""
    vocab = build_reduced_vocab(tokenizer)
    return {
        tid: token_str
        for token_str, tids in vocab.items()
        for tid in tids
        if tid != tokenizer.eos_token_id
    }
