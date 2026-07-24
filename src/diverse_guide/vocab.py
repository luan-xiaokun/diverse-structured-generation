"""Vocabulary decoding utilities for HuggingFace tokenizers."""

import re
from functools import lru_cache

from transformers import PreTrainedTokenizerBase

RE_LLAMA_BYTE_TOKEN = re.compile(r"^<0x[0-9A-F]{2}>$")

# The "▁*" prefix is required to handle Gemma and GPT-SW3 tokenizers.
# The "\.*" suffix is required to handle the NorwAI tokenizer.
# The "\.*" prefix is required to handle the Salamandra tokenizer.
# The "s*$" suffix is required to handle the OpenCoder tokenizer.
RE_REPLACEMENT_SEQ = re.compile(r"^▁*\.*\ufffd+\.*s*$")


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
) -> dict[bytes, set[int]]:
    """Exact token bytes -> set[token_id], for use with DiverseGuideDFA.

    Logic mirrors reduced_vocabulary() from the reference implementation, using
    HuggingFace tokenizer API (get_vocab, all_special_tokens, convert_tokens_to_string).
    """
    vocabulary: dict[bytes, set[int]] = {}
    special_tokens = set(tokenizer.all_special_tokens)
    vocab = tokenizer.get_vocab()

    for token, token_idx in vocab.items():
        if token in special_tokens:
            continue

        if isinstance(token, bytes):
            token_bytes = bytes(token)
        elif RE_LLAMA_BYTE_TOKEN.fullmatch(token):
            token_bytes = bytes([int(token[3:5], 16)])
        else:
            token_str = tokenizer.convert_tokens_to_string([token])
            if not token_str:
                continue
            if "\ufffd" in token_str and not RE_REPLACEMENT_SEQ.fullmatch(token):
                recovered = [gpt2_unicode_to_bytes().get(char) for char in token]
                if any(byte is None for byte in recovered):
                    raise RuntimeError(
                        f"Cannot convert token `{token}` "
                        f"({token_idx}) to bytes: {token_str}"
                    )
                token_bytes = bytes(recovered)
            else:
                token_bytes = token_str.encode("utf-8")

        if token_bytes:
            vocabulary.setdefault(token_bytes, set()).add(token_idx)

    return vocabulary


@lru_cache
def build_token_id_map(
    tokenizer: PreTrainedTokenizerBase,
) -> dict[int, bytes]:
    """Token ID -> exact raw bytes, for use with DiverseGuideDFA."""
    vocab = build_reduced_vocab(tokenizer)
    return {
        tid: token_bytes
        for token_bytes, tids in vocab.items()
        for tid in tids
        if tid != tokenizer.eos_token_id
    }
