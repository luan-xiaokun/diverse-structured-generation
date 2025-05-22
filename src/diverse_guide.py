"""
Diverse structured generation for regular expression constraints.
The generation process is guided by a finite automaton (DFA) and
adjusted based on DFA traversal history.
"""

import re
from collections import Counter
from functools import lru_cache
from typing import cast

import torch
from outlines.generate.api import SequenceGeneratorAdapter
from outlines.models.tokenizer import Tokenizer
from outlines.models.transformers import Transformers
from outlines.processors.structured import GuideLogitsProcessor
from outlines.samplers import Sampler, multinomial
from outlines_core.fsm.guide import Generate, Guide, Write

from minimal_dfa import MinDivDFA

Instruction = Generate | Write


re_llama_byte_token = re.compile(r"^<0x[0-9A-F]{2}>$")

# The "▁*" prefix is required to handle Gemma and GPT-SW3 tokenizers.
# The "\.*" suffix is required to handle the NorwAI tokenizer.
# The "\.*" prefix is required to handle the Salamandra tokenizer.
# The "s*$" suffix is required to handle the OpenCoder tokenizer.
re_replacement_seq = re.compile(r"^▁*\.*�+\.*s*$")


def byte_symbol(byte: int) -> str:
    return f"\x00{byte:02X}" if byte >= 0x80 else chr(byte)


# Copied from transformers.models.gpt2.tokenization_gpt2.bytes_to_unicode
@lru_cache()
def gpt2_bytes_to_unicode():
    """
    Returns list of utf-8 byte and a mapping to unicode strings. We specifically avoids mapping to whitespace/control
    characters the bpe code barfs on.

    The reversible bpe codes work on unicode strings. This means you need a large # of unicode characters in your vocab
    if you want to avoid UNKs. When you're at something like a 10B token dataset you end up needing around 5K for
    decent coverage. This is a significant percentage of your normal, say, 32K bpe vocab. To avoid that, we want lookup
    tables between utf-8 bytes and unicode strings.
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


@lru_cache()
def gpt2_unicode_to_bytes():
    return {v: k for k, v in gpt2_bytes_to_unicode().items()}


@lru_cache
def reduced_vocabulary(
    tokenizer: Tokenizer,
) -> dict[str, set[int]]:
    """Create a map from decoded vocabulary tokens to lists of equivalent token ids."""
    vocabulary: dict[str, set[int]] = {}
    for token, token_idx in tokenizer.vocabulary.items():
        if token in tokenizer.special_tokens:
            continue

        token_str: str | tuple[str, ...] = tokenizer.convert_token_to_string(token)

        if token_str:
            if isinstance(token, bytes):
                # Handle BPE tokenizers where the tokens are directly stored as bytes
                # https://github.com/QwenLM/Qwen/blob/main/tokenization_note.md#regular-tokens
                token_str = "".join(byte_symbol(b) for b in token)

            elif "\ufffd" in token_str and not re_replacement_seq.match(token):
                # invalid utf-8 sequences are replaced with � (\ufffd), but there
                # might also be tokens specifically for �, ��, ���, etc.

                if re_llama_byte_token.match(token):
                    # llama-like tokenizers have <0xXX> tokens for all
                    # bytes >= 0x80 and represent all incomplete utf-8
                    # sequences using such tokens
                    token_bytes = [int(token[3:5], 16)]
                else:
                    # gpt2-like tokenizers have multi-byte tokens that can
                    # have a mix of full and incomplete utf-8 characters,
                    # for example, b` \xf0` can be one token; these tokenizers
                    # map each byte to a valid utf-8 character
                    token_bytes = cast(
                        list[int], [gpt2_unicode_to_bytes().get(c) for c in token]
                    )
                    if None in token_bytes:
                        raise RuntimeError(
                            f"Cannot convert token `{token}` ({token_idx}) to bytes: {token_str}"
                        )
                token_str = "".join(byte_symbol(b) for b in token_bytes)

            assert isinstance(token_str, str)

            vocabulary.setdefault(token_str, set()).add(token_idx)

    return vocabulary


class DiverseRegexGuide(Guide):
    """Guide to generate text diversely in the language of a regular expression."""

    initial_state = 0

    def __init__(
        self, regex_string: str, tokenizer: Tokenizer, *args, **kwargs
    ) -> None:
        tokens_to_token_ids = reduced_vocabulary(tokenizer)
        self.token_id_to_token = {
            token_id: token
            for token, token_ids in tokens_to_token_ids.items()
            for token_id in token_ids
        }
        self.dfa = MinDivDFA(regex_string, tokenizer.eos_token_id, tokens_to_token_ids)
        self.eos_tensor = torch.tensor([tokenizer.eos_token_id])
        self.initial_state = self.dfa.get_initial_state()

    def get_next_instruction(self, state: int) -> Instruction:
        """Return the next instruction for guided generation.

        The initialization of the guide builds an index which maps FSM states to a
        map from authorized tokens to the state in which the guide needs to move
        if said token is generated. Therefore the authorized tokens at the
        current state are the keys of the map returned by the value of the index
        for current state.

        If the current state is not contained in the end this means that we are
        in a final state of the guide. We only authorize EOS tokens in the final
        state.

        Parameters
        ----------
        state
            The current state of the guide.

        Returns
        -------
        A `Generate` instance that contains the model and the allowed token ids.

        """
        if state == -1:
            return Write(self.eos_tensor)
        next_tokens_mask = self.dfa.get_allowed_token_ids(state)
        if next_tokens_mask is None:
            return Write(self.eos_tensor)
        return Generate(torch.tensor(next_tokens_mask))

    def get_next_state(self, state: int, token_id: int) -> int:
        """Update the state of the guide.

        We use the index to determine to which state the guide should transition
        given a token id.

        Parameters
        ----------
        state
            The current state of the guide.
        token_id
            The id of the token that was just generated.

        Returns
        -------
        The new state of the guide.

        """
        return self.dfa.get_next_token_state(state, token_id)

    def is_final_state(self, state: int) -> bool:
        """Determine whether the current state of the guide is a final state."""
        return state == -1 or self.dfa.is_final_state(state)

    def copy(self):
        return self

    def get_state_sequence_from_token_id(self, state: int, token_id: int) -> list[int]:
        if token_id == self.eos_tensor.item():
            next_state = self.dfa.get_next_token_state(state, token_id)
            return [state, next_state]
        state_seq = self.dfa.get_state_sequence_from_string(
            state, self.token_id_to_token[token_id]
        )
        if state_seq is None:
            return []
        return [state] + state_seq


class DiverseRegexLogitsProcessor(GuideLogitsProcessor):
    """Diverse generation based on a regular expression.

    Attributes
    ----------
    tokenizer
        The tokenizer used to convert tokens to ids.
    guide
        The `outlines.fsm.RegexGuide. which is used to bias the logits.
    """

    # gamma default 0.3
    # beta default 1.0

    def __init__(
        self,
        regex_string: str,
        tokenizer: Tokenizer,
        strategy: str = "byte-transition-freq",
        gamma: float = 0.5,
        beta: float = 3.0,
    ):
        """Compile the RegexGuide that drives the regex-guided generation.

        Parameters
        ----------
        regex_string
            A string that represents a regular expression
        tokenizer
            An Outlines tokenizer
        """
        guide = DiverseRegexGuide(regex_string, tokenizer)
        super().__init__(tokenizer=tokenizer, guide=guide)
        self.dfa = guide.dfa
        self.state_counter = Counter()
        self.transition_counter = Counter()
        self.local_state_counter = Counter()
        self.local_transition_counter = Counter()
        self.strategy = strategy
        self.gamma = gamma
        self.beta = beta

    def process_logits(
        self, input_ids: torch.LongTensor, logits: torch.FloatTensor
    ) -> torch.Tensor:
        """Use the Guide to bias the logits before sampling the next token.

        Parameters
        ----------
        input_ids
            The input token ids.
        logits
            The logits.

        Returns
        -------
        torch.Tensor
            The biased logits.
        """
        if self._seq_start_idx is None:
            self._seq_start_idx = len(input_ids[0])

        sequence_states: list[int] = []  # vector of states corresponding to `input_ids`

        for seq_ids in input_ids:
            gen_ids = seq_ids[self._seq_start_idx :]
            curr_state_key = hash(tuple(gen_ids.tolist()))

            if curr_state_key not in self._guide_states:
                prev_state = self._guide_states[hash(tuple(gen_ids[:-1].tolist()))]
                curr_state = self.guide.get_next_state(prev_state, gen_ids[-1].item())
                self._guide_states[curr_state_key] = curr_state
                state_seq = self.guide.get_state_sequence_from_token_id(
                    prev_state, gen_ids[-1].item()
                )
                assert state_seq is not None and state_seq[-1] == curr_state
                for state in state_seq:
                    self.local_state_counter[state] += 1
                for transition in zip(state_seq, state_seq[1:]):
                    self.local_transition_counter[transition] += 1

            sequence_states.append(self._guide_states[curr_state_key])

        device = logits.device
        allowed_tokens_batch = []
        batch_indices = []
        adjusts = []
        for i, guide_state in enumerate(sequence_states):
            allowed_tokens = self.guide.get_next_instruction(guide_state).tokens
            allowed_tokens_batch.append(allowed_tokens)
            batch_indices.append(
                torch.full_like(allowed_tokens, i)
            )  # Store batch index for each allowed token
            counts = [0] * len(allowed_tokens)
            local_counts = [0] * len(allowed_tokens)
            for j, token_id in enumerate(allowed_tokens):
                # the returned state seq should not be empty
                state_seq = self.guide.get_state_sequence_from_token_id(
                    guide_state, token_id.item()
                )
                counts[j] = min(
                    self.transition_counter[t] for t in zip(state_seq, state_seq[1:])
                )
                local_count = max(self.local_state_counter[s] for s in state_seq[1:])
                local_counts[j] = max(1, local_count)
            counts = torch.tensor(counts, dtype=torch.float)
            reward = torch.log(1 + counts.sum()) / (1 + counts)
            local_counts = torch.tensor(local_counts, dtype=torch.float)
            penalty = self.beta * local_counts**2
            adjusts.append(reward / penalty)

        allowed_tokens_concat = torch.cat(allowed_tokens_batch).to(device)
        batch_indices_concat = torch.cat(batch_indices).to(device)
        adjusts_concat = torch.cat(adjusts).to(device)

        mask = torch.ones_like(logits, dtype=torch.bool)
        mask[batch_indices_concat, allowed_tokens_concat] = False
        logits.masked_fill_(mask, float("-inf"))

        # get logits range
        min_logits = logits[batch_indices_concat, allowed_tokens_concat].min()
        max_logits = logits[batch_indices_concat, allowed_tokens_concat].max()
        logits_range = max_logits - min_logits

        # adjust the logits
        adjust_tensor = torch.zeros_like(logits)
        adjust_tensor[batch_indices_concat, allowed_tokens_concat] = (
            self.gamma * logits_range * adjusts_concat
        )
        logits.add_(adjust_tensor)

        return logits


class StatefulSequenceGeneratorAdapter(SequenceGeneratorAdapter):
    def __init__(
        self,
        model: Transformers,
        logits_processor: DiverseRegexLogitsProcessor,
        sampler: Sampler,
    ):
        self.model = model
        self.logits_processor = logits_processor
        self.sampling_params = sampler.sampling_params

    def __call__(
        self,
        prompts: str | list[str],
        max_tokens: int | None = None,
        stop_at: str | list[str] | None = None,
        seed: int | None = None,
        **model_specific_params,
    ):
        """Generate text from a prompt of list of prompts."""

        generation_params = self.prepare_generation_parameters(
            max_tokens, stop_at, seed
        )

        self.logits_processor.local_state_counter = Counter()
        self.logits_processor.local_transition_counter = Counter()

        completions = self.model.generate(
            prompts,
            generation_params,
            self.logits_processor,
            self.sampling_params,
            **model_specific_params,
        )
        self.logits_processor._seq_start_idx = None

        return self._format(completions)

    def update_generated_content(self, generated_content: str):
        dfa = self.logits_processor.dfa
        init_state = dfa.get_initial_state()
        states = dfa.get_state_sequence_from_string(init_state, generated_content)
        states = [init_state] + states
        transitions = list(zip(states, states[1:]))
        for state in states:
            self.logits_processor.state_counter[state] += 1
        for transition in transitions:
            self.logits_processor.transition_counter[transition] += 1


def diverse_regex(
    model: Transformers, regex_str: str, sampler: Sampler = multinomial()
):
    """Generate structured text in the language of a regular expression.

    Parameters
    ----------
    model:
        An instance of `Transformer` that represents a model from the
        `transformers` library.
    regex_str:
        The regular expression that the output must follow.
    sampler:
        The sampling algorithm to use to generate token ids from the logits
        distribution.

    Returns
    -------
    A `SequenceGeneratorAdapter` instance that generates text constrained by the
    regular expression.

    """
    logits_processor = DiverseRegexLogitsProcessor(regex_str, model.tokenizer)
    return StatefulSequenceGeneratorAdapter(model, logits_processor, sampler)
