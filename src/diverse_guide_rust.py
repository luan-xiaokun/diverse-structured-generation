"""
Diverse structured generation for regular expression constraints.
The generation process is guided by a finite automaton (DFA) and
adjusted based on DFA traversal history.
This module calls the Rust implementation of the reward function
and penalty function to obtain the final adjustment term.
"""

import re
from functools import lru_cache

import torch
from outlines.generate.api import SequenceGeneratorAdapter
from outlines.models.tokenizer import Tokenizer
from outlines.models.transformers import Transformers
from outlines.processors.structured import GuideLogitsProcessor
from outlines.samplers import Sampler, multinomial
from outlines_core.fsm.guide import Generate, Guide, Write

from minimal_dfa import DiverseGuideDFA

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


class DiverseRegexGuide(Guide):
    """Guide to generate text diversely in the language of a regular expression."""

    initial_state = 0

    def __init__(
        self, regex_string: str, tokenizer: Tokenizer, *args, **kwargs
    ) -> None:
        token_id_to_token = {
            token_id: token for token, token_id in tokenizer.vocabulary.items()
        }
        self.dfa = DiverseGuideDFA(
            regex_string, tokenizer.eos_token_id, token_id_to_token
        )
        self.eos_tensor = torch.tensor([tokenizer.eos_token_id])
        self.initial_state = self.dfa.get_initial_state()

    def get_next_instruction(self, state: int) -> Instruction:
        if state == -1:
            return Write(self.eos_tensor)
        next_tokens_mask = self.dfa.get_allowed_token_ids(state)
        if next_tokens_mask is None:
            return Write(self.eos_tensor)
        return Generate(torch.tensor(next_tokens_mask))

    def get_next_state(self, state: int, token_id: int) -> int:
        return self.dfa.get_next_token_state(state, token_id)

    def is_final_state(self, state: int) -> bool:
        return state == -1 or self.dfa.is_final_state(state)

    def copy(self):
        return self


class DiverseRegexLogitsProcessor(GuideLogitsProcessor):
    def __init__(
        self,
        regex_string: str,
        tokenizer: Tokenizer,
        gamma: float = 0.5,
        beta: float = 3.0,
    ):
        guide = DiverseRegexGuide(regex_string, tokenizer)
        super().__init__(tokenizer=tokenizer, guide=guide)
        self.dfa = guide.dfa
        self.gamma = gamma
        self.beta = beta

    def process_logits(
        self, input_ids: torch.LongTensor, logits: torch.FloatTensor
    ) -> torch.Tensor:
        if self._seq_start_idx is None:
            self._seq_start_idx = len(input_ids[0])

        sequence_states: list[int] = []

        for seq_ids in input_ids:
            gen_ids = seq_ids[self._seq_start_idx :]
            curr_state_key = hash(tuple(gen_ids.tolist()))

            if curr_state_key not in self._guide_states:
                token_id = gen_ids[-1].item()
                prev_state = self._guide_states[hash(tuple(gen_ids[:-1].tolist()))]
                curr_state = self.guide.get_next_state(prev_state, token_id)
                self._guide_states[curr_state_key] = curr_state
                state_seq = self.dfa.get_byte_state_sequence(prev_state, token_id)
                assert state_seq is not None and state_seq[-1] == curr_state
                self.dfa.update_local_state_counter(prev_state, token_id)

            sequence_states.append(self._guide_states[curr_state_key])

        device = logits.device
        allowed_tokens_batch = []
        batch_indices = []
        adjusts = []
        for i, guide_state in enumerate(sequence_states):
            allowed_tokens, reward_counts, penalty_counts = self.dfa.compute_counts(
                guide_state
            )
            allowed_tokens = torch.tensor(allowed_tokens, dtype=torch.long)
            allowed_tokens_batch.append(allowed_tokens)
            batch_indices.append(torch.full_like(allowed_tokens, i))
            reward_counts = torch.tensor(reward_counts, dtype=torch.float)
            penalty_counts = torch.tensor(penalty_counts, dtype=torch.float)
            reward = torch.log(1 + reward_counts.sum()) / (1 + reward_counts)
            penalty = self.beta * penalty_counts**2
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

        self.logits_processor.dfa.reset_local_state_counter()

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
        dfa.update_path_counter(generated_content)


def diverse_regex(
    model: Transformers, regex_str: str, sampler: Sampler = multinomial()
):
    logits_processor = DiverseRegexLogitsProcessor(regex_str, model.tokenizer)
    return StatefulSequenceGeneratorAdapter(model, logits_processor, sampler)
