"""
Diverse structured generation for regular expression constraints.
Rust-backed implementation using HuggingFace transformers API.
"""

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase
from transformers.generation.logits_process import LogitsProcessor, LogitsProcessorList

from minimal_dfa import DiverseGuideDFA

from .vocab import build_token_id_map, build_reduced_vocab


class DiverseRegexGuide:
    """Guide to generate text diversely in the language of a regular expression."""

    def __init__(self, regex_string: str, tokenizer: PreTrainedTokenizerBase):
        token_id_to_token = build_token_id_map(tokenizer)  # fixes vocab decoding bug
        self.dfa = DiverseGuideDFA(regex_string, tokenizer.eos_token_id, token_id_to_token)
        self.eos_token_id = tokenizer.eos_token_id
        self.initial_state = self.dfa.get_initial_state()

    def get_next_instruction(self, state: int) -> list[int] | None:
        """Return allowed token ids for the given state, or None if final."""
        if state == -1:
            return [self.eos_token_id]
        next_tokens = self.dfa.get_allowed_token_ids(state)
        if next_tokens is None:
            return [self.eos_token_id]
        return next_tokens

    def get_next_state(self, state: int, token_id: int) -> int:
        return self.dfa.get_next_token_state(state, token_id)

    def is_final_state(self, state: int) -> bool:
        return state == -1 or self.dfa.is_final_state(state)


class DiverseRegexLogitsProcessor(LogitsProcessor):
    """Logits processor for diverse regex-constrained generation."""

    def __init__(
        self,
        regex_string: str,
        tokenizer: PreTrainedTokenizerBase,
        gamma: float = 0.5,
        beta: float = 3.0,
    ):
        guide = DiverseRegexGuide(regex_string, tokenizer)
        self.guide = guide
        self.dfa = guide.dfa
        self.gamma = gamma
        self.beta = beta
        self._guide_states: dict[int, int] = {hash(()): guide.initial_state}
        self._seq_start_idx: int | None = None

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor
    ) -> torch.FloatTensor:
        if self._seq_start_idx is None:
            self._seq_start_idx = input_ids.shape[1]

        sequence_states: list[int] = []

        for seq_ids in input_ids:
            gen_ids = seq_ids[self._seq_start_idx:]
            curr_state_key = hash(tuple(gen_ids.tolist()))

            if curr_state_key not in self._guide_states:
                token_id = gen_ids[-1].item()
                prev_state = self._guide_states[hash(tuple(gen_ids[:-1].tolist()))]
                curr_state = self.guide.get_next_state(prev_state, token_id)
                self._guide_states[curr_state_key] = curr_state
                self.dfa.update_local_state_counter(prev_state, token_id)

            sequence_states.append(self._guide_states[curr_state_key])

        device = scores.device
        allowed_tokens_batch = []
        batch_indices = []
        adjusts = []
        for i, guide_state in enumerate(sequence_states):
            allowed_tokens, reward_counts, penalty_counts = self.dfa.compute_counts(guide_state)
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

        mask = torch.ones_like(scores, dtype=torch.bool)
        mask[batch_indices_concat, allowed_tokens_concat] = False
        scores.masked_fill_(mask, float("-inf"))

        min_logits = scores[batch_indices_concat, allowed_tokens_concat].min()
        max_logits = scores[batch_indices_concat, allowed_tokens_concat].max()
        logits_range = max_logits - min_logits

        adjust_tensor = torch.zeros_like(scores)
        adjust_tensor[batch_indices_concat, allowed_tokens_concat] = (
            self.gamma * logits_range * adjusts_concat
        )
        scores.add_(adjust_tensor)

        return scores


class StatefulSequenceGeneratorAdapter:
    """Wraps a HuggingFace model and logits processor for stateful generation."""

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        logits_processor: DiverseRegexLogitsProcessor,
        **generation_defaults,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.logits_processor = logits_processor
        self.generation_defaults = generation_defaults

    def __call__(
        self, prompt: str, max_tokens: int | None = None, **kwargs
    ) -> str:
        gen_kwargs = {**self.generation_defaults, **kwargs}
        if max_tokens is not None:
            gen_kwargs["max_new_tokens"] = max_tokens
        self.logits_processor.dfa.reset_local_state_counter()
        self.logits_processor._seq_start_idx = None

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        output_ids = self.model.generate(
            **inputs,
            logits_processor=LogitsProcessorList([self.logits_processor]),
            do_sample=True,
            **gen_kwargs,
        )
        new_ids = output_ids[0, inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(new_ids, skip_special_tokens=True)

    def update_generated_content(self, generated_content: str):
        self.logits_processor.dfa.update_path_counter(generated_content)


def diverse_regex(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    regex_str: str,
    gamma: float = 0.5,
    beta: float = 3.0,
    **generation_kwargs,
) -> StatefulSequenceGeneratorAdapter:
    """Create a diverse regex-constrained generator.

    Parameters
    ----------
    model
        A HuggingFace PreTrainedModel.
    tokenizer
        A HuggingFace PreTrainedTokenizerBase.
    regex_str
        The regular expression that output must follow.
    gamma
        Diversity reward scale factor.
    beta
        Diversity penalty scale factor.
    **generation_kwargs
        Additional kwargs forwarded to model.generate().
    """
    logits_processor = DiverseRegexLogitsProcessor(regex_str, tokenizer, gamma=gamma, beta=beta)
    return StatefulSequenceGeneratorAdapter(model, tokenizer, logits_processor, **generation_kwargs)


def baseline_regex(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    regex_str: str,
    **generation_kwargs,
) -> StatefulSequenceGeneratorAdapter:
    """Create a baseline (non-diverse) regex-constrained generator using gamma=0."""
    return diverse_regex(model, tokenizer, regex_str, gamma=0.0, beta=1.0, **generation_kwargs)
