"""
Diverse structured generation for regular expression constraints.
Pure Python reference implementation using HuggingFace transformers API.

This module mirrors the algorithm in :mod:`guide_rust` but implements all
counter tracking in pure Python (byte-level state and transition counters).
It relies on :class:`~regex_dfa_guide.DiverseGuideDFA` only for DFA structural
queries (transitions, allowed tokens), not for counter management.

The diversity adjustment formula is identical to the Rust implementation:

.. math::

    \\Delta_i = \\gamma \\cdot \\text{logits\\_range}
                \\cdot \\frac{\\log(1 + \\sum_j c^{\\text{trans}}_j)}
                             {1 + c^{\\text{trans}}_i}
                \\cdot \\frac{1}{(c^{\\text{local}}_i)^2}

where :math:`c^{\\text{trans}}_i` is the selected path-frequency estimate along
token *i*'s path (global, cross-sequence reward signal) and
:math:`c^{\\text{local}}_i` is the maximum byte-state visit count along the
path within the current generation call (per-call penalty, lower-bounded at 1).
"""

from collections import Counter

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase
from transformers.generation.logits_process import LogitsProcessor, LogitsProcessorList

try:
    from regex_dfa_guide import DiverseGuideDFA
except ImportError:
    from regex_dfa_guide.regex_dfa_guide_rs import DiverseGuideDFA

from .vocab import build_token_id_map


class DiverseRegexGuide:
    """Thin Python wrapper around :class:`DiverseGuideDFA` with a token-level interface.

    Exposes token-level DFA queries used by :class:`DiverseRegexLogitsProcessor`.
    Unlike the Rust-backed version in :mod:`guide_rust`, counter management is
    handled entirely in Python; this class is stateless with respect to counters.
    """

    def __init__(self, regex_string: str, tokenizer: PreTrainedTokenizerBase):
        """Build the DFA guide for *regex_string* and the given tokenizer.

        Parameters
        ----------
        regex_string:
            Regular expression that generated text must match.  The pattern is
            automatically anchored (``(?:...)$``) before DFA construction,
            matching the behaviour of :mod:`guide_rust`.
        tokenizer:
            HuggingFace tokenizer whose vocabulary is compiled into the DFA's
            token-transition table.
        """
        token_id_to_token = build_token_id_map(tokenizer)
        self.dfa = DiverseGuideDFA(
            "(?:" + regex_string + ")$", tokenizer.eos_token_id, token_id_to_token
        )
        self.eos_token_id = tokenizer.eos_token_id
        self.initial_state = self.dfa.get_initial_state()

    def get_next_instruction(self, state: int) -> list[int]:
        """Return allowed token ids for *state*, or ``[eos_token_id]`` if terminal."""
        if state == -1:
            return [self.eos_token_id]
        next_tokens = self.dfa.get_allowed_token_ids(state)
        if next_tokens is None:
            return [self.eos_token_id]
        return next_tokens

    def get_next_state(self, state: int, token_id: int) -> int:
        """Advance the DFA by one token and return the resulting state id."""
        return self.dfa.get_next_token_state(state, token_id)

    def get_distance_to_accepting(self, state: int) -> int:
        """Return the shortest token-level distance to an accepting state."""
        if state == -1:
            return 0
        return self.dfa.get_distance_to_accepting(state)

    def is_final_state(self, state: int) -> bool:
        """Return ``True`` if *state* is an accepting (final) state."""
        return state == -1 or self.dfa.is_final_state(state)

    def get_state_sequence_from_token_id(self, state: int, token_id: int) -> list[int]:
        """Return the byte-level DFA state sequence for consuming *token_id*
        from *state*.

        Returns ``[state, s1, s2, ...]`` where each subsequent element is the DFA
        state reached after consuming one more byte of the token string.
        Returns ``[]`` if the token is not valid from *state*.
        """
        try:
            return list(self.dfa.get_byte_state_sequence(state, token_id))
        except Exception:
            return []


class DiverseRegexLogitsProcessor(LogitsProcessor):
    """Pure Python logits processor for diverse regex-constrained generation.

    Functionally equivalent to the Rust-backed processor in :mod:`guide_rust`.
    Counter tracking uses Python :class:`~collections.Counter` objects operating
    on byte-level DFA states and transitions instead of the Rust path counter.

    See :class:`guide_rust.DiverseRegexLogitsProcessor` for parameter details.
    """

    def __init__(
        self,
        regex_string: str,
        tokenizer: PreTrainedTokenizerBase,
        gamma: float = 0.15,
        completion_bias_eta: float = 0.15,
        path_frequency_estimator: str = "harmonic",
    ):
        self.guide = DiverseRegexGuide(regex_string, tokenizer)
        self.state_counter: Counter = Counter()
        self.transition_counter: Counter = Counter()
        self.local_state_counter: Counter = Counter()
        self.local_transition_counter: Counter = Counter()
        if completion_bias_eta < 0.0:
            raise ValueError("completion_bias_eta must be non-negative")
        if path_frequency_estimator not in {"min", "harmonic"}:
            raise ValueError("path_frequency_estimator must be 'min' or 'harmonic'")
        self.gamma = gamma
        self.completion_bias_eta = completion_bias_eta
        self.path_frequency_estimator = path_frequency_estimator
        self._guide_states: dict[tuple[int, ...], int] = {(): self.guide.initial_state}
        self._seq_start_idx: int | None = None

    def reset(self) -> None:
        """Reset per-call local counters and guide-state cache."""
        self._seq_start_idx = None
        self._guide_states = {(): self.guide.initial_state}
        self.local_state_counter = Counter()
        self.local_transition_counter = Counter()

    @staticmethod
    def _shifted_harmonic_mean(counts: list[int]) -> float:
        if not counts:
            return 0.0
        reciprocal_sum = sum(1.0 / (count + 1.0) for count in counts)
        return (len(counts) / reciprocal_sum) - 1.0

    @classmethod
    def _estimate_path_frequency(cls, counts: list[int], estimator: str) -> float:
        if not counts:
            return 0.0
        if estimator == "min":
            return float(min(counts))
        return cls._shifted_harmonic_mean(counts)

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor
    ) -> torch.FloatTensor:
        """Apply masking and diversity adjustment; called by ``model.generate``
        at each step."""
        if self._seq_start_idx is None:
            self._seq_start_idx = input_ids.shape[1]

        sequence_states: list[int] = []

        for seq_ids in input_ids:
            gen_ids = seq_ids[self._seq_start_idx :]
            curr_state_key = tuple(gen_ids.tolist())

            if curr_state_key not in self._guide_states:
                prev_state = self._guide_states[tuple(gen_ids[:-1].tolist())]
                token_id = gen_ids[-1].item()
                if prev_state == -1:
                    curr_state = -1
                else:
                    try:
                        curr_state = self.guide.get_next_state(prev_state, token_id)
                    except ValueError:
                        # In batched generation, finished rows may carry
                        # padding-like tokens in subsequent steps.
                        curr_state = -1
                self._guide_states[curr_state_key] = curr_state
                if prev_state != -1 and curr_state != -1:
                    state_seq = self.guide.get_state_sequence_from_token_id(
                        prev_state, token_id
                    )
                    for state in state_seq:
                        self.local_state_counter[state] += 1
                    for transition in zip(state_seq, state_seq[1:]):
                        self.local_transition_counter[transition] += 1

            sequence_states.append(self._guide_states[curr_state_key])

        device = scores.device
        allowed_tokens_batch = []
        batch_indices = []
        adjusts = []
        for i, guide_state in enumerate(sequence_states):
            allowed_tokens = self.guide.get_next_instruction(guide_state)
            allowed_tokens = torch.tensor(allowed_tokens, dtype=torch.long)
            allowed_tokens_batch.append(allowed_tokens)
            batch_indices.append(torch.full_like(allowed_tokens, i))

            counts = []
            local_counts = []
            current_distance = self.guide.get_distance_to_accepting(guide_state)
            completion_bonuses = []
            for token_id in allowed_tokens.tolist():
                state_seq = self.guide.get_state_sequence_from_token_id(
                    guide_state, token_id
                )
                next_state = state_seq[-1] if state_seq else guide_state
                transitions = list(zip(state_seq, state_seq[1:]))
                if transitions:
                    counts.append(
                        self._estimate_path_frequency(
                            [self.transition_counter[t] for t in transitions],
                            self.path_frequency_estimator,
                        )
                    )
                else:
                    counts.append(0.0)
                local_count = max(
                    (self.local_state_counter[s] for s in state_seq[1:]), default=0
                )
                local_counts.append(max(1, local_count))
                if self.completion_bias_eta > 0.0:
                    next_distance = self.guide.get_distance_to_accepting(next_state)
                    progress = max(0, current_distance - next_distance)
                    completion_bonuses.append(
                        self.completion_bias_eta * progress / (1 + current_distance)
                    )
                else:
                    completion_bonuses.append(0.0)

            counts_t = torch.tensor(counts, dtype=torch.float)
            reward = torch.log(1 + counts_t.sum()) / (1 + counts_t)
            local_counts_t = torch.tensor(local_counts, dtype=torch.float)
            penalty = local_counts_t**2
            adjusts.append(
                (reward / penalty)
                + torch.tensor(completion_bonuses, dtype=torch.float)
            )

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

    def update_generated_content(self, generated_content: str) -> None:
        """Update cross-call counters only for valid completed strings."""
        try:
            states = list(self.guide.dfa.get_state_sequence(generated_content))
        except ValueError:
            return
        if not states or not self.guide.dfa.is_final_state(states[-1]):
            return
        states.extend(
            self.guide.dfa.get_byte_state_sequence(
                states[-1], self.guide.eos_token_id
            )[1:]
        )
        transitions = list(zip(states, states[1:]))
        for state in states:
            self.state_counter[state] += 1
        for transition in transitions:
            self.transition_counter[transition] += 1


class StatefulSequenceGeneratorAdapter:
    """Wraps a HuggingFace model and logits processor for stateful generation.

    Pure Python counterpart of :class:`guide_rust.StatefulSequenceGeneratorAdapter`.
    Maintains a cross-call byte-level state and transition counter so that successive
    calls are diversified relative to all previously generated sequences.
    """

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
        if "pad_token_id" not in self.generation_defaults:
            pad_token_id = tokenizer.pad_token_id
            if pad_token_id is None:
                pad_token_id = tokenizer.eos_token_id
            if pad_token_id is not None:
                self.generation_defaults["pad_token_id"] = pad_token_id

    def __call__(self, prompt: str, max_tokens: int | None = None, **kwargs) -> str:
        """Generate one sequence that matches the regex."""
        gen_kwargs = {**self.generation_defaults, **kwargs}
        if max_tokens is not None:
            gen_kwargs["max_new_tokens"] = max_tokens
        self.logits_processor.reset()

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        output_ids = self.model.generate(
            **inputs,
            logits_processor=LogitsProcessorList([self.logits_processor]),
            do_sample=True,
            **gen_kwargs,
        )
        new_ids = output_ids[0, inputs["input_ids"].shape[1] :]
        return self.tokenizer.decode(new_ids, skip_special_tokens=True)

    def update_generated_content(self, generated_content: str) -> None:
        """Update global byte-level state and transition counters with
        *generated_content*.

        Must be called after each successful :meth:`__call__` so that subsequent
        calls are diversified relative to *generated_content*.
        """
        self.logits_processor.update_generated_content(generated_content)


class DiverseGuide(StatefulSequenceGeneratorAdapter):
    """Pure-Python reference implementation of the ``DiverseGuide`` API."""

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        regex_str: str,
        gamma: float = 0.15,
        completion_bias_eta: float = 0.15,
        path_frequency_estimator: str = "harmonic",
        **generation_kwargs,
    ):
        logits_processor = DiverseRegexLogitsProcessor(
            regex_str,
            tokenizer,
            gamma=gamma,
            completion_bias_eta=completion_bias_eta,
            path_frequency_estimator=path_frequency_estimator,
        )
        super().__init__(model, tokenizer, logits_processor, **generation_kwargs)
        self.regex_str = regex_str
        self.gamma = gamma
        self.completion_bias_eta = completion_bias_eta
        self.path_frequency_estimator = path_frequency_estimator


def diverse_regex(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    regex_str: str,
    gamma: float = 0.15,
    completion_bias_eta: float = 0.15,
    path_frequency_estimator: str = "harmonic",
    **generation_kwargs,
) -> StatefulSequenceGeneratorAdapter:
    """Create a diverse regex-constrained generator (pure Python reference).

    Compatibility helper for :class:`DiverseGuide` with Python counter tracking.
    """
    return DiverseGuide(
        model,
        tokenizer,
        regex_str,
        gamma=gamma,
        completion_bias_eta=completion_bias_eta,
        path_frequency_estimator=path_frequency_estimator,
        **generation_kwargs,
    )


def baseline_regex(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    regex_str: str,
    **generation_kwargs,
) -> StatefulSequenceGeneratorAdapter:
    """Create a baseline (non-diverse) regex-constrained generator using gamma=0."""
    return diverse_regex(model, tokenizer, regex_str, gamma=0.0, **generation_kwargs)
