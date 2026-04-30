"""
Diverse structured generation for regular expression constraints.
Rust-backed implementation using HuggingFace transformers API.
"""

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase
from transformers.generation.logits_process import LogitsProcessor, LogitsProcessorList

try:
    from regex_dfa_guide import DiverseGuideDFA
except ImportError:
    from regex_dfa_guide.regex_dfa_guide_rs import DiverseGuideDFA

from .vocab import build_token_id_map


class DiverseRegexGuide:
    """Guide to generate text diversely in the language of a regular expression.

    Wraps a :class:`DiverseGuideDFA` (Rust) and exposes a token-level interface
    used by :class:`DiverseRegexLogitsProcessor`.
    """

    def __init__(self, regex_string: str, tokenizer: PreTrainedTokenizerBase):
        """Build the DFA guide for *regex_string* and the given tokenizer.

        Parameters
        ----------
        regex_string:
            Regular expression that generated text must match.  The pattern is
            automatically anchored (``(?:...)$``) before DFA construction.
        tokenizer:
            HuggingFace tokenizer whose vocabulary is compiled into the DFA's
            token-transition table.
        """
        token_id_to_token = build_token_id_map(tokenizer)  # fixes vocab decoding bug
        self.dfa = DiverseGuideDFA(
            "(?:" + regex_string + ")$", tokenizer.eos_token_id, token_id_to_token
        )
        self.eos_token_id = tokenizer.eos_token_id
        self.initial_state = self.dfa.get_initial_state()

    def get_next_instruction(self, state: int) -> list[int] | None:
        """Return allowed token ids for *state*, or ``[eos_token_id]`` if terminal.

        Parameters
        ----------
        state:
            Current DFA state id, or ``-1`` to signal that generation has ended.
        """
        if state == -1:
            return [self.eos_token_id]
        next_tokens = self.dfa.get_allowed_token_ids(state)
        if next_tokens is None:
            return [self.eos_token_id]
        return next_tokens

    def get_next_state(self, state: int, token_id: int) -> int:
        """Advance the DFA by one token and return the resulting state id."""
        return self.dfa.get_next_token_state(state, token_id)

    def is_final_state(self, state: int) -> bool:
        """Return ``True`` if *state* is an accepting (final) state."""
        return state == -1 or self.dfa.is_final_state(state)


class RegexMaskLogitsProcessor(LogitsProcessor):
    """HuggingFace ``LogitsProcessor`` that enforces regex constraints only."""

    def __init__(self, regex_string: str, tokenizer: PreTrainedTokenizerBase):
        self.guide = DiverseRegexGuide(regex_string, tokenizer)
        self.dfa = self.guide.dfa
        self._seq_start_idx: int | None = None
        self._guide_states_by_row: list[int] | None = None
        self._processed_lengths_by_row: list[int] | None = None

    def reset(self) -> None:
        """Reset per-call state before a new ``model.generate()`` call."""
        self._seq_start_idx = None
        self._guide_states_by_row = None
        self._processed_lengths_by_row = None

    def _ensure_rows(self, batch_size: int) -> None:
        if (
            self._guide_states_by_row is None
            or len(self._guide_states_by_row) != batch_size
        ):
            self._guide_states_by_row = [self.guide.initial_state] * batch_size
            self._processed_lengths_by_row = [0] * batch_size

    def _advance_row(self, row: int, gen_ids: torch.Tensor) -> int:
        assert self._guide_states_by_row is not None
        assert self._processed_lengths_by_row is not None

        state = self._guide_states_by_row[row]
        processed = self._processed_lengths_by_row[row]
        while processed < gen_ids.shape[0]:
            token_id = gen_ids[processed].item()
            if state != -1:
                try:
                    state = self.guide.get_next_state(state, token_id)
                except ValueError:
                    state = -1
            processed += 1

        self._guide_states_by_row[row] = state
        self._processed_lengths_by_row[row] = processed
        return state

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor
    ) -> torch.FloatTensor:
        if self._seq_start_idx is None:
            self._seq_start_idx = input_ids.shape[1]
        self._ensure_rows(input_ids.shape[0])

        allowed_tokens_batch = []
        batch_indices = []
        for row, seq_ids in enumerate(input_ids):
            gen_ids = seq_ids[self._seq_start_idx :]
            guide_state = self._advance_row(row, gen_ids)
            allowed_tokens = self.guide.get_next_instruction(guide_state)
            allowed_tokens = torch.tensor(
                allowed_tokens, dtype=torch.long, device=scores.device
            )
            allowed_tokens_batch.append(allowed_tokens)
            batch_indices.append(torch.full_like(allowed_tokens, row))

        allowed_tokens_concat = torch.cat(allowed_tokens_batch)
        batch_indices_concat = torch.cat(batch_indices)
        allowed_scores = scores[batch_indices_concat, allowed_tokens_concat]
        scores.fill_(float("-inf"))
        scores[batch_indices_concat, allowed_tokens_concat] = allowed_scores
        return scores


class DiverseRegexLogitsProcessor(LogitsProcessor):
    """HuggingFace ``LogitsProcessor`` that enforces a regex and promotes diversity.

    At each generation step this processor:

    1. **Masks** all token ids that would violate the regex (sets their logit to
       ``-inf``).
    2. **Adjusts** the remaining logits to reward tokens that traverse
       under-explored DFA paths and penalise tokens already used heavily in the
       current batch:

       .. math::

           \\Delta_i = \\gamma \\cdot \\text{logits\\_range}
                       \\cdot \\frac{\\log(1 + \\sum_j c^{\\text{path}}_j)}
                                    {1 + c^{\\text{path}}_i}
                       \\cdot \\frac{1}{\\beta \\cdot (c^{\\text{local}}_i)^2}

    The path counter (global, cross-sequence) is updated externally via
    :meth:`StatefulSequenceGeneratorAdapter.update_generated_content`.
    The local counter (per ``model.generate`` call) is reset by :meth:`reset`.
    """

    def __init__(
        self,
        regex_string: str,
        tokenizer: PreTrainedTokenizerBase,
        gamma: float = 0.5,
        beta: float = 3.0,
        no_reward: bool = False,
        no_penalty: bool = False,
        no_range_scaling: bool = False,
    ):
        """
        Parameters
        ----------
        regex_string:
            Regular expression that every generated string must match.
        tokenizer:
            HuggingFace tokenizer whose vocabulary is compiled into the DFA.
        gamma:
            Reward scale factor.  Higher values more aggressively boost tokens
            on under-explored paths.  Set to ``0`` for baseline (no diversity).
        beta:
            Penalty scale factor.  Higher values more strongly suppress tokens
            already used heavily within the current batch step.
        no_reward:
            If ``True``, do not apply reward adjustments. For ablation purposes.
        no_penalty:
            If ``True``, do not apply penalty adjustments. For ablation purposes.
        no_range_scaling:
            If ``True``, do not scale logits by the range of available tokens. For
            ablation purposes.
        """
        guide = DiverseRegexGuide(regex_string, tokenizer)
        self.guide = guide
        self.dfa = guide.dfa
        self.gamma = gamma
        self.beta = beta
        self.no_reward = no_reward
        self.no_penalty = no_penalty
        self.no_range_scaling = no_range_scaling

        self._seq_start_idx: int | None = None
        self._guide_states_by_row: list[int] | None = None
        self._processed_lengths_by_row: list[int] | None = None

    def reset(self) -> None:
        """Reset per-call state. Must be called before each new ``model.generate()``
        call."""
        self._seq_start_idx = None
        self._guide_states_by_row = None
        self._processed_lengths_by_row = None

    def _ensure_rows(self, batch_size: int) -> None:
        if (
            self._guide_states_by_row is None
            or len(self._guide_states_by_row) != batch_size
        ):
            self._guide_states_by_row = [self.guide.initial_state] * batch_size
            self._processed_lengths_by_row = [0] * batch_size

    def _advance_row(self, row: int, gen_ids: torch.Tensor) -> int:
        assert self._guide_states_by_row is not None
        assert self._processed_lengths_by_row is not None

        state = self._guide_states_by_row[row]
        processed = self._processed_lengths_by_row[row]
        while processed < gen_ids.shape[0]:
            token_id = gen_ids[processed].item()
            prev_state = state
            if prev_state == -1:
                state = -1
            else:
                try:
                    state = self.guide.get_next_state(prev_state, token_id)
                except ValueError:
                    state = -1
                if state != -1:
                    self.dfa.update_local_state_counter(prev_state, token_id)
            processed += 1

        self._guide_states_by_row[row] = state
        self._processed_lengths_by_row[row] = processed
        return state

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor
    ) -> torch.FloatTensor:
        """Apply masking and diversity adjustment; called by ``model.generate`` at each
        step."""
        if self._seq_start_idx is None:
            self._seq_start_idx = input_ids.shape[1]
        self._ensure_rows(input_ids.shape[0])

        sequence_states = []
        for row, seq_ids in enumerate(input_ids):
            gen_ids = seq_ids[self._seq_start_idx :]
            sequence_states.append(self._advance_row(row, gen_ids))

        device = scores.device
        allowed_tokens_batch = []
        batch_indices = []
        adjusts = []
        for i, guide_state in enumerate(sequence_states):
            if guide_state == -1:
                allowed_tokens = [self.guide.eos_token_id]
                reward_counts = [0]
                penalty_counts = [1]
            else:
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

            if self.no_reward:
                reward = torch.ones_like(reward_counts)
            if self.no_penalty:
                penalty = torch.ones_like(penalty_counts)

            adjusts.append(reward / penalty)

        allowed_tokens_concat = torch.cat(allowed_tokens_batch).to(device)
        batch_indices_concat = torch.cat(batch_indices).to(device)
        adjusts_concat = torch.cat(adjusts).to(device)
        allowed_scores = scores[batch_indices_concat, allowed_tokens_concat]

        min_logits = allowed_scores.min()
        max_logits = allowed_scores.max()
        logits_range = max_logits - min_logits

        if self.no_range_scaling:
            final_adjustment = self.gamma * adjusts_concat
        else:
            final_adjustment = self.gamma * logits_range * adjusts_concat

        scores.fill_(float("-inf"))
        scores[batch_indices_concat, allowed_tokens_concat] = (
            allowed_scores + final_adjustment
        )

        return scores


class StatefulSequenceGeneratorAdapter:
    """Wraps a HuggingFace model and :class:`DiverseRegexLogitsProcessor` for
    stateful generation.

    Maintains a cross-call path counter so that successive calls (or batches)
    are diversified relative to all previously generated sequences.

    Typical usage::

        generator = diverse_regex(model, tokenizer, regex_str)
        for _ in range(100):
            text = generator(prompt, max_tokens=40)
            generator.update_generated_content(text)  # update global path counter

    Or more conveniently via :meth:`generate_batch`, which updates counters
    automatically.
    """

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        logits_processor: RegexMaskLogitsProcessor | DiverseRegexLogitsProcessor,
        **generation_defaults,
    ):
        """
        Parameters
        ----------
        model:
            A HuggingFace ``PreTrainedModel`` (must support ``model.generate``).
        tokenizer:
            Matching ``PreTrainedTokenizerBase``.
        logits_processor:
            A configured :class:`DiverseRegexLogitsProcessor`.
        **generation_defaults:
            Default kwargs forwarded to ``model.generate`` on every call
            (e.g. ``temperature=0.8``, ``top_p=0.95``).
        """
        self.model = model
        self.tokenizer = tokenizer
        self.logits_processor = logits_processor
        self.generation_defaults = generation_defaults
        if "pad_token_id" not in self.generation_defaults:
            pad_token_id = getattr(tokenizer, "pad_token_id", None)
            if pad_token_id is None:
                pad_token_id = tokenizer.eos_token_id
            if pad_token_id is not None:
                self.generation_defaults["pad_token_id"] = pad_token_id

    def __call__(self, prompt: str, max_tokens: int | None = None, **kwargs) -> str:
        """Generate one sequence that matches the regex.

        Parameters
        ----------
        prompt:
            Input prompt string.
        max_tokens:
            Maximum number of new tokens to generate.
        **kwargs:
            Override any ``generation_defaults`` for this call only.
        """
        gen_kwargs = {**self.generation_defaults, **kwargs}
        if max_tokens is not None:
            gen_kwargs["max_new_tokens"] = max_tokens
        self.logits_processor.dfa.reset_local_state_counter()
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

    def generate_batch(
        self, prompt: str, n: int, max_tokens: int | None = None, **kwargs
    ) -> list[str]:
        """Generate n sequences from the same prompt in a single model.generate call.

        The diversity mechanism operates across the batch: within the generation call
        local_state_counter penalises tokens used by any sequence in the current batch,
        and path_counter is updated for each completed sequence so that subsequent
        calls (or batches) are also diversified.

        Note: diversity pressure during a batch reflects path_counter state from
        *previous* calls only, not the sequences currently being generated in parallel.
        """
        gen_kwargs = {**self.generation_defaults, **kwargs}
        if max_tokens is not None:
            gen_kwargs["max_new_tokens"] = max_tokens
        self.logits_processor.dfa.reset_local_state_counter()
        self.logits_processor.reset()

        inputs = self.tokenizer([prompt] * n, return_tensors="pt").to(self.model.device)
        prompt_len = inputs["input_ids"].shape[1]

        output_ids = self.model.generate(
            **inputs,
            logits_processor=LogitsProcessorList([self.logits_processor]),
            do_sample=True,
            **gen_kwargs,
        )

        results = []
        for i in range(n):
            new_ids = output_ids[i, prompt_len:]
            text = self.tokenizer.decode(new_ids, skip_special_tokens=True)
            results.append(text)
            self.logits_processor.dfa.update_path_counter(text)

        return results

    def update_generated_content(self, generated_content: str) -> None:
        """Update the global path counter with a previously generated string.

        Must be called after each successful generation when using
        :meth:`__call__` directly, so that subsequent calls are diversified
        relative to *generated_content*.  Not needed when using
        :meth:`generate_batch`, which updates counters automatically.
        """
        self.logits_processor.dfa.update_path_counter(generated_content)


def diverse_regex(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    regex_str: str,
    gamma: float = 0.5,
    beta: float = 3.0,
    ablation_component: str | None = None,
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
    ablation_component
        If not ``None``, one of ``"no_reward"``, ``"no_penalty"``, or
        ``"no_range_scaling"``, to ablate the corresponding component of the
        diversity mechanism for analysis.
    **generation_kwargs
        Additional kwargs forwarded to model.generate().
    """
    no_reward = ablation_component == "reward"
    no_penalty = ablation_component == "penalty"
    no_range_scaling = ablation_component == "range_scaling"

    logits_processor = DiverseRegexLogitsProcessor(
        regex_str,
        tokenizer,
        gamma=gamma,
        beta=beta,
        no_reward=no_reward,
        no_penalty=no_penalty,
        no_range_scaling=no_range_scaling,
    )
    return StatefulSequenceGeneratorAdapter(
        model, tokenizer, logits_processor, **generation_kwargs
    )


def baseline_regex(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    regex_str: str,
    **generation_kwargs,
) -> StatefulSequenceGeneratorAdapter:
    """Create a non-diverse regex-constrained baseline generator."""
    logits_processor = RegexMaskLogitsProcessor(regex_str, tokenizer)
    return StatefulSequenceGeneratorAdapter(
        model, tokenizer, logits_processor, **generation_kwargs
    )
