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


def _device_cache_key(device: torch.device) -> tuple[str, int | None]:
    return (device.type, device.index)


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

    def get_distance_to_accepting(self, state: int) -> int:
        """Return the shortest token-level distance to an accepting state."""
        if state == -1:
            return 0
        return self.dfa.get_distance_to_accepting(state)

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
        self._allowed_token_tensor_cache: dict[
            tuple[int, tuple[str, int | None]], torch.LongTensor
        ] = {}

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

    def _allowed_token_tensor(
        self, state: int, device: torch.device
    ) -> torch.LongTensor:
        cache_key = (state, _device_cache_key(device))
        cached = self._allowed_token_tensor_cache.get(cache_key)
        if cached is not None:
            return cached

        allowed_tokens = self.guide.get_next_instruction(state)
        tensor = torch.tensor(allowed_tokens, dtype=torch.long, device=device)
        self._allowed_token_tensor_cache[cache_key] = tensor
        return tensor

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
            allowed_tokens = self._allowed_token_tensor(guide_state, scores.device)
            allowed_tokens_batch.append(allowed_tokens)
            batch_indices.append(torch.full_like(allowed_tokens, row))

        allowed_tokens_concat = torch.cat(allowed_tokens_batch)
        batch_indices_concat = torch.cat(batch_indices)
        allowed_scores = scores[batch_indices_concat, allowed_tokens_concat]
        scores.fill_(float("-inf"))
        scores[batch_indices_concat, allowed_tokens_concat] = allowed_scores
        return scores

    def update_generated_content(self, generated_content: str) -> None:
        """Baseline processor has no cross-call diversity state to update."""
        del generated_content

    def update_generated_token_ids(self, generated_token_ids: torch.Tensor) -> None:
        """Baseline processor has no cross-call diversity state to update."""
        del generated_token_ids


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
                       \\cdot \\frac{1}{(c^{\\text{local}}_i)^2}

    The path counter (global, cross-sequence) is updated incrementally as newly
    generated token ids are consumed.
    The local counter (per ``model.generate`` call) is reset by :meth:`reset`.
    """

    def __init__(
        self,
        regex_string: str,
        tokenizer: PreTrainedTokenizerBase,
        gamma: float = 0.15,
        completion_bias_eta: float = 0.15,
        path_frequency_estimator: str = "harmonic",
        no_reward: bool = False,
        no_penalty: bool = False,
        no_range_scaling: bool = False,
        eos_completion_mode: str = "legacy",
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
            on under-explored paths. Set to ``0`` to disable the reward
            adjustment for this processor. The public internal baseline uses
            :class:`RegexMaskLogitsProcessor`.
        completion_bias_eta:
            Small bonus weight for tokens that move the DFA closer to an
            accepting state. Set to ``0.0`` to disable the completion bias.
        path_frequency_estimator:
            Either ``"min"`` for the original token-path frequency estimate or
            ``"harmonic"`` for shifted harmonic-mean path frequency.
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
        if completion_bias_eta < 0.0:
            raise ValueError("completion_bias_eta must be non-negative")
        if path_frequency_estimator not in {"min", "harmonic"}:
            raise ValueError("path_frequency_estimator must be 'min' or 'harmonic'")
        self.gamma = gamma
        self.completion_bias_eta = completion_bias_eta
        self.path_frequency_estimator = path_frequency_estimator
        self.no_reward = no_reward
        self.no_penalty = no_penalty
        self.no_range_scaling = no_range_scaling
        if eos_completion_mode not in {"legacy", "constant", "entry_count"}:
            raise ValueError(
                "eos_completion_mode must be legacy, constant or entry_count"
            )
        self.eos_completion_mode = eos_completion_mode
        self._accepting_entries: list[int] = []
        self._seq_start_idx: int | None = None
        self._guide_states_by_row: list[int] | None = None
        self._processed_lengths_by_row: list[int] | None = None
        self._allowed_token_tensor_cache: dict[
            tuple[int, tuple[str, int | None]], torch.LongTensor
        ] = {}

    def reset(self) -> None:
        """Reset per-call state. Must be called before each new ``model.generate()``
        call."""
        self._seq_start_idx = None
        self._guide_states_by_row = None
        self._processed_lengths_by_row = None
        self._accepting_entries = []

    @property
    def accepting_entries(self) -> tuple[int, ...]:
        """Accepting-set entries at consumed token boundaries, per batch row."""
        return tuple(self._accepting_entries)

    @property
    def sequence_states(self) -> tuple[int, ...]:
        """Current consumed-token states, exposed for read-only diagnostics."""
        return tuple(self._guide_states_by_row or ())

    def _ensure_rows(self, batch_size: int) -> None:
        if (
            self._guide_states_by_row is None
            or len(self._guide_states_by_row) != batch_size
        ):
            self._guide_states_by_row = [self.guide.initial_state] * batch_size
            self._processed_lengths_by_row = [0] * batch_size
            self._accepting_entries = [
                int(self.dfa.is_final_state(self.guide.initial_state))
            ] * batch_size

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
                    if (
                        token_id != self.guide.eos_token_id
                        and not self.dfa.is_final_state(prev_state)
                        and self.dfa.is_final_state(state)
                    ):
                        self._accepting_entries[row] += 1
            processed += 1

        self._guide_states_by_row[row] = state
        self._processed_lengths_by_row[row] = processed
        return state

    def _compute_adjustments_from_counts(
        self,
        reward_counts: list[float],
        penalty_counts: list[int],
    ) -> torch.FloatTensor:
        reward_counts_tensor = torch.tensor(reward_counts, dtype=torch.float)
        penalty_counts_tensor = torch.tensor(penalty_counts, dtype=torch.float)
        reward = torch.log(1 + reward_counts_tensor.sum()) / (1 + reward_counts_tensor)
        penalty = penalty_counts_tensor**2

        if self.no_reward:
            reward = torch.ones_like(reward_counts_tensor)
        if self.no_penalty:
            penalty = torch.ones_like(penalty_counts_tensor)

        return reward / penalty

    def _allowed_token_tensor(
        self, state: int, device: torch.device
    ) -> torch.LongTensor:
        cache_key = (state, _device_cache_key(device))
        allowed_token_tensor = self._allowed_token_tensor_cache.get(cache_key)
        if allowed_token_tensor is None:
            allowed_tokens = self.guide.get_next_instruction(state)
            allowed_token_tensor = torch.tensor(
                allowed_tokens, dtype=torch.long, device=device
            )
            self._allowed_token_tensor_cache[cache_key] = allowed_token_tensor
        return allowed_token_tensor

    def _compute_row_allowed_tokens_and_adjustments(
        self, guide_state: int, device: torch.device
    ) -> tuple[torch.LongTensor, torch.FloatTensor]:
        if guide_state == -1:
            adjustments = self._compute_adjustments_from_counts([0], [1]).to(device)
            return (
                self._allowed_token_tensor(guide_state, device),
                adjustments,
            )

        cache_key = (guide_state, _device_cache_key(device))
        allowed_token_tensor = self._allowed_token_tensor_cache.get(cache_key)
        if allowed_token_tensor is None:
            allowed_tokens, adjustments = self.dfa.compute_adjustments_with_estimator(
                guide_state,
                self.no_reward,
                self.no_penalty,
                self.completion_bias_eta,
                self.path_frequency_estimator,
            )
            allowed_token_tensor = torch.tensor(
                allowed_tokens, dtype=torch.long, device=device
            )
            self._allowed_token_tensor_cache[cache_key] = allowed_token_tensor
        else:
            adjustments = self.dfa.compute_adjustment_values_with_estimator(
                guide_state,
                self.no_reward,
                self.no_penalty,
                self.completion_bias_eta,
                self.path_frequency_estimator,
            )
        return (
            allowed_token_tensor,
            torch.tensor(adjustments, dtype=torch.float, device=device),
        )

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
            allowed_tokens, adjustments = (
                self._compute_row_allowed_tokens_and_adjustments(guide_state, device)
            )
            if (
                self.eos_completion_mode != "legacy"
                and self.completion_bias_eta != 0.0
                and guide_state != -1
                and self.dfa.is_final_state(guide_state)
            ):
                signal = (
                    self._accepting_entries[i]
                    if self.eos_completion_mode == "entry_count"
                    else 1
                )
                adjustments[allowed_tokens == self.guide.eos_token_id] += (
                    self.completion_bias_eta * signal
                )
            allowed_tokens_batch.append(allowed_tokens)
            batch_indices.append(torch.full_like(allowed_tokens, i))
            adjusts.append(adjustments)

        allowed_tokens_concat = torch.cat(allowed_tokens_batch)
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

    def update_generated_content(self, generated_content: str) -> None:
        # Rust-backed modes commit valid completed token sequences through
        # update_generated_token_ids().
        del generated_content

    def update_generated_token_ids(self, generated_token_ids: torch.Tensor) -> None:
        """Commit generated token paths to global counters only if samples are valid."""
        if generated_token_ids.ndim == 1:
            generated_token_ids = generated_token_ids.unsqueeze(0)
        for gen_ids in generated_token_ids:
            self.dfa.commit_valid_token_path([int(token_id) for token_id in gen_ids])


class StatefulSequenceGeneratorAdapter:
    """Wraps a HuggingFace model and :class:`DiverseRegexLogitsProcessor` for
    stateful generation.

    Maintains a cross-call path counter so that successive calls (or batches)
    are diversified relative to valid completed token paths generated so far.

    Typical usage::

        generator = DiverseGuide(model, tokenizer, regex_str)
        for _ in range(100):
            text = generator(prompt, max_tokens=40)

    The logits processor updates local counters during generation and commits
    global path counters after generation only for valid completed samples.
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
        self.logits_processor.update_generated_token_ids(new_ids)
        text = self.tokenizer.decode(new_ids, skip_special_tokens=True)
        self.logits_processor.update_generated_content(text)
        return text

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
        self.logits_processor.update_generated_token_ids(output_ids[:, prompt_len:])
        for i in range(n):
            new_ids = output_ids[i, prompt_len:]
            text = self.tokenizer.decode(new_ids, skip_special_tokens=True)
            results.append(text)
            self.logits_processor.update_generated_content(text)

        return results

    def update_generated_content(self, generated_content: str) -> None:
        """Update the global path counter with a previously generated string.

        Deprecated for Rust-backed diverse modes. Runtime path counters are
        committed automatically from generated token ids only when the completed
        sample is valid.
        """
        self.logits_processor.update_generated_content(generated_content)


class DiverseGuide(StatefulSequenceGeneratorAdapter):
    """User-facing generator for diverse regex-constrained generation.

    ``DiverseGuide`` is the canonical public API for the DiverseGuide software.
    It compiles the regex constraint into a DFA-backed logits processor and
    exposes the same single-sequence and batch generation methods as
    :class:`StatefulSequenceGeneratorAdapter`.
    """

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        regex_str: str,
        gamma: float = 0.15,
        completion_bias_eta: float = 0.15,
        path_frequency_estimator: str = "harmonic",
        ablation_component: str | None = None,
        eos_completion_mode: str = "legacy",
        **generation_kwargs,
    ):
        no_reward = ablation_component == "reward"
        no_penalty = ablation_component == "penalty"
        no_range_scaling = ablation_component == "range_scaling"
        logits_processor = DiverseRegexLogitsProcessor(
            regex_str,
            tokenizer,
            gamma=gamma,
            completion_bias_eta=completion_bias_eta,
            path_frequency_estimator=path_frequency_estimator,
            no_reward=no_reward,
            no_penalty=no_penalty,
            no_range_scaling=no_range_scaling,
            eos_completion_mode=eos_completion_mode,
        )
        super().__init__(model, tokenizer, logits_processor, **generation_kwargs)
        self.regex_str = regex_str
        self.gamma = gamma
        self.completion_bias_eta = completion_bias_eta
        self.path_frequency_estimator = path_frequency_estimator
        self.ablation_component = ablation_component
        self.eos_completion_mode = eos_completion_mode


def diverse_regex(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    regex_str: str,
    gamma: float = 0.15,
    completion_bias_eta: float = 0.15,
    path_frequency_estimator: str = "harmonic",
    ablation_component: str | None = None,
    eos_completion_mode: str = "legacy",
    **generation_kwargs,
) -> StatefulSequenceGeneratorAdapter:
    """Create a diverse regex-constrained generator.

    This compatibility helper returns :class:`DiverseGuide`. New code should
    instantiate ``DiverseGuide`` directly.

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
    completion_bias_eta
        Small bonus weight for tokens that move the DFA closer to acceptance.
    path_frequency_estimator
        Either ``"min"`` or ``"harmonic"`` for token-path frequency estimation.
    ablation_component
        If not ``None``, one of ``"no_reward"``, ``"no_penalty"``, or
        ``"no_range_scaling"``, to ablate the corresponding component of the
        diversity mechanism for analysis.
    **generation_kwargs
        Additional kwargs forwarded to model.generate().
    """
    return DiverseGuide(
        model,
        tokenizer,
        regex_str,
        gamma=gamma,
        completion_bias_eta=completion_bias_eta,
        path_frequency_estimator=path_frequency_estimator,
        ablation_component=ablation_component,
        eos_completion_mode=eos_completion_mode,
        **generation_kwargs,
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
