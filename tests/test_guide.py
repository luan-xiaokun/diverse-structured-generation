"""Tests for DiverseRegexGuide, DiverseRegexLogitsProcessor, and generate_batch."""

import re

import pytest
import torch

from diverse_guide import guide_python as gp
from diverse_guide import guide_rust as gr
from diverse_guide.guide_rust import (
    DiverseRegexGuide,
    DiverseRegexLogitsProcessor,
    RegexMaskLogitsProcessor,
    StatefulSequenceGeneratorAdapter,
)

# ---------------------------------------------------------------------------
# DiverseRegexGuide
# ---------------------------------------------------------------------------


def test_guide_wraps_regex_and_exposes_token_transitions(mock_tokenizer):
    guide = DiverseRegexGuide("[ab]", mock_tokenizer)
    assert guide.initial_state == guide.dfa.get_initial_state()
    assert guide.eos_token_id == 99
    # The regex is wrapped as (?:[ab])$: the empty string is rejected, "a" accepted.
    assert not guide.dfa.is_final_state(guide.initial_state)
    assert guide.dfa.is_final_state(guide.dfa.get_state_sequence("a")[-1])

    assert sorted(guide.get_next_instruction(guide.initial_state)) == [10, 20]
    after_a = guide.get_next_state(guide.initial_state, 10)
    assert after_a != guide.initial_state
    assert 99 in guide.get_next_instruction(after_a)

    # State -1 is the "done" sentinel.
    assert guide.get_next_instruction(-1) == [99]
    assert guide.is_final_state(-1)
    assert not guide.is_final_state(guide.initial_state)


# ---------------------------------------------------------------------------
# Logits processors (diverse and mask-only)
# ---------------------------------------------------------------------------

PROCESSORS = (DiverseRegexLogitsProcessor, RegexMaskLogitsProcessor)


def _make_scores(vocab_size: int = 100, batch_size: int = 1) -> torch.FloatTensor:
    return torch.zeros(batch_size, vocab_size)


def _make_input_ids(prompt_len: int = 5) -> torch.LongTensor:
    return torch.zeros(1, prompt_len, dtype=torch.long)


def test_processors_reset_and_track_row_state(mock_tokenizer):
    for processor_cls in PROCESSORS:
        proc = processor_cls("[ab]", mock_tokenizer)
        proc._guide_states_by_row = [42]
        proc._processed_lengths_by_row = [3]
        proc._seq_start_idx = 5
        proc.reset()
        assert proc._seq_start_idx is None
        assert proc._guide_states_by_row is None
        assert proc._processed_lengths_by_row is None

        proc(_make_input_ids(prompt_len=7), _make_scores())
        assert proc._seq_start_idx == 7
        assert proc._guide_states_by_row == [proc.guide.initial_state]
        assert proc._processed_lengths_by_row == [0]


def test_processors_mask_invalid_tokens_and_allow_eos_at_final_state(
    mock_tokenizer,
):
    for processor_cls in PROCESSORS:
        proc = processor_cls("[ab]", mock_tokenizer)
        scores = _make_scores()
        proc(_make_input_ids(), scores)
        assert not scores[0, 10].isinf()
        assert not scores[0, 20].isinf()
        for tok in [0, 1, 9, 11, 19, 21, 98, 99]:
            assert scores[0, tok].item() == float("-inf"), f"token {tok}"

        # A row that has consumed 'a' is in a final state, so EOS is allowed.
        proc = processor_cls("[ab]", mock_tokenizer)
        proc._seq_start_idx = 0
        proc._guide_states_by_row = [
            proc.dfa.get_next_token_state(proc.guide.initial_state, 10)
        ]
        proc._processed_lengths_by_row = [1]
        scores = _make_scores()
        proc(torch.tensor([[10]], dtype=torch.long), scores)
        assert not scores[0, 99].isinf()


def test_processors_keep_only_eos_after_terminal_state(mock_tokenizer):
    """A finished row followed by a non-DFA token must not crash the processor."""
    for processor_cls in PROCESSORS:
        proc = processor_cls("[ab]", mock_tokenizer)
        proc._seq_start_idx = 0
        proc._guide_states_by_row = [-1]  # terminal sentinel after one token
        proc._processed_lengths_by_row = [1]
        scores = torch.linspace(-1.0, 1.0, 100, dtype=torch.float).unsqueeze(0)

        out = proc(torch.tensor([[10, 77]], dtype=torch.long), scores)

        assert not out[0, 99].isinf()
        assert out[0, 10].item() == float("-inf")
        assert out[0, 20].item() == float("-inf")


def test_allowed_token_tensor_cache_reuses_state_tensors(mock_tokenizer):
    """Repeated calls at the same state reuse one cached tensor per processor."""
    diverse = DiverseRegexLogitsProcessor("[ab]", mock_tokenizer)
    mask_only = RegexMaskLogitsProcessor("[ab]", mock_tokenizer)
    scores = torch.linspace(-1.0, 1.0, 100).unsqueeze(0)

    first = diverse(_make_input_ids(), scores.clone())
    second = diverse(_make_input_ids(), scores.clone())
    fresh = DiverseRegexLogitsProcessor("[ab]", mock_tokenizer)(
        _make_input_ids(), scores.clone()
    )
    assert len(diverse._allowed_token_tensor_cache) == 1
    assert torch.equal(first, second)
    assert torch.equal(first, fresh)

    mask_first = mask_only(_make_input_ids(), scores.clone())
    mask_second = mask_only(_make_input_ids(), scores.clone())
    assert len(mask_only._allowed_token_tensor_cache) == 1
    assert torch.equal(mask_first, mask_second)


def test_gamma_zero_no_adjustment(mock_tokenizer):
    """With gamma=0, logit adjustment is zero; only masking applies."""
    proc = DiverseRegexLogitsProcessor("[ab]", mock_tokenizer, gamma=0.0, beta=1.0)
    scores = torch.tensor([[1.0] * 100])

    proc(_make_input_ids(), scores)

    assert scores[0, 10].item() == pytest.approx(1.0)
    assert scores[0, 20].item() == pytest.approx(1.0)


def test_mask_only_keeps_batch_rows_independent_and_counters_untouched(
    mock_tokenizer,
):
    proc = RegexMaskLogitsProcessor("ab", mock_tokenizer)
    initial = proc.guide.initial_state
    before = proc.dfa.compute_counts(initial)
    proc._seq_start_idx = 0
    scores = _make_scores(batch_size=2)

    proc(torch.tensor([[10], [20]], dtype=torch.long), scores)

    assert not scores[0, 20].isinf()
    assert scores[0, 10].item() == float("-inf")
    assert scores[0, 99].item() == float("-inf")
    assert not scores[1, 99].isinf()
    assert scores[1, 10].item() == float("-inf")
    assert scores[1, 20].item() == float("-inf")
    assert proc.dfa.compute_counts(initial) == before


# ---------------------------------------------------------------------------
# Helpers for generate_batch tests
# ---------------------------------------------------------------------------

PROMPT_LEN = 3
VOCAB_SIZE = 100


class _BatchTokenizerMock:
    """
    Wraps MockTokenizer and adds batch encoding / decode support.
    model.generate only sees integer tensors, so decode just maps token IDs
    deterministically to single characters.
    """

    eos_token_id = 99

    def __init__(self, base_tok):
        self._base = base_tok

    # Batch call used by generate_batch: tokenizer([prompt]*n, return_tensors="pt")
    def __call__(self, prompts, return_tensors=None):
        n = len(prompts)
        input_ids = torch.zeros(n, PROMPT_LEN, dtype=torch.long)
        attention_mask = torch.ones(n, PROMPT_LEN, dtype=torch.long)

        class _Encoding:
            def to(self_, device):  # noqa: N805
                return {"input_ids": input_ids, "attention_mask": attention_mask}

            def __getitem__(self_, key):  # noqa: N805
                return {"input_ids": input_ids, "attention_mask": attention_mask}[key]

        return _Encoding()

    def decode(self, ids, skip_special_tokens=True):
        # Map each token id to a char; token 10 → 'a', token 20 → 'b', else ''
        mapping = {10: "a", 20: "b"}
        return "".join(mapping.get(t.item(), "") for t in ids)

    # Forward property accesses needed by lru_cache wrappers
    @property
    def all_special_tokens(self):
        return self._base.all_special_tokens

    def get_vocab(self):
        return self._base.get_vocab()

    def convert_tokens_to_string(self, tokens):
        return self._base.convert_tokens_to_string(tokens)

    def __hash__(self):
        return id(self)

    def __eq__(self, other):
        return self is other


class _DeterministicModel:
    """
    Mock model whose generate() runs the logits processor for real and always
    picks the first non-(-inf) token at each step, stopping when EOS (99) is chosen
    or max_new_tokens is reached.
    """

    device = "cpu"

    def generate(
        self,
        input_ids,
        attention_mask=None,
        logits_processor=None,
        do_sample=True,
        max_new_tokens=5,
        **kwargs,
    ):
        output = input_ids.clone()
        for _ in range(max_new_tokens):
            scores = torch.zeros(output.shape[0], VOCAB_SIZE)
            if logits_processor:
                for proc in logits_processor:
                    scores = proc(output, scores)
            # Greedily pick first valid token for each sequence in the batch
            next_tokens = []
            for b in range(output.shape[0]):
                valid = (scores[b] != float("-inf")).nonzero(as_tuple=True)[0]
                tok = int(valid[0]) if len(valid) > 0 else 99
                next_tokens.append(tok)
            next_tok_t = torch.tensor(next_tokens, dtype=torch.long).unsqueeze(1)
            output = torch.cat([output, next_tok_t], dim=1)
            if all(t == 99 for t in next_tokens):
                break
        return output


class _RecordingModel(_DeterministicModel):
    """Deterministic mock model that records the kwargs of each generate call."""

    def __init__(self):
        self.calls = []

    def generate(self, input_ids, **kwargs):
        self.calls.append(kwargs)
        return super().generate(input_ids, **kwargs)


def test_adapter_samples_without_model_generation_config_defaults(mock_tokenizer):
    """Unset sampling arguments are pinned to pure multinomial sampling, so the
    model's generation_config is not inherited; constructor and per-call
    arguments still take precedence."""
    tok = _BatchTokenizerMock(mock_tokenizer)
    model = _RecordingModel()
    proc = DiverseRegexLogitsProcessor("[ab]", tok, gamma=0.0, beta=1.0)
    adapter = StatefulSequenceGeneratorAdapter(model, tok, proc, temperature=1.5)

    adapter.generate_batch("prompt", n=2, max_tokens=2)
    adapter.generate_batch("prompt", n=2, max_tokens=2, top_p=0.9)

    expected = {**gr.PURE_SAMPLING_DEFAULTS, "temperature": 1.5}
    for key, value in expected.items():
        assert model.calls[0][key] == value
    assert model.calls[1]["top_p"] == 0.9
    assert model.calls[1]["top_k"] == 0
    assert gr.baseline_regex(model, tok, r"a+").generation_defaults["top_k"] == 0


# ---------------------------------------------------------------------------
# generate_batch tests
# ---------------------------------------------------------------------------


def test_generate_batch_returns_valid_samples_and_counts_them(mock_tokenizer):
    tok = _BatchTokenizerMock(mock_tokenizer)
    proc = DiverseRegexLogitsProcessor("[ab]+", tok, gamma=0.0, beta=1.0)
    adapter = StatefulSequenceGeneratorAdapter(_DeterministicModel(), tok, proc)
    initial = proc.dfa.get_initial_state()
    assert all(c == 0 for c in proc.dfa.compute_counts(initial)[1])

    results = adapter.generate_batch("prompt", n=3, max_tokens=3)

    assert len(results) == 3
    assert all(re.fullmatch(r"[ab]+", text) for text in results)
    assert sum(proc.dfa.compute_counts(initial)[1]) > 0


def test_generate_batch_resets_per_call_state(mock_tokenizer):
    """generate_batch clears the local state counter and stale row states."""
    tok = _BatchTokenizerMock(mock_tokenizer)
    proc = DiverseRegexLogitsProcessor("[ab]", tok, gamma=0.0, beta=1.0)
    adapter = StatefulSequenceGeneratorAdapter(_DeterministicModel(), tok, proc)
    initial = proc.dfa.get_initial_state()
    for _ in range(5):
        proc.dfa.update_local_state_counter(initial, 10)
    assert max(proc.dfa.compute_counts(initial)[2]) > 1
    proc._guide_states_by_row = [42]
    proc._processed_lengths_by_row = [3]
    proc._seq_start_idx = 100

    # With max_tokens=1 no generated token is replayed into the local counter.
    adapter.generate_batch("prompt", n=2, max_tokens=1)

    assert all(c == 1 for c in proc.dfa.compute_counts(initial)[2])
    assert proc._seq_start_idx == PROMPT_LEN
    assert proc._guide_states_by_row is not None
    assert 42 not in proc._guide_states_by_row


def _reward_by_token(proc, state):
    token_ids, reward_counts, _ = proc.dfa.compute_counts(state)
    return dict(zip(token_ids, reward_counts, strict=True))


def test_generate_batch_skips_unfinished_samples(mock_tokenizer):
    """Samples cut off before an accepting state do not update path counters."""
    tok = _BatchTokenizerMock(mock_tokenizer)
    proc = DiverseRegexLogitsProcessor("ab", tok, gamma=0.0, beta=1.0)
    adapter = StatefulSequenceGeneratorAdapter(_DeterministicModel(), tok, proc)
    initial = proc.dfa.get_initial_state()

    assert adapter.generate_batch("prompt", n=2, max_tokens=1) == ["a", "a"]
    assert all(c == 0 for c in _reward_by_token(proc, initial).values())

    assert adapter.generate_batch("prompt", n=2, max_tokens=3) == ["ab", "ab"]
    assert _reward_by_token(proc, initial)[10] == 2


def test_generate_batch_counts_eos_transition_once(mock_tokenizer):
    """The final EOS transition is counted once per completed sample."""
    tok = _BatchTokenizerMock(mock_tokenizer)
    proc = DiverseRegexLogitsProcessor("[ab]", tok, gamma=0.0, beta=1.0)
    adapter = StatefulSequenceGeneratorAdapter(_DeterministicModel(), tok, proc)
    final_state = proc.guide.get_next_state(proc.guide.initial_state, 10)

    # max_tokens=1 stops before EOS; max_tokens=4 emits EOS and then padding.
    adapter.generate_batch("prompt", n=2, max_tokens=1)
    adapter.generate_batch("prompt", n=2, max_tokens=4)

    assert _reward_by_token(proc, final_state)[99] == 4


def test_update_generated_content_counts_only_full_matches(mock_tokenizer):
    tok = _BatchTokenizerMock(mock_tokenizer)
    proc = DiverseRegexLogitsProcessor("ab", tok, gamma=0.0, beta=1.0)
    adapter = StatefulSequenceGeneratorAdapter(_DeterministicModel(), tok, proc)
    initial = proc.dfa.get_initial_state()
    final_state = proc.dfa.get_state_sequence("ab")[-1]

    for text in ["a", "ax", "abb", ""]:
        adapter.update_generated_content(text)
    assert all(c == 0 for c in _reward_by_token(proc, initial).values())

    adapter.update_generated_content("ab")
    assert _reward_by_token(proc, initial)[10] == 1
    assert _reward_by_token(proc, final_state)[99] == 1


def test_python_reference_counts_only_full_matches():
    tok = _CoverageTokenizer(pad_token_id=None)
    proc = gp.DiverseRegexLogitsProcessor("[ab]b", tok, gamma=0.5, beta=3.0)
    adapter = gp.StatefulSequenceGeneratorAdapter(_CoverageModel(), tok, proc)

    for text in ["a", "ax", "abb"]:
        adapter.update_generated_content(text)
    assert sum(proc.transition_counter.values()) == 0

    adapter.update_generated_content("ab")
    final_state = proc.guide.dfa.get_state_sequence("ab")[-1]
    assert proc.transition_counter[(final_state, final_state)] == 1


class _CoverageTokenizer:
    """Tokenizer stub for guide branch-coverage tests."""

    def __init__(self, pad_token_id: int | None = None):
        self.eos_token_id = 99
        self.pad_token_id = pad_token_id
        self._id = id(self)

    @property
    def all_special_tokens(self):
        return ["<eos>"]

    def get_vocab(self):
        return {"a": 10, "b": 20, "<eos>": 99}

    def convert_tokens_to_string(self, tokens):
        return "".join(tokens)

    def __call__(self, prompts, return_tensors=None):
        n = len(prompts) if isinstance(prompts, list) else 1
        input_ids = torch.zeros(n, 2, dtype=torch.long)
        attention_mask = torch.ones(n, 2, dtype=torch.long)

        class _Enc:
            def to(self_inner, device):  # noqa: ANN001
                return {"input_ids": input_ids, "attention_mask": attention_mask}

            def __getitem__(self_inner, key):  # noqa: ANN001
                return {"input_ids": input_ids, "attention_mask": attention_mask}[key]

        return _Enc()

    def decode(self, ids, skip_special_tokens=True):  # noqa: ANN001
        mapping = {10: "a", 20: "b"}
        return "".join(
            mapping.get(int(t), "") for t in ids if int(t) != self.eos_token_id
        )

    def __hash__(self):
        return self._id

    def __eq__(self, other):
        return self is other


class _CoverageModel:
    """Model stub for guide branch-coverage tests."""

    device = "cpu"

    def generate(  # noqa: PLR0913
        self,
        input_ids,
        attention_mask=None,
        logits_processor=None,
        do_sample=True,
        max_new_tokens=2,
        **kwargs,
    ):
        output = input_ids.clone()
        for _ in range(max_new_tokens):
            scores = torch.zeros(output.shape[0], 128)
            if logits_processor:
                for proc in logits_processor:
                    scores = proc(output, scores)
            next_tokens = []
            for b in range(output.shape[0]):
                valid = (scores[b] != float("-inf")).nonzero(as_tuple=True)[0]
                next_tokens.append(int(valid[0]) if len(valid) > 0 else 99)
            next_tok_t = torch.tensor(next_tokens, dtype=torch.long).unsqueeze(1)
            output = torch.cat(
                [output, next_tok_t],
                dim=1,
            )
            if all(t == 99 for t in next_tokens):
                break
        return output


def _make_scores_cov(batch: int = 1, vocab: int = 128):
    return torch.zeros(batch, vocab)


def test_python_reference_guide_and_processor_branches():
    tok = _CoverageTokenizer()
    guide = gp.DiverseRegexGuide("[ab]", tok)
    assert guide.get_next_instruction(-1) == [tok.eos_token_id]

    class _DfaNone:
        def get_allowed_token_ids(self, state):  # noqa: ANN001
            return None

    guide_none = gp.DiverseRegexGuide.__new__(gp.DiverseRegexGuide)
    guide_none.eos_token_id = tok.eos_token_id
    guide_none.dfa = _DfaNone()
    assert guide_none.get_next_instruction(0) == [tok.eos_token_id]

    class _DfaRaise:
        def get_byte_state_sequence(self, state, token_id):  # noqa: ANN001
            raise ValueError("x")

    guide_raise = gp.DiverseRegexGuide.__new__(gp.DiverseRegexGuide)
    guide_raise.dfa = _DfaRaise()
    assert guide_raise.get_state_sequence_from_token_id(0, 10) == []

    proc = gp.DiverseRegexLogitsProcessor("[ab]", tok)
    proc._seq_start_idx = 0
    proc._guide_states[(10,)] = -1
    out = proc(torch.tensor([[10, 77]], dtype=torch.long), _make_scores_cov())
    assert not out[0, tok.eos_token_id].isinf()

    proc.reset()
    proc._seq_start_idx = 0
    proc(torch.tensor([[10]], dtype=torch.long), _make_scores_cov())
    assert len(proc.local_state_counter) > 0

    proc.reset()
    proc._seq_start_idx = 0
    valid_next = proc.guide.get_next_state(proc.guide.initial_state, 10)
    proc._guide_states[(10,)] = valid_next
    proc(torch.tensor([[10, 77]], dtype=torch.long), _make_scores_cov())


def test_python_adapter_and_factory_coverage():
    tok = _CoverageTokenizer(pad_token_id=None)
    model = _CoverageModel()
    proc = gp.DiverseRegexLogitsProcessor("[ab]", tok, gamma=0.5, beta=3.0)
    adapter = gp.StatefulSequenceGeneratorAdapter(model, tok, proc)
    assert adapter.generation_defaults["pad_token_id"] == tok.eos_token_id
    assert isinstance(adapter("prompt", max_tokens=2), str)
    adapter.update_generated_content("a")
    assert sum(proc.transition_counter.values()) >= 1

    d = gp.diverse_regex(model, tok, "[ab]")
    b = gp.baseline_regex(model, tok, "[ab]")
    assert isinstance(d, gp.DiverseGuide)
    assert b.logits_processor.gamma == 0.0


def test_rust_get_next_instruction_none_and_ablation_branches():
    tok = _CoverageTokenizer()

    class _DfaNone:
        def get_allowed_token_ids(self, state):  # noqa: ANN001
            return None

    guide_none = gr.DiverseRegexGuide.__new__(gr.DiverseRegexGuide)
    guide_none.eos_token_id = tok.eos_token_id
    guide_none.dfa = _DfaNone()
    assert guide_none.get_next_instruction(0) == [tok.eos_token_id]

    proc = gr.DiverseRegexLogitsProcessor(
        "[ab]",
        tok,
        gamma=0.5,
        beta=3.0,
        no_reward=True,
        no_penalty=True,
        no_range_scaling=True,
    )
    out = proc(torch.zeros(1, 2, dtype=torch.long), _make_scores_cov())
    assert torch.isfinite(out).any()


def test_rust_adapter_call_update_and_ablation_mapping():
    tok = _CoverageTokenizer(pad_token_id=None)
    model = _CoverageModel()
    proc = gr.DiverseRegexLogitsProcessor("[ab]", tok)
    adapter = gr.StatefulSequenceGeneratorAdapter(model, tok, proc, pad_token_id=0)
    assert adapter.generation_defaults["pad_token_id"] == 0
    assert isinstance(adapter("prompt", max_tokens=2), str)
    adapter.update_generated_content("a")
    baseline = gr.baseline_regex(model, tok, "[ab]")
    assert isinstance(baseline.logits_processor, gr.RegexMaskLogitsProcessor)
    guide = gr.DiverseGuide(model, tok, "[ab]", gamma=0.5, beta=3.0)
    assert isinstance(guide, gr.StatefulSequenceGeneratorAdapter)
    assert isinstance(guide.logits_processor, gr.DiverseRegexLogitsProcessor)
    assert guide.regex_str == "[ab]"
    assert guide.gamma == 0.5
    assert guide.beta == 3.0

    a1 = gr.diverse_regex(model, tok, "[ab]", ablation_component="reward")
    a2 = gr.diverse_regex(model, tok, "[ab]", ablation_component="penalty")
    a3 = gr.diverse_regex(model, tok, "[ab]", ablation_component="range_scaling")
    assert isinstance(a1, gr.DiverseGuide)
    assert a1.logits_processor.no_reward
    assert a2.logits_processor.no_penalty
    assert a3.logits_processor.no_range_scaling
    with pytest.raises(ValueError, match="ablation_component"):
        gr.DiverseGuide(model, tok, "[ab]", ablation_component="no_reward")
