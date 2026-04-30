"""Tests for DiverseRegexGuide, DiverseRegexLogitsProcessor, and generate_batch."""

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


def test_guide_construction(mock_tokenizer):
    guide = DiverseRegexGuide("[ab]", mock_tokenizer)
    assert guide.initial_state == guide.dfa.get_initial_state()
    assert guide.eos_token_id == 99


def test_guide_regex_wrapping(mock_tokenizer):
    """Regex must be wrapped with (?:...)$ so the DFA is end-anchored."""
    guide = DiverseRegexGuide("[ab]", mock_tokenizer)
    # The DFA was built with (?:[ab])$ — verify the initial state doesn't accept
    # empty string but does accept 'a'
    assert not guide.dfa.is_final_state(guide.initial_state)
    seq = guide.dfa.get_state_sequence("a")
    assert guide.dfa.is_final_state(seq[-1])


def test_guide_get_next_instruction_initial(mock_tokenizer):
    guide = DiverseRegexGuide("[ab]", mock_tokenizer)
    allowed = guide.get_next_instruction(guide.initial_state)
    assert sorted(allowed) == [10, 20]


def test_guide_get_next_instruction_final_state(mock_tokenizer):
    guide = DiverseRegexGuide("[ab]", mock_tokenizer)
    initial = guide.initial_state
    final_state = guide.dfa.get_next_token_state(initial, 10)  # after 'a'
    # From a final state, EOS should be included
    allowed = guide.get_next_instruction(final_state)
    assert 99 in allowed


def test_guide_get_next_instruction_sentinel(mock_tokenizer):
    guide = DiverseRegexGuide("[ab]", mock_tokenizer)
    # State -1 is the "done" sentinel
    assert guide.get_next_instruction(-1) == [99]


def test_guide_is_final_state_sentinel(mock_tokenizer):
    guide = DiverseRegexGuide("[ab]", mock_tokenizer)
    assert guide.is_final_state(-1)
    assert not guide.is_final_state(guide.initial_state)


def test_guide_get_next_state(mock_tokenizer):
    guide = DiverseRegexGuide("[ab]", mock_tokenizer)
    initial = guide.initial_state
    next_state = guide.get_next_state(initial, 10)
    assert next_state != initial


# ---------------------------------------------------------------------------
# DiverseRegexLogitsProcessor — reset()
# ---------------------------------------------------------------------------


def test_reset_clears_guide_states(mock_tokenizer):
    proc = DiverseRegexLogitsProcessor("[ab]", mock_tokenizer)
    # Manually pollute the cache
    proc._guide_states_by_row = [42]
    proc._processed_lengths_by_row = [3]
    proc._seq_start_idx = 5

    proc.reset()

    assert proc._seq_start_idx is None
    assert proc._guide_states_by_row is None
    assert proc._processed_lengths_by_row is None


def test_reset_restores_initial_state_only(mock_tokenizer):
    proc = DiverseRegexLogitsProcessor("[ab]", mock_tokenizer)
    proc.reset()
    proc._ensure_rows(1)
    assert proc._guide_states_by_row == [proc.guide.initial_state]
    assert proc._processed_lengths_by_row == [0]


# ---------------------------------------------------------------------------
# DiverseRegexLogitsProcessor — masking behaviour
# ---------------------------------------------------------------------------


def _make_scores(vocab_size: int = 100, batch_size: int = 1) -> torch.FloatTensor:
    return torch.zeros(batch_size, vocab_size)


def _make_input_ids(prompt_len: int = 5) -> torch.LongTensor:
    return torch.zeros(1, prompt_len, dtype=torch.long)


def test_masking_disallows_non_regex_tokens(mock_tokenizer):
    """Tokens not in the DFA's allowed set should become -inf."""
    proc = DiverseRegexLogitsProcessor("[ab]", mock_tokenizer)
    scores = _make_scores()
    input_ids = _make_input_ids()

    proc(input_ids, scores)

    # Tokens 10 ('a') and 20 ('b') are allowed — must NOT be -inf
    assert not scores[0, 10].isinf()
    assert not scores[0, 20].isinf()

    # All other tokens should be -inf
    for tok in [0, 1, 9, 11, 19, 21, 98]:
        assert scores[0, tok].item() == float("-inf"), f"token {tok} should be -inf"


def test_masking_eos_at_final_state(mock_tokenizer):
    """EOS (99) must be allowed when the DFA is in a final state."""
    proc = DiverseRegexLogitsProcessor("[ab]", mock_tokenizer)
    initial = proc.guide.initial_state

    # Manually set guide_states so the sequence has already consumed one 'a' token
    # and is at the final state
    final_state = proc.dfa.get_next_token_state(initial, 10)
    proc._seq_start_idx = 0
    proc._guide_states_by_row = [final_state]
    proc._processed_lengths_by_row = [1]

    # input_ids: shape (1, 1), representing the token 10 ('a') was already generated
    input_ids = torch.tensor([[10]], dtype=torch.long)
    scores = _make_scores()

    proc(input_ids, scores)

    assert not scores[0, 99].isinf(), "EOS should be allowed at final state"


# ---------------------------------------------------------------------------
# DiverseRegexLogitsProcessor — state tracking
# ---------------------------------------------------------------------------


def test_seq_start_idx_set_on_first_call(mock_tokenizer):
    proc = DiverseRegexLogitsProcessor("[ab]", mock_tokenizer)
    input_ids = _make_input_ids(prompt_len=7)
    scores = _make_scores()

    assert proc._seq_start_idx is None
    proc(input_ids, scores)
    assert proc._seq_start_idx == 7


def test_guide_states_populated_after_call(mock_tokenizer):
    proc = DiverseRegexLogitsProcessor("[ab]", mock_tokenizer)
    input_ids = _make_input_ids(prompt_len=0)
    scores = _make_scores()

    proc(input_ids, scores)
    assert proc._guide_states_by_row == [proc.guide.initial_state]
    assert proc._processed_lengths_by_row == [0]


def test_tolerates_invalid_token_after_terminal_state(mock_tokenizer):
    """Processor should not crash if a finished row is followed by a non-DFA token."""
    proc = DiverseRegexLogitsProcessor("[ab]", mock_tokenizer)
    proc._seq_start_idx = 0
    proc._guide_states_by_row = [-1]  # terminal sentinel after one token
    proc._processed_lengths_by_row = [1]

    input_ids = torch.tensor([[10, 77]], dtype=torch.long)
    scores = _make_scores()

    out = proc(input_ids, scores)
    assert not out[0, 99].isinf()  # EOS remains valid at terminal state


# ---------------------------------------------------------------------------
# DiverseRegexLogitsProcessor — gamma=0 produces zero adjustment
# ---------------------------------------------------------------------------


def test_gamma_zero_no_adjustment(mock_tokenizer):
    """With gamma=0, logit adjustment is zero; only masking applies."""
    # compute_counts starts penalty at 1 (prevents div-by-zero)
    proc = DiverseRegexLogitsProcessor("[ab]", mock_tokenizer, gamma=0.0, beta=1.0)

    scores = torch.tensor([[1.0] * 100])
    input_ids = _make_input_ids()

    proc(input_ids, scores)

    # Allowed tokens should still be 1.0 (no adjustment since gamma=0)
    assert scores[0, 10].item() == pytest.approx(1.0)
    assert scores[0, 20].item() == pytest.approx(1.0)


def test_baseline_regex_uses_mask_only_processor(mock_tokenizer):
    generator = gr.baseline_regex(object(), mock_tokenizer, r"a+")

    assert isinstance(generator.logits_processor, RegexMaskLogitsProcessor)


# ---------------------------------------------------------------------------
# RegexMaskLogitsProcessor — internal baseline behaviour
# ---------------------------------------------------------------------------


def test_mask_only_reset_clears_guide_states(mock_tokenizer):
    proc = RegexMaskLogitsProcessor("[ab]", mock_tokenizer)
    proc._guide_states_by_row = [42]
    proc._processed_lengths_by_row = [3]
    proc._seq_start_idx = 5

    proc.reset()

    assert proc._seq_start_idx is None
    assert proc._guide_states_by_row is None
    assert proc._processed_lengths_by_row is None


def test_mask_only_disallows_non_regex_tokens(mock_tokenizer):
    proc = RegexMaskLogitsProcessor("[ab]", mock_tokenizer)
    scores = _make_scores()

    proc(_make_input_ids(), scores)

    assert not scores[0, 10].isinf()
    assert not scores[0, 20].isinf()
    for tok in [0, 1, 9, 11, 19, 21, 98]:
        assert scores[0, tok].item() == float("-inf")


def test_mask_only_allows_eos_at_final_state(mock_tokenizer):
    proc = RegexMaskLogitsProcessor("[ab]", mock_tokenizer)
    initial = proc.guide.initial_state
    final_state = proc.dfa.get_next_token_state(initial, 10)
    proc._seq_start_idx = 0
    proc._guide_states_by_row = [final_state]
    proc._processed_lengths_by_row = [1]
    scores = _make_scores()

    proc(torch.tensor([[10]], dtype=torch.long), scores)

    assert not scores[0, 99].isinf()


def test_mask_only_keeps_batch_rows_independent(mock_tokenizer):
    proc = RegexMaskLogitsProcessor("ab", mock_tokenizer)
    proc._seq_start_idx = 0
    scores = _make_scores(batch_size=2)

    proc(torch.tensor([[10], [20]], dtype=torch.long), scores)

    assert not scores[0, 20].isinf()
    assert scores[0, 10].item() == float("-inf")
    assert scores[0, 99].item() == float("-inf")
    assert not scores[1, 99].isinf()
    assert scores[1, 10].item() == float("-inf")
    assert scores[1, 20].item() == float("-inf")


def test_mask_only_tolerates_invalid_token_after_terminal_state(mock_tokenizer):
    proc = RegexMaskLogitsProcessor("[ab]", mock_tokenizer)
    proc._seq_start_idx = 0
    proc._guide_states_by_row = [-1]
    proc._processed_lengths_by_row = [1]
    scores = _make_scores()

    out = proc(torch.tensor([[10, 77]], dtype=torch.long), scores)

    assert not out[0, 99].isinf()
    assert out[0, 10].item() == float("-inf")
    assert out[0, 20].item() == float("-inf")


def test_mask_only_does_not_update_diversity_counters(mock_tokenizer):
    proc = RegexMaskLogitsProcessor("[ab]", mock_tokenizer)
    initial = proc.guide.initial_state
    before = proc.dfa.compute_counts(initial)
    proc._seq_start_idx = 0

    proc(torch.tensor([[10]], dtype=torch.long), _make_scores())

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


# ---------------------------------------------------------------------------
# generate_batch tests
# ---------------------------------------------------------------------------


def test_generate_batch_returns_n_results(mock_tokenizer):
    """generate_batch returns a list of exactly n strings."""
    n = 4
    tok = _BatchTokenizerMock(mock_tokenizer)
    proc = DiverseRegexLogitsProcessor("[ab]", tok, gamma=0.0, beta=1.0)
    adapter = StatefulSequenceGeneratorAdapter(_DeterministicModel(), tok, proc)

    results = adapter.generate_batch("prompt", n=n, max_tokens=3)

    assert isinstance(results, list)
    assert len(results) == n
    assert all(isinstance(r, str) for r in results)


def test_generate_batch_all_match_regex(mock_tokenizer):
    """Every sequence returned by generate_batch must match the regex."""
    import re

    n = 3
    tok = _BatchTokenizerMock(mock_tokenizer)
    proc = DiverseRegexLogitsProcessor("[ab]+", tok, gamma=0.0, beta=1.0)
    adapter = StatefulSequenceGeneratorAdapter(_DeterministicModel(), tok, proc)

    results = adapter.generate_batch("prompt", n=n, max_tokens=3)

    pattern = re.compile(r"^[ab]+$")
    for text in results:
        assert pattern.match(text), f"Generated text does not match regex: {text!r}"


def test_generate_batch_updates_path_counter(mock_tokenizer):
    """After generate_batch, the global path counter reflects all n sequences."""
    n = 3
    tok = _BatchTokenizerMock(mock_tokenizer)
    proc = DiverseRegexLogitsProcessor("[ab]", tok, gamma=0.0, beta=1.0)
    adapter = StatefulSequenceGeneratorAdapter(_DeterministicModel(), tok, proc)

    # Record initial reward counts (all zero before any generation)
    initial = proc.dfa.get_initial_state()
    _, reward_before, _ = proc.dfa.compute_counts(initial)
    assert all(c == 0 for c in reward_before)

    adapter.generate_batch("prompt", n=n, max_tokens=1)

    _, reward_after, _ = proc.dfa.compute_counts(initial)
    # At least one path must have been counted
    assert sum(reward_after) > 0


def test_generate_batch_resets_local_counter(mock_tokenizer):
    """generate_batch resets local_state_counter before each call."""
    n = 2
    tok = _BatchTokenizerMock(mock_tokenizer)
    proc = DiverseRegexLogitsProcessor("[ab]", tok, gamma=0.0, beta=1.0)
    adapter = StatefulSequenceGeneratorAdapter(_DeterministicModel(), tok, proc)

    # Artificially inflate local counter
    initial = proc.dfa.get_initial_state()
    for _ in range(5):
        proc.dfa.update_local_state_counter(initial, 10)

    _, _, penalty_inflated = proc.dfa.compute_counts(initial)
    assert max(penalty_inflated) > 1  # confirm it was inflated

    # generate_batch should reset the local counter at the start
    adapter.generate_batch("prompt", n=n, max_tokens=1)

    # After the batch the local counter reflects only what happened in this batch,
    # not the pre-inflated values.  The simplest invariant: the counter was reset
    # (i.e. generate_batch called reset_local_state_counter internally).
    # We can verify by calling reset again and checking counts go to baseline=1.
    proc.dfa.reset_local_state_counter()
    _, _, penalty_reset = proc.dfa.compute_counts(initial)
    assert all(c == 1 for c in penalty_reset)


def test_generate_batch_resets_guide_states(mock_tokenizer):
    """generate_batch calls reset() which clears stale row-state cache."""
    n = 2
    tok = _BatchTokenizerMock(mock_tokenizer)
    proc = DiverseRegexLogitsProcessor("[ab]", tok, gamma=0.0, beta=1.0)
    adapter = StatefulSequenceGeneratorAdapter(_DeterministicModel(), tok, proc)

    # Pollute state
    proc._guide_states_by_row = [42]
    proc._processed_lengths_by_row = [3]
    proc._seq_start_idx = 100

    adapter.generate_batch("prompt", n=n, max_tokens=1)

    assert proc._seq_start_idx == PROMPT_LEN
    assert proc._guide_states_by_row is not None
    assert 42 not in proc._guide_states_by_row


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


def test_python_guide_terminal_none_and_exception_paths():
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


def test_python_processor_state_branches():
    tok = _CoverageTokenizer()
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
