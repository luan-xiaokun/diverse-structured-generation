"""EOS completion behavior at the public logits-processor boundary."""

import json
import subprocess
import types
from pathlib import Path

import pytest
import torch

from diverse_guide import guide_python, guide_rust


def test_legacy_matches_group2_frozen_processor(mock_tokenizer):
    source = subprocess.check_output(
        ["git", "show", "2c3b2a2:src/diverse_guide/guide_rust.py"],
        text=True,
        cwd=Path(__file__).parents[1],
    )
    frozen = types.ModuleType("diverse_guide._group2_frozen")
    frozen.__package__ = "diverse_guide"
    exec(compile(source, "group2_frozen", "exec"), frozen.__dict__)
    old = frozen.DiverseRegexLogitsProcessor("a(?:ba)*", mock_tokenizer)
    new = guide_rust.DiverseRegexLogitsProcessor("a(?:ba)*", mock_tokenizer)
    for tokens in [[10, 20, 10, 99], [10, 20], [10, 99]]:
        for p in [old, new]:
            p.dfa.reset_local_state_counter()
            p.reset()
        prefix = [0]
        for token in [None, *tokens]:
            if token is not None:
                prefix.append(token)
            assert torch.equal(walk(old, [prefix]), walk(new, [prefix]))
            assert old.dfa.compute_counts(
                old.guide.initial_state
            ) == new.dfa.compute_counts(new.guide.initial_state)
        for p in [old, new]:
            p.update_generated_token_ids(torch.tensor(tokens))
        assert old.dfa.compute_counts(
            old.guide.initial_state
        ) == new.dfa.compute_counts(new.guide.initial_state)


@pytest.mark.parametrize("module", [guide_rust, guide_python])
def test_reentry_grows_only_accepting_eos_bonus(module, mock_tokenizer):
    processors = {
        mode: module.DiverseRegexLogitsProcessor(
            "a(?:ba)*",
            mock_tokenizer,
            gamma=0.3,
            completion_bias_eta=2.0,
            eos_completion_mode=mode,
        )
        for mode in ("legacy", "constant", "entry_count")
    }
    prefix = [0]
    for token, count in [(None, 0), (10, 1), (20, 1), (10, 2)]:
        if token is not None:
            prefix.append(token)
        scores = torch.zeros(1, 100)
        scores[0, 20], scores[0, 99] = 2.0, 6.0
        results = {
            mode: proc(torch.tensor([prefix]), scores.clone())
            for mode, proc in processors.items()
        }
        assert processors["entry_count"].accepting_entries == (count,)
        accepting = token == 10
        assert torch.equal(results["legacy"][:, :99], results["entry_count"][:, :99])
        if accepting:
            assert (
                results["entry_count"][0, 99] - results["legacy"][0, 99]
            ).item() == pytest.approx(2.4 * count)
            assert (
                results["constant"][0, 99] - results["legacy"][0, 99]
            ).item() == pytest.approx(2.4)
        else:
            assert torch.equal(results["legacy"], results["entry_count"])


class Tokenizer:
    eos_token_id = 99
    all_special_tokens = ["<eos>"]

    def __init__(self, vocab=None):
        self.vocab = vocab or {"a": 10, "b": 20, "ab": 30, "ba": 40, "<eos>": 99}

    def get_vocab(self):
        return self.vocab

    def convert_tokens_to_string(self, tokens):
        return "".join(tokens)


def walk(proc, rows):
    scores = torch.arange(100, dtype=torch.float).repeat(len(rows), 1)
    return proc(torch.tensor(rows), scores)


@pytest.mark.parametrize("module", [guide_rust, guide_python])
def test_boundaries_rows_repeat_reset_and_eos(module):
    p = module.DiverseRegexLogitsProcessor(
        "a(?:ba)*", Tokenizer(), eos_completion_mode="entry_count"
    )
    walk(p, [[0], [0]])
    walk(p, [[0, 10], [0, 30]])  # ab crosses F internally but finishes outside F
    assert p.accepting_entries == (1, 0)
    before = walk(p, [[0, 10], [0, 30]])
    assert torch.equal(before, walk(p, [[0, 10], [0, 30]]))
    walk(p, [[0, 10, 40], [0, 30, 10]])  # F -> F does not count internal exit/reentry
    assert p.accepting_entries == (1, 1)
    walk(p, [[0, 10, 40, 99], [0, 30, 10, 99]])
    assert p.accepting_entries == (1, 1)
    p.reset()
    walk(p, [[0], [0]])
    assert p.accepting_entries == (0, 0)
    walk(p, [[0, 10], [0, 20]])  # failed row cannot gain an accepting entry
    assert p.accepting_entries == (1, 0)


@pytest.mark.parametrize("module", [guide_rust, guide_python])
@pytest.mark.parametrize(
    "regex,tokens,counts",
    [
        ("a*", [10, 10, 99], [1, 1, 1]),
        ("a|ab|aba", [10, 20, 10], [1, 1, 1]),
        ("a(?:ba)*|abb", [10, 20, 20], [1, 1, 2]),
    ],
)
def test_accepting_set_and_initial_state(module, regex, tokens, counts):
    p = module.DiverseRegexLogitsProcessor(
        regex, Tokenizer(), eos_completion_mode="entry_count"
    )
    prefix = [0]
    walk(p, [prefix])
    assert p.accepting_entries == (int(regex == "a*"),)
    for token, count in zip(tokens, counts):
        prefix.append(token)
        walk(p, [prefix])
        assert p.accepting_entries == (count,)


@pytest.mark.parametrize("eta,gamma", [(0, 0.3), (2, 0)])
@pytest.mark.parametrize("module", [guide_rust, guide_python])
def test_zero_weights_disable_new_mode(module, eta, gamma):
    ps = [
        module.DiverseRegexLogitsProcessor(
            "a(?:ba)*",
            Tokenizer(),
            gamma=gamma,
            completion_bias_eta=eta,
            eos_completion_mode=mode,
        )
        for mode in ["legacy", "entry_count", "constant"]
    ]
    for prefix in [[0], [0, 10], [0, 10, 20], [0, 10, 20, 10]]:
        results = [walk(p, [prefix]) for p in ps]
        assert all(torch.equal(results[0], r) for r in results[1:])


@pytest.mark.parametrize(
    "regex,no_range,expected",
    [("a(?:ba)*", True, 0.6), ("a", False, 0.0), ("a", True, 0.6)],
)
def test_no_range_and_only_eos(regex, no_range, expected):
    ps = [
        guide_rust.DiverseRegexLogitsProcessor(
            regex,
            Tokenizer(),
            gamma=0.3,
            completion_bias_eta=2,
            no_range_scaling=no_range,
            eos_completion_mode=mode,
        )
        for mode in ["legacy", "entry_count"]
    ]
    for p in ps:
        walk(p, [[0]])
    result = [walk(p, [[0, 10]]) for p in ps]
    assert (result[1][0, 99] - result[0][0, 99]).item() == pytest.approx(
        expected, abs=1e-5
    )


def test_zero_range_and_incremental_prefix_catchup():
    p = guide_rust.DiverseRegexLogitsProcessor(
        "a(?:ba)*", Tokenizer(), eos_completion_mode="entry_count"
    )
    walk(p, [[0]])
    result = p(torch.tensor([[0, 10, 20, 10]]), torch.zeros(1, 100))
    assert p.accepting_entries == (2,)
    assert result[0, 99].item() == 0.0


@pytest.mark.parametrize("module", [guide_rust, guide_python])
def test_real_attempt1403_token_boundary_entries(module):
    fixture = json.loads(
        (Path(__file__).parent / "fixtures/eos_attempt1403.json").read_text()
    )
    tokenizer = Tokenizer({s: int(i) for i, s in fixture["vocab"].items()})
    tokenizer.eos_token_id = fixture["eos_token_id"]
    ps = [
        module.DiverseRegexLogitsProcessor(
            fixture["regex"],
            tokenizer,
            gamma=0.3,
            completion_bias_eta=2,
            eos_completion_mode=mode,
        )
        for mode in ["legacy", "entry_count"]
    ]
    prefix = [0]
    positions = []
    last = 0
    for index in range(len(fixture["token_ids"]) + 1):
        scores = torch.zeros(1, 151646)
        scores[0, 151645] = 1.0
        out = [p(torch.tensor([prefix]), scores.clone()) for p in ps]
        count = ps[1].accepting_entries[0]
        if count > last:
            positions.append(index)
            assert (out[1][0, 151645] - out[0][0, 151645]).item() == pytest.approx(
                0.6 * count, abs=1e-5
            )
        last = count
        if index < len(fixture["token_ids"]):
            prefix.append(fixture["token_ids"][index])
    assert positions == [19, 38, 57]


def test_probe_is_observational_and_records_last_eos(mock_tokenizer):
    from diverse_guide.eos_diagnostics import StepProbeProcessor

    reference = guide_rust.DiverseRegexLogitsProcessor(
        "a(?:ba)*",
        mock_tokenizer,
        gamma=0.3,
        completion_bias_eta=2,
        eos_completion_mode="entry_count",
    )
    observed = guide_rust.DiverseRegexLogitsProcessor(
        "a(?:ba)*",
        mock_tokenizer,
        gamma=0.3,
        completion_bias_eta=2,
        eos_completion_mode="entry_count",
    )
    records = []
    probe = StepProbeProcessor(observed, records.append)
    rng = torch.random.get_rng_state().clone()
    for prefix in [[0], [0, 10], [0, 10, 20], [0, 10, 20, 10]]:
        assert torch.equal(walk(reference, [prefix]), walk(probe, [prefix]))
    probe.finish(99)
    assert torch.equal(rng, torch.random.get_rng_state())
    assert records[-1]["selected_token"] == 99
    assert records[-1]["accepting_entries"] == 2
    assert records[-1]["eos"]["completion_bonus"] == 4.0
    assert len(records) == 4
    assert records[0]["eos"]["raw_logit"] == 99.0
    assert records[0]["eos"]["final_logit"] is None
    assert records[0]["eos"]["masked"] is True


def test_rust_only_benchmark_alternate_path(mock_tokenizer):
    from scripts.benchmark_generation_engineering import (
        build_rust_only_processor_class,
    )

    cls = build_rust_only_processor_class(guide_rust)
    ps = [
        cls(
            "a(?:ba)*",
            mock_tokenizer,
            gamma=0.3,
            completion_bias_eta=2,
            path_frequency_estimator="harmonic",
            eos_completion_mode=mode,
        )
        for mode in ["legacy", "entry_count"]
    ]
    for prefix in [[0], [0, 10], [0, 10, 20], [0, 10, 20, 10]]:
        old, new = [walk(p, [prefix]) for p in ps]
        assert torch.equal(old[:, :99], new[:, :99])
    assert ps[1].accepting_entries == (2,)
    assert (new[0, 99] - old[0, 99]).item() == pytest.approx(94.8, abs=1e-4)


def test_replayed_attempt_diagnostics():
    from diverse_guide.eos_diagnostics import trajectory

    p = guide_rust.DiverseRegexLogitsProcessor("a(?:ba)*", Tokenizer())
    result = trajectory(p.dfa, [10, 20, 10, 99], 99)
    assert result == dict(
        first_accepting_token=1,
        accepting_entries=2,
        accepting_exits=1,
        final_accepting=True,
        ever_accepting=True,
        generated_tokens=4,
        termination="eos",
        eos_opportunities={"1": 1, "2": 1},
        eos_selected={"2": 1},
    )
    lost = trajectory(p.dfa, [10, 20], 99)
    assert lost["ever_accepting"] and not lost["final_accepting"]
    assert lost["termination"] == "budget"
