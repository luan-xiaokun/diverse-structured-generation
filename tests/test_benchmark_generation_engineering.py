import json

import pytest
import torch

from scripts import benchmark_generation_engineering as bench
from scripts import collect_engineering_benchmarks as collect


class FakeBatch(dict):
    def to(self, device):
        self["input_ids"] = self["input_ids"].to(device)
        return self


class FakeTokenizer:
    eos_token_id = 99
    pad_token_id = 99
    all_special_tokens = ["<eos>"]

    def __init__(self):
        self._vocab = {"a": 1, "b": 2, "<eos>": self.eos_token_id}

    def get_vocab(self):
        return dict(self._vocab)

    def convert_tokens_to_string(self, tokens):
        return "".join(tokens)

    def __call__(self, text, return_tensors=None):
        del return_tensors
        ids = self.encode(text, add_special_tokens=False)
        return FakeBatch({"input_ids": torch.tensor([ids], dtype=torch.long)})

    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        return [self._vocab.get(char, 1) for char in text] or [1]

    def decode(self, token_ids, skip_special_tokens=True):
        del skip_special_tokens
        inverse = {value: key for key, value in self._vocab.items()}
        values = token_ids.tolist() if hasattr(token_ids, "tolist") else list(token_ids)
        return "".join(inverse.get(int(token_id), "") for token_id in values)


class FakeModel:
    def __init__(self):
        self.device = torch.device("cpu")
        self._param = torch.nn.Parameter(torch.zeros(1))

    def to(self, device):
        self.device = torch.device(device)
        self._param = torch.nn.Parameter(torch.zeros(1, device=self.device))
        return self

    def eval(self):
        return None

    def parameters(self):
        return iter([self._param])

    def generate(self, input_ids, logits_processor=None, streamer=None, **kwargs):
        max_new_tokens = int(kwargs.get("max_new_tokens", 3))
        sequences = input_ids.clone()
        if streamer is not None:
            streamer.put(sequences)
        for _ in range(max_new_tokens):
            scores = torch.zeros(
                (sequences.shape[0], 100),
                dtype=torch.float32,
                device=sequences.device,
            )
            if logits_processor is not None:
                for processor in logits_processor:
                    scores = processor(sequences, scores)
            next_token = torch.full(
                (sequences.shape[0], 1),
                1,
                dtype=torch.long,
                device=sequences.device,
            )
            sequences = torch.cat([sequences, next_token], dim=1)
            if streamer is not None:
                streamer.put(next_token)
        return sequences


def _install_fakes(monkeypatch):
    monkeypatch.setattr(
        bench.AutoModelForCausalLM,
        "from_pretrained",
        lambda *args, **kwargs: FakeModel(),
    )
    monkeypatch.setattr(
        bench.AutoTokenizer,
        "from_pretrained",
        lambda *args, **kwargs: FakeTokenizer(),
    )


def test_resolve_diverse_guide_uses_rust_only_min_preset():
    args = bench.parse_args(["no-bomb", "--method", "diverse-guide"])

    config = bench.resolve_method_config(args)

    assert config.engine == "rust-only-legacy-min"
    assert config.path_frequency_estimator == "min"
    assert config.completion_bias_eta == 0.0


def test_benchmark_records_build_latency_vocab_dfa_and_token_latency(monkeypatch):
    _install_fakes(monkeypatch)

    result = bench.run_benchmark(
        [
            "no-bomb",
            "--model",
            "fake/model",
            "--method",
            "diverse-guide",
            "--runs",
            "2",
            "--warmup-runs",
            "1",
            "--max-tokens",
            "3",
        ]
    )

    assert result["method"] == "diverse-guide"
    assert result["method_config"]["python_fallback_used"] is False
    assert result["tokenizer"]["vocab_size"] == 3
    assert result["grammar_dfa"]["state_count"] > 0
    assert len(result["runs"]) == 2
    assert result["runs"][0]["method_dfa"]["token_transition_count"] > 0
    assert result["runs"][0]["generated_token_count"] == 3
    assert result["summary"]["build_seconds"]["mean"] >= 0.0
    assert result["summary"]["avg_inter_token_latency_ms"]["mean"] >= 0.0


def test_collect_engineering_benchmark_summary(tmp_path):
    raw = tmp_path / "raw" / "method" / "model" / "grammar.json"
    raw.parent.mkdir(parents=True)
    payload = {
        "experiment": "generation_engineering_benchmark",
        "grammar": "no-bomb",
        "model": "fake/model",
        "method": "diverse-guide",
        "method_config": {
            "engine": "rust-only-legacy-min",
            "gamma": 0.16666666666666666,
            "completion_bias_eta": 0.0,
            "path_frequency_estimator": "min",
        },
        "parameters": {"runs": 2, "max_tokens": 3},
        "tokenizer": {"vocab_size": 3, "score_width": 100},
        "grammar_dfa": {"state_count": 4, "byte_transition_count": 7},
        "model_loading": {
            "model_load_seconds": 1.0,
            "tokenizer_load_seconds": 0.5,
            "model_parameter_device": "cpu",
        },
        "runs": [{"method_dfa": {"state_count": 4, "byte_transition_count": 7}}],
        "summary": {
            "build_seconds": {"mean": 0.1, "std": 0.01},
            "generation_seconds": {"mean": 0.2},
            "tokens_per_second": {"mean": 12.0, "std": 1.0},
            "ttft_ms": {"mean": 3.0, "std": 0.5},
            "avg_inter_token_latency_ms": {"mean": 2.0, "std": 0.2},
        },
    }
    raw.write_text(json.dumps(payload), encoding="utf-8")

    rows = collect.collect(tmp_path / "raw")

    assert rows[0]["method"] == "diverse-guide"
    assert rows[0]["build_seconds_mean"] == pytest.approx(0.1)
    assert rows[0]["avg_inter_token_latency_ms_mean"] == pytest.approx(2.0)
