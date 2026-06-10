import json
import sys
from unittest.mock import Mock

import pytest

import scripts.generate_re as generate_re
from scripts.generate_re import GeneratedData, set_generation_seed


def test_generated_data_round_trips_seed():
    data = GeneratedData(
        grammar="css-color",
        regex="#[0-9a-fA-F]{6}",
        prompt="Give me a CSS color code.",
        model="test/model",
        max_tokens=18,
        top_k=None,
        top_p=None,
        temperature=None,
        seed=42,
        baseline_backend="internal",
        samples=["#aabbcc"],
        elapsed_seconds=12.5,
        attempted_batches=3,
        accepted_samples=1,
        rejected_samples=2,
    )

    payload = data.to_dict()
    restored = GeneratedData.from_dict(payload)

    assert payload["seed"] == 42
    assert payload["baseline_backend"] == "internal"
    assert payload["elapsed_seconds"] == 12.5
    assert restored.seed == 42
    assert restored.baseline_backend == "internal"
    assert restored.attempted_batches == 3
    assert restored.accepted_samples == 1
    assert restored.rejected_samples == 2


def test_generated_data_accepts_legacy_payload_without_seed():
    restored = GeneratedData.from_dict(
        {
            "grammar": "css-color",
            "regex": "#[0-9a-fA-F]{6}",
            "prompt": "Give me a CSS color code.",
            "model": "test/model",
            "max_tokens": 18,
            "top_k": None,
            "top_p": None,
            "temperature": None,
            "samples": ["#aabbcc"],
        }
    )

    assert restored.seed is None
    assert restored.baseline_backend == "internal"
    assert restored.elapsed_seconds is None
    assert restored.accepted_samples is None


def test_set_generation_seed_seeds_python_numpy_and_torch(monkeypatch):
    random_seed = Mock()
    numpy_seed = Mock()
    torch_manual_seed = Mock()
    torch_cuda_manual_seed_all = Mock()

    monkeypatch.setattr("scripts.generate_re.random.seed", random_seed)
    monkeypatch.setattr("scripts.generate_re.np.random.seed", numpy_seed)
    monkeypatch.setattr("scripts.generate_re.torch.manual_seed", torch_manual_seed)
    monkeypatch.setattr("scripts.generate_re.torch.cuda.is_available", lambda: True)
    monkeypatch.setattr(
        "scripts.generate_re.torch.cuda.manual_seed_all",
        torch_cuda_manual_seed_all,
    )

    set_generation_seed(42)

    random_seed.assert_called_once_with(42)
    numpy_seed.assert_called_once_with(42)
    torch_manual_seed.assert_called_once_with(42)
    torch_cuda_manual_seed_all.assert_called_once_with(42)


def test_set_generation_seed_noops_when_seed_is_none(monkeypatch):
    random_seed = Mock()
    numpy_seed = Mock()
    torch_manual_seed = Mock()

    monkeypatch.setattr("scripts.generate_re.random.seed", random_seed)
    monkeypatch.setattr("scripts.generate_re.np.random.seed", numpy_seed)
    monkeypatch.setattr("scripts.generate_re.torch.manual_seed", torch_manual_seed)

    set_generation_seed(None)

    random_seed.assert_not_called()
    numpy_seed.assert_not_called()
    torch_manual_seed.assert_not_called()


def test_parse_args_accepts_local_files_only(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        ["generate_re.py", "css-color", "--local-files-only"],
    )

    args = generate_re.parse_args()

    assert args.local_files_only is True


def test_parse_args_requires_baseline_for_outlines_backend(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        ["generate_re.py", "css-color", "--baseline-backend", "outlines"],
    )

    with pytest.raises(SystemExit):
        generate_re.parse_args()


def test_main_forwards_local_files_only_and_records_generation_metrics(
    monkeypatch, tmp_path
):
    calls = {}

    class FakeModel:
        def to(self, device):
            return self

        def eval(self):
            return None

    class FakeTokenizer:
        def encode(self, text):
            return list(text)

    class FakeGuide:
        def __init__(self):
            self.calls = 0

        def generate_batch(self, prompt, n, max_tokens):
            self.calls += 1
            if self.calls == 1:
                return ["bad", "#aabbcc"]
            return ["#112233"]

    fake_guide = FakeGuide()

    def fake_model_from_pretrained(*args, **kwargs):
        calls["model_kwargs"] = kwargs
        return FakeModel()

    def fake_tokenizer_from_pretrained(*args, **kwargs):
        calls["tokenizer_kwargs"] = kwargs
        return FakeTokenizer()

    def fake_diverse_guide(model, tokenizer, regex, **kwargs):
        calls["guide_kwargs"] = kwargs
        return fake_guide

    perf_values = iter([100.0, 101.0, 102.0, 103.0])

    monkeypatch.setattr(generate_re.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(
        generate_re.AutoModelForCausalLM,
        "from_pretrained",
        fake_model_from_pretrained,
    )
    monkeypatch.setattr(
        generate_re.AutoTokenizer,
        "from_pretrained",
        fake_tokenizer_from_pretrained,
    )
    monkeypatch.setattr(generate_re, "DiverseGuide", fake_diverse_guide)
    monkeypatch.setattr(generate_re, "perf_counter", lambda: next(perf_values))
    output = tmp_path / "samples.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "generate_re.py",
            "css-color",
            "-n",
            "2",
            "--max-tokens",
            "3",
            "--batch-size",
            "2",
            "--local-files-only",
            "--output",
            str(output),
        ],
    )

    generate_re.main()

    saved = json.loads(output.read_text(encoding="utf-8"))
    assert calls["model_kwargs"]["local_files_only"] is True
    assert calls["tokenizer_kwargs"]["local_files_only"] is True
    assert calls["guide_kwargs"]["ablation_component"] is None
    assert saved["accepted_samples"] == 2
    assert saved["rejected_samples"] == 1
    assert saved["attempted_batches"] == 2
    assert saved["elapsed_seconds"] == 3.0
