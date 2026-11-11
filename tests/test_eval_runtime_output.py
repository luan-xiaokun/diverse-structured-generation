import json
import sys
from argparse import Namespace
from importlib.metadata import PackageNotFoundError

import pytest

import scripts.eval_runtime as eval_runtime
from diverse_guide import PURE_SAMPLING_DEFAULTS
from scripts.eval_runtime import (
    OutlinesRegexGenerator,
    build_runtime_result,
    make_outlines_regex_generator,
    parse_args,
    write_runtime_result,
)


def _args(**overrides):
    values = {
        "grammar": "css-color",
        "model": "test/model",
        "n": 2000,
        "max_tokens": 60,
        "top_k": None,
        "top_p": None,
        "temperature": None,
        "baseline": True,
        "baseline_backend": "internal",
        "seed": None,
        "local_files_only": False,
    }
    values.update(overrides)
    return Namespace(**values)


def test_build_and_write_runtime_result(tmp_path):
    metadata = {"timestamp_utc": "2026-04-27T00:00:00Z", "git_commit": None}
    result = build_runtime_result(
        args=_args(temperature=1.5, baseline_backend="outlines", seed=42),
        total_token_num=2031,
        total_time=123.45,
        metadata=metadata,
    )
    assert result["schema_version"] == 1
    assert result["experiment"] == "runtime"
    assert result["setting"] == "baseline"
    assert result["grammar"] == "css-color"
    assert result["parameters"]["temperature"] == 1.5
    assert result["parameters"]["baseline_backend"] == "outlines"
    assert result["parameters"]["seed"] == 42
    assert result["tokens"] == {"generated": 2031, "target": 2000}
    assert result["timing"]["seconds"] == 123.45
    assert result["timing"]["tokens_per_second"] == 2031 / 123.45

    zero = build_runtime_result(
        args=_args(baseline=False), total_token_num=0, total_time=0.0, metadata=metadata
    )
    assert zero["setting"] == "diverse"
    assert zero["parameters"]["baseline_backend"] == "internal"
    assert zero["parameters"]["seed"] is None
    assert zero["timing"]["tokens_per_second"] is None

    output_path = tmp_path / "results" / "runtime" / "css-color.json"
    write_runtime_result(output_path, result)
    assert json.loads(output_path.read_text())["tokens"]["generated"] == 2031


def test_parse_args_local_files_only_and_outlines_requires_baseline(monkeypatch):
    monkeypatch.setattr(
        sys, "argv", ["eval_runtime.py", "css-color", "--local-files-only"]
    )
    assert parse_args().local_files_only is True

    monkeypatch.setattr(
        sys, "argv", ["eval_runtime.py", "css-color", "--baseline-backend", "outlines"]
    )
    with pytest.raises(SystemExit):
        parse_args()


def test_require_outlines_version(monkeypatch):
    monkeypatch.setattr(eval_runtime, "package_version", lambda name: "1.2.12")
    eval_runtime.require_outlines_version(expected="1.2.12", group="outlines")

    def raise_missing(name):
        raise PackageNotFoundError(name)

    for fake_version in (lambda name: "0.2.3", raise_missing):
        monkeypatch.setattr(eval_runtime, "package_version", fake_version)
        with pytest.raises(RuntimeError) as exc_info:
            eval_runtime.require_outlines_version(expected="1.2.12", group="outlines")
        assert "Outlines 1.2.12" in str(exc_info.value)
        assert "uv sync --group outlines" in str(exc_info.value)


def test_outlines_regex_generator_wrappers(monkeypatch):
    calls = {}

    class FakeGenerator:
        def __call__(self, prompt, **kwargs):
            calls["prompt"] = prompt
            calls["kwargs"] = kwargs
            return "ab"

    generator = OutlinesRegexGenerator(FakeGenerator(), do_sample=True)
    assert generator("prompt", max_tokens=5) == "ab"
    assert calls == {
        "prompt": "prompt",
        "kwargs": {"do_sample": True, "max_new_tokens": 5},
    }

    calls.clear()

    class FakeOutlines:
        @staticmethod
        def from_transformers(model, tokenizer):
            calls["model"] = model
            calls["tokenizer"] = tokenizer
            return "wrapped"

        @staticmethod
        def regex(pattern):
            calls["regex"] = pattern
            return f"regex:{pattern}"

        @staticmethod
        def Generator(model, output_type):
            calls["generator_model"] = model
            calls["output_type"] = output_type
            return "generator"

    monkeypatch.setitem(sys.modules, "outlines", FakeOutlines)
    monkeypatch.setattr(
        eval_runtime,
        "require_outlines_version",
        lambda *, expected, group: None,
    )

    wrapped = make_outlines_regex_generator("model", "tokenizer", "[ab]+")

    assert isinstance(wrapped, OutlinesRegexGenerator)
    assert wrapped.generator == "generator"
    assert calls == {
        "model": "model",
        "tokenizer": "tokenizer",
        "regex": "[ab]+",
        "generator_model": "wrapped",
        "output_type": "regex:[ab]+",
    }


class _FakeModel:
    def to(self, device):
        return self

    def eval(self):
        return None


class _FakeTokenizer:
    pad_token_id = None
    eos_token_id = 9

    def encode(self, text):
        return list(text)


def _install_runtime_fakes(monkeypatch, calls=None):
    calls = {} if calls is None else calls

    def fake_model_from_pretrained(*args, **kwargs):
        calls["model_kwargs"] = kwargs
        return _FakeModel()

    def fake_tokenizer_from_pretrained(*args, **kwargs):
        calls["tokenizer_kwargs"] = kwargs
        return _FakeTokenizer()

    monkeypatch.setattr(eval_runtime.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(
        eval_runtime.AutoModelForCausalLM, "from_pretrained", fake_model_from_pretrained
    )
    monkeypatch.setattr(
        eval_runtime.AutoTokenizer, "from_pretrained", fake_tokenizer_from_pretrained
    )


def test_main_passes_pure_sampling_defaults_to_internal_baseline(monkeypatch, tmp_path):
    calls = {}
    seed_calls = []

    def fake_baseline_regex(model, tokenizer, regex, **kwargs):
        calls["kwargs"] = kwargs

        def generate(prompt, max_tokens=None):
            calls["max_tokens"] = max_tokens
            return "abcde"

        return generate

    _install_runtime_fakes(monkeypatch)
    monkeypatch.setattr(eval_runtime, "baseline_regex", fake_baseline_regex)
    monkeypatch.setattr(eval_runtime, "set_generation_seed", seed_calls.append)
    output = tmp_path / "runtime.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "eval_runtime.py",
            "css-color",
            "--baseline",
            "-n",
            "5",
            "--max-tokens",
            "2",
            "--seed",
            "42",
            "--output",
            str(output),
        ],
    )

    eval_runtime.main()

    assert calls["kwargs"] == {
        "temperature": 1.0,
        "top_k": 0,
        "top_p": 1.0,
        "repetition_penalty": 1.0,
    }
    assert calls["max_tokens"] == 2
    assert seed_calls == [42]
    saved = json.loads(output.read_text(encoding="utf-8"))
    assert saved["parameters"]["sampling"] == calls["kwargs"]


def test_main_uses_diverseguide_and_forwards_local_files_only(monkeypatch, tmp_path):
    calls = {}

    def fake_diverse_guide(model, tokenizer, regex, **kwargs):
        calls["model"] = model
        calls["tokenizer"] = tokenizer
        calls["regex"] = regex
        calls["kwargs"] = kwargs

        def generate(prompt, max_tokens=None):
            calls["prompt"] = prompt
            calls["max_tokens"] = max_tokens
            return "abcde"

        return generate

    _install_runtime_fakes(monkeypatch, calls)
    monkeypatch.setattr(eval_runtime, "DiverseGuide", fake_diverse_guide)
    output = tmp_path / "runtime.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "eval_runtime.py",
            "css-color",
            "-n",
            "5",
            "--max-tokens",
            "2",
            "--temperature",
            "1.5",
            "--local-files-only",
            "--output",
            str(output),
        ],
    )

    eval_runtime.main()

    assert calls["regex"] == eval_runtime.GRAMMAR_REGEX["css-color"]
    assert calls["kwargs"] == {**PURE_SAMPLING_DEFAULTS, "temperature": 1.5}
    assert calls["max_tokens"] == 2
    saved = json.loads(output.read_text(encoding="utf-8"))
    assert saved["setting"] == "diverse"
    assert saved["parameters"]["baseline_backend"] == "internal"
    assert calls["model_kwargs"]["local_files_only"] is True
    assert calls["tokenizer_kwargs"]["local_files_only"] is True


def test_main_passes_sampling_defaults_to_outlines_baseline(monkeypatch, tmp_path):
    calls = {}
    seed_calls = []

    def fake_outlines_generator(model, tokenizer, regex, **kwargs):
        calls["kwargs"] = kwargs

        def generate(prompt, max_tokens=None):
            calls["max_tokens"] = max_tokens
            return "abcde"

        return generate

    _install_runtime_fakes(monkeypatch)
    monkeypatch.setattr(eval_runtime, "set_generation_seed", seed_calls.append)
    monkeypatch.setattr(
        eval_runtime,
        "make_outlines_regex_generator",
        fake_outlines_generator,
    )
    output = tmp_path / "runtime.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "eval_runtime.py",
            "css-color",
            "--baseline",
            "--baseline-backend",
            "outlines",
            "-n",
            "5",
            "--max-tokens",
            "2",
            "--seed",
            "42",
            "--output",
            str(output),
        ],
    )

    eval_runtime.main()

    assert calls["kwargs"] == {
        **PURE_SAMPLING_DEFAULTS,
        "do_sample": True,
        "pad_token_id": 9,
    }
    assert calls["max_tokens"] == 2
    assert seed_calls == [42]
    saved = json.loads(output.read_text(encoding="utf-8"))
    assert saved["parameters"]["baseline_backend"] == "outlines"
    assert saved["parameters"]["seed"] == 42
