import json
import sys
from argparse import Namespace
from importlib.metadata import PackageNotFoundError

import pytest

import scripts.eval_runtime as eval_runtime
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
    }
    values.update(overrides)
    return Namespace(**values)


def test_build_runtime_result_includes_timing_and_context():
    result = build_runtime_result(
        args=_args(temperature=1.5),
        total_token_num=2031,
        total_time=123.45,
        metadata={"timestamp_utc": "2026-04-27T00:00:00Z", "git_commit": None},
    )

    assert result["schema_version"] == 1
    assert result["experiment"] == "runtime"
    assert result["setting"] == "baseline"
    assert result["grammar"] == "css-color"
    assert result["parameters"]["temperature"] == 1.5
    assert result["parameters"]["baseline_backend"] == "internal"
    assert result["parameters"]["seed"] is None
    assert result["tokens"] == {"generated": 2031, "target": 2000}
    assert result["timing"]["seconds"] == 123.45
    assert result["timing"]["tokens_per_second"] == 2031 / 123.45


def test_build_runtime_result_records_outlines_baseline_backend():
    result = build_runtime_result(
        args=_args(baseline_backend="outlines"),
        total_token_num=100,
        total_time=10.0,
        metadata={"timestamp_utc": "2026-04-27T00:00:00Z", "git_commit": None},
    )

    assert result["parameters"]["baseline_backend"] == "outlines"


def test_build_runtime_result_records_seed():
    result = build_runtime_result(
        args=_args(seed=42),
        total_token_num=100,
        total_time=10.0,
        metadata={"timestamp_utc": "2026-04-27T00:00:00Z", "git_commit": None},
    )

    assert result["parameters"]["seed"] == 42


def test_parse_args_requires_baseline_for_outlines_backend(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        ["eval_runtime.py", "css-color", "--baseline-backend", "outlines"],
    )

    with pytest.raises(SystemExit):
        parse_args()


def test_require_outlines_version_accepts_expected_version(monkeypatch):
    monkeypatch.setattr(eval_runtime, "package_version", lambda name: "1.2.12")

    eval_runtime.require_outlines_version(expected="1.2.12", group="outlines")


def test_require_outlines_version_rejects_wrong_version(monkeypatch):
    monkeypatch.setattr(eval_runtime, "package_version", lambda name: "0.2.3")

    with pytest.raises(RuntimeError) as exc_info:
        eval_runtime.require_outlines_version(expected="1.2.12", group="outlines")

    assert "Outlines 1.2.12" in str(exc_info.value)
    assert "uv sync --group outlines" in str(exc_info.value)


def test_require_outlines_version_reports_missing_package(monkeypatch):
    def raise_missing(name):
        raise PackageNotFoundError(name)

    monkeypatch.setattr(eval_runtime, "package_version", raise_missing)

    with pytest.raises(RuntimeError) as exc_info:
        eval_runtime.require_outlines_version(expected="1.2.12", group="outlines")

    assert "Outlines 1.2.12" in str(exc_info.value)
    assert "uv sync --group outlines" in str(exc_info.value)


def test_outlines_generator_reuses_prebuilt_generator():
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


def test_make_outlines_regex_generator_wraps_transformers_model(monkeypatch):
    calls = {}

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

    generator = make_outlines_regex_generator("model", "tokenizer", "[ab]+")

    assert isinstance(generator, OutlinesRegexGenerator)
    assert generator.generator == "generator"
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


def _install_runtime_fakes(monkeypatch, generator_factory):
    monkeypatch.setattr(eval_runtime.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(
        eval_runtime.AutoModelForCausalLM,
        "from_pretrained",
        lambda *args, **kwargs: _FakeModel(),
    )
    monkeypatch.setattr(
        eval_runtime.AutoTokenizer,
        "from_pretrained",
        lambda *args, **kwargs: _FakeTokenizer(),
    )
    return generator_factory


def test_main_keeps_internal_baseline_generation_kwargs_unchanged(
    monkeypatch, tmp_path
):
    calls = {}
    seed_calls = []

    def fake_baseline_regex(model, tokenizer, regex, **kwargs):
        calls["kwargs"] = kwargs

        def generate(prompt, max_tokens=None):
            calls["max_tokens"] = max_tokens
            return "abcde"

        return generate

    _install_runtime_fakes(monkeypatch, fake_baseline_regex)
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

    assert calls["kwargs"] == {}
    assert calls["max_tokens"] == 2
    assert seed_calls == [42]


def test_main_passes_sampling_defaults_to_outlines_baseline(monkeypatch, tmp_path):
    calls = {}

    def fake_outlines_generator(model, tokenizer, regex, **kwargs):
        calls["kwargs"] = kwargs

        def generate(prompt, max_tokens=None):
            calls["max_tokens"] = max_tokens
            return "abcde"

        return generate

    _install_runtime_fakes(monkeypatch, fake_outlines_generator)
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
            "--output",
            str(output),
        ],
    )

    eval_runtime.main()

    assert calls["kwargs"] == {"do_sample": True, "pad_token_id": 9}
    assert calls["max_tokens"] == 2


def test_build_runtime_result_handles_zero_time():
    result = build_runtime_result(
        args=_args(baseline=False),
        total_token_num=0,
        total_time=0.0,
        metadata={"timestamp_utc": "2026-04-27T00:00:00Z", "git_commit": None},
    )

    assert result["setting"] == "diverse"
    assert result["timing"]["tokens_per_second"] is None


def test_write_runtime_result_creates_json(tmp_path):
    output_path = tmp_path / "results" / "runtime" / "css-color.json"
    result = build_runtime_result(
        args=_args(),
        total_token_num=2000,
        total_time=100.0,
        metadata={"timestamp_utc": "2026-04-27T00:00:00Z", "git_commit": None},
    )

    write_runtime_result(output_path, result)

    with output_path.open() as f:
        saved = json.load(f)
    assert saved["timing"]["tokens_per_second"] == 20.0
