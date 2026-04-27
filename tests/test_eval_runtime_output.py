import json
from argparse import Namespace

from scripts.eval_runtime import build_runtime_result, write_runtime_result


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
    assert result["tokens"] == {"generated": 2031, "target": 2000}
    assert result["timing"]["seconds"] == 123.45
    assert result["timing"]["tokens_per_second"] == 2031 / 123.45


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
