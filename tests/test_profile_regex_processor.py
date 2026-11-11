import json

import pytest

import scripts.profile_regex_processor as profile_regex_processor
from tests.conftest import MockTokenizer


def test_parse_args_validates_repeat_and_runs(tmp_path):
    output = str(tmp_path / "profile.json")
    args = profile_regex_processor.parse_args(
        ["css-color", "--repeat", "2", "--runs", "2", "--output", output]
    )
    assert args.grammar == "css-color"
    assert args.repeat == 2
    assert args.runs == 2

    with pytest.raises(SystemExit):
        profile_regex_processor.parse_args(
            ["css-color", "--runs", "0", "--output", output]
        )


def test_main_writes_single_and_multi_run_profiles(tmp_path, monkeypatch):
    monkeypatch.setattr(profile_regex_processor, "GRAMMAR_REGEX", {"toy": "[ab]+"})
    monkeypatch.setattr(
        profile_regex_processor.AutoTokenizer,
        "from_pretrained",
        lambda *args, **kwargs: MockTokenizer(),
    )
    base_argv = ["toy", "--prefixes", "", "a", "--repeat", "1", "--local-files-only"]

    single_output = tmp_path / "profile.json"
    result = profile_regex_processor.main([*base_argv, "--output", str(single_output)])
    saved = json.loads(single_output.read_text())
    assert result == saved
    assert saved["schema_version"] == 1
    assert saved["experiment"] == "regex_processor_profile"
    assert saved["grammar"] == "toy"
    assert saved["vocab_size"] == 3
    assert saved["score_width"] == 100
    assert saved["timing"]["guide_build_seconds"] >= 0.0
    assert [p["prefix"] for p in saved["prefixes"]] == ["", "a"]
    first = saved["prefixes"][0]
    assert first["allowed_count"] >= 1
    assert first["compute_counts_ms"] >= 0.0
    assert first["mask_processor_ms"] >= 0.0
    assert first["diverse_processor_ms"] >= 0.0
    assert first["error"] is None

    multi_output = tmp_path / "profile_runs.json"
    result = profile_regex_processor.main(
        [*base_argv, "--runs", "2", "--output", str(multi_output)]
    )
    saved = json.loads(multi_output.read_text())
    assert result == saved
    assert [run["run_index"] for run in saved["runs"]] == [0, 1]
    aggregates = saved["aggregates"]
    assert aggregates["timing"]["guide_build_seconds"]["count"] == 2
    assert aggregates["prefixes"][0]["prefix"] == ""
    assert aggregates["prefixes"][0]["metrics"]["compute_counts_ms"]["count"] == 2
