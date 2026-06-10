import json
from argparse import Namespace

import scripts.profile_regex_processor as profile_regex_processor
from tests.conftest import MockTokenizer


def test_parse_args_requires_positive_repeat(tmp_path):
    args = profile_regex_processor.parse_args(
        [
            "css-color",
            "--repeat",
            "2",
            "--output",
            str(tmp_path / "profile.json"),
        ]
    )

    assert args.grammar == "css-color"
    assert args.repeat == 2


def test_parse_args_requires_positive_runs(tmp_path):
    args = profile_regex_processor.parse_args(
        [
            "css-color",
            "--runs",
            "2",
            "--output",
            str(tmp_path / "profile.json"),
        ]
    )

    assert args.runs == 2


def test_run_profile_with_mock_tokenizer(monkeypatch):
    monkeypatch.setattr(profile_regex_processor, "GRAMMAR_REGEX", {"toy": "[ab]+"})
    args = Namespace(
        grammar="toy",
        model="fake",
        prefixes=["", "a"],
        repeat=1,
        runs=1,
        output=None,
        local_files_only=True,
    )

    result = profile_regex_processor.run_profile(args, tokenizer=MockTokenizer())

    assert result["schema_version"] == 1
    assert result["experiment"] == "regex_processor_profile"
    assert result["grammar"] == "toy"
    assert result["vocab_size"] == 3
    assert result["score_width"] == 100
    assert result["timing"]["guide_build_seconds"] >= 0.0
    assert len(result["prefixes"]) == 2

    first = result["prefixes"][0]
    assert first["prefix"] == ""
    assert first["allowed_count"] >= 1
    assert first["compute_counts_ms"] >= 0.0
    assert first["mask_processor_ms"] >= 0.0
    assert first["diverse_processor_ms"] >= 0.0
    assert first["error"] is None


def test_run_profile_rejects_non_positive_runs(tmp_path):
    try:
        profile_regex_processor.parse_args(
            [
                "css-color",
                "--runs",
                "0",
                "--output",
                str(tmp_path / "profile.json"),
            ]
        )
    except SystemExit:
        return

    raise AssertionError("expected parse_args() to reject --runs 0")


def test_run_profile_includes_aggregates_for_multiple_runs(monkeypatch):
    monkeypatch.setattr(profile_regex_processor, "GRAMMAR_REGEX", {"toy": "[ab]+"})
    args = Namespace(
        grammar="toy",
        model="fake",
        prefixes=["", "a"],
        repeat=1,
        runs=2,
        output=None,
        local_files_only=True,
    )

    result = profile_regex_processor.run_profile(args, tokenizer=MockTokenizer())

    assert len(result["runs"]) == 2
    assert result["runs"][0]["run_index"] == 0
    assert result["runs"][1]["run_index"] == 1
    assert "aggregates" in result
    assert result["aggregates"]["timing"]["guide_build_seconds"]["count"] == 2
    assert result["aggregates"]["prefixes"][0]["prefix"] == ""
    assert (
        result["aggregates"]["prefixes"][0]["metrics"]["compute_counts_ms"]["count"]
        == 2
    )


def test_main_writes_profile_json(tmp_path, monkeypatch):
    output = tmp_path / "profile.json"
    monkeypatch.setattr(profile_regex_processor, "GRAMMAR_REGEX", {"toy": "[ab]+"})
    monkeypatch.setattr(
        profile_regex_processor.AutoTokenizer,
        "from_pretrained",
        lambda *args, **kwargs: MockTokenizer(),
    )

    result = profile_regex_processor.main(
        [
            "toy",
            "--prefixes",
            "",
            "a",
            "--repeat",
            "1",
            "--output",
            str(output),
            "--local-files-only",
        ]
    )

    saved = json.loads(output.read_text())
    assert saved["experiment"] == "regex_processor_profile"
    assert saved["prefixes"][1]["prefix"] == "a"
    assert result == saved


def test_main_writes_multi_run_profile_json(tmp_path, monkeypatch):
    output = tmp_path / "profile.json"
    monkeypatch.setattr(profile_regex_processor, "GRAMMAR_REGEX", {"toy": "[ab]+"})
    monkeypatch.setattr(
        profile_regex_processor.AutoTokenizer,
        "from_pretrained",
        lambda *args, **kwargs: MockTokenizer(),
    )

    result = profile_regex_processor.main(
        [
            "toy",
            "--prefixes",
            "",
            "a",
            "--repeat",
            "1",
            "--runs",
            "2",
            "--output",
            str(output),
            "--local-files-only",
        ]
    )

    saved = json.loads(output.read_text())
    assert len(saved["runs"]) == 2
    assert "aggregates" in saved
    assert result == saved
