import csv
import json
import sys

import pytest

from scripts.collect_results import MissingResultInputsError, collect_all, main


def _write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f)


def _metrics_result(experiment, setting, grammar):
    return {
        "schema_version": 1,
        "experiment": experiment,
        "setting": setting,
        "grammar": grammar,
        "model": "test/model",
        "sample_count": 2,
        "average_length": 1.5,
        "dfa": {"state_count": 3, "transition_count": 4},
        "metrics": {
            "state_coverage": 1.0,
            "transition_coverage": 0.5,
            "path_coverage": 0.25,
            "distinct_2gram": [1, 2],
            "distinct_3gram": [0, 2],
            "vendi_score": 1.75,
            "average_perplexity": None,
            "perplexity_count": 0,
            "perplexity_error_count": 0,
        },
        "parameters": {"temperature": None, "ablation_component": None},
        "metadata": {"timestamp_utc": "2026-04-27T00:00:00Z"},
    }


def _runtime_result(setting, grammar):
    return {
        "schema_version": 1,
        "experiment": "runtime",
        "setting": setting,
        "grammar": grammar,
        "model": "test/model",
        "parameters": {"n": 2000, "max_tokens": 60, "temperature": None},
        "tokens": {"generated": 2000, "target": 2000},
        "timing": {"seconds": 100.0, "tokens_per_second": 20.0},
        "metadata": {"timestamp_utc": "2026-04-27T00:00:00Z"},
    }


def test_collect_all_writes_csv_tables(tmp_path):
    results_dir = tmp_path / "results"
    _write_json(
        results_dir / "diversity" / "diverse" / "css-color.json",
        _metrics_result("diversity", "diverse", "css-color"),
    )
    _write_json(
        results_dir / "runtime" / "baseline" / "css-color.json",
        _runtime_result("baseline", "css-color"),
    )
    _write_json(
        results_dir
        / "temperature_ablation"
        / "diverse"
        / "temperature-1.5"
        / "css-color.json",
        _metrics_result("temperature_ablation", "diverse", "css-color"),
    )
    _write_json(
        results_dir / "component_ablation" / "default" / "css-color.json",
        _metrics_result("component_ablation", "diverse", "css-color"),
    )

    collect_all(results_dir)

    diversity_csv = results_dir / "tables" / "diversity.csv"
    runtime_csv = results_dir / "tables" / "runtime.csv"
    assert diversity_csv.exists()
    assert runtime_csv.exists()
    with diversity_csv.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert rows[0]["experiment"] == "diversity"
    assert rows[0]["setting"] == "diverse"
    assert rows[0]["grammar"] == "css-color"
    assert rows[0]["state_coverage"] == "1.0"
    with runtime_csv.open(encoding="utf-8") as f:
        runtime_rows = list(csv.DictReader(f))
    assert runtime_rows[0]["tokens_per_second"] == "20.0"


def test_collect_all_reports_all_missing_or_empty_inputs_without_writing_tables(
    tmp_path,
):
    results_dir = tmp_path / "results"
    (results_dir / "temperature_ablation").mkdir(parents=True)
    _write_json(
        results_dir / "diversity" / "diverse" / "css-color.json",
        _metrics_result("diversity", "diverse", "css-color"),
    )

    with pytest.raises(MissingResultInputsError) as exc_info:
        collect_all(results_dir)

    message = str(exc_info.value)
    assert "Missing result inputs:" in message
    assert f"- {results_dir / 'temperature_ablation'} (no JSON result files)" in message
    assert f"- {results_dir / 'component_ablation'} (missing directory)" in message
    assert f"- {results_dir / 'runtime'} (missing directory)" in message
    assert not (results_dir / "tables").exists()


def test_main_prints_clean_preflight_error_without_traceback(
    tmp_path, monkeypatch, capsys
):
    results_dir = tmp_path / "results"
    (results_dir / "diversity").mkdir(parents=True)
    monkeypatch.setattr(
        sys,
        "argv",
        ["collect_results.py", "--results-dir", str(results_dir)],
    )

    with pytest.raises(SystemExit) as exc_info:
        main()

    assert exc_info.value.code == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "Missing result inputs:" in captured.err
    assert "diversity" in captured.err
    assert "(no JSON result files)" in captured.err
    assert "runtime" in captured.err
    assert "(missing directory)" in captured.err
    assert "Traceback" not in captured.err
    assert not (results_dir / "tables").exists()
